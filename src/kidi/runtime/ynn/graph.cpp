#include "kidi/runtime/ynn/graph.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "kidi/runtime/ynn/numa.h"
#include "kidi/runtime/ynn/worker_pool.h"
#include "ynnpack/base/arch.h"

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace kidi::runtime::ynn {
namespace {

constexpr ynn_scheduler YNN_SCHEDULER{
    .num_threads = [](void* context) -> int { return static_cast<WorkerPool*>(context)->thread_count(); },
    .schedule = [](void* context, void* task_context, void (*task)(void*)) -> void {
        static_cast<WorkerPool*>(context)->schedule_async(task_context, task);
    },
};

} // namespace

class ThreadPool {
public:
    static auto create(std::size_t total_threads) -> Result<std::shared_ptr<ThreadPool>> {
        auto result = std::shared_ptr<ThreadPool>(new ThreadPool(static_cast<int>(total_threads - 1)));
        return result;
    }

    ~ThreadPool() {
        if (ynn_handle_ != nullptr) ynn_delete_threadpool(ynn_handle_);
    }

    auto ynn_handle() -> Result<ynn_threadpool_t> {
        std::scoped_lock lock(adapter_mutex_);
        if (ynn_handle_ != nullptr) return ynn_handle_;
        if (pool_->thread_count() <= 0) return nullptr;
        ynn_threadpool_t created = nullptr;
        if (auto status = check_status(ynn_create_threadpool(&YNN_SCHEDULER, pool_.get(), 0, &created),
                                       "create YNNPACK threadpool adapter");
            !status) {
            return std::unexpected(std::move(status.error()));
        }
        ynn_handle_ = created;
        return ynn_handle_;
    }

    auto worker_pool() const noexcept -> WorkerPool* { return pool_.get(); }
    auto parallel_for(std::size_t size, const std::function<void(std::size_t)>& body) -> void {
        pool_->parallel_for(size, body);
    }
    auto run_tasks(std::size_t size, void* context, void (*body)(void*, std::size_t)) -> void {
        pool_->run_tasks(size, context, body);
    }
    auto run_ranges(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes, void* context,
                    void (*body)(void*, const WorkerContext&, std::size_t, std::size_t)) -> void {
        pool_->run_ranges(items, min_grain, scratch_bytes, context, body);
    }
    auto run_team(std::size_t max_threads, std::size_t scratch_bytes, void* context,
                  void (*body)(void*, const WorkerContext&, PhaseBarrier&)) -> void {
        pool_->run_team(max_threads, scratch_bytes, context, body);
    }

private:
    explicit ThreadPool(int worker_count) : pool_(std::make_unique<WorkerPool>(worker_count)) {}

    std::unique_ptr<WorkerPool> pool_;
    std::mutex adapter_mutex_;
    ynn_threadpool_t ynn_handle_ = nullptr;
};

namespace {

std::atomic<std::size_t> configured_thread_count = 0;

struct CachedPool {
    std::size_t threads = 0;
    std::shared_ptr<ThreadPool> pool;
};

auto ensure_interactive_qos() noexcept -> void {
#if defined(__APPLE__)
    thread_local bool configured = false;
    if (!configured) {
        if (::pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0)
            (void)::pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
        configured = true;
    }
#endif
}

auto borrow_thread_pool(std::size_t total_threads) -> Result<ThreadPool*> {
    struct PoolCache {
        std::mutex mutex;
        std::unordered_map<std::size_t, std::shared_ptr<ThreadPool>> pools;
    };
    static PoolCache cache;

    if (total_threads == 0) total_threads = thread_count();
    if (total_threads == 0 || total_threads > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "CPU thread count exceeds supported range"});

    ensure_interactive_qos();
    thread_local CachedPool local;
    if (local.pool && local.threads == total_threads) return local.pool.get();

    std::scoped_lock lock(cache.mutex);
    if (auto found = cache.pools.find(total_threads); found != cache.pools.end()) {
        if (total_threads > 1 || !local.pool) local = {total_threads, found->second};
        return found->second.get();
    }
    auto created = ThreadPool::create(total_threads);
    if (!created) return std::unexpected(std::move(created.error()));
    auto* raw = created->get();
    cache.pools.emplace(total_threads, *created);
    if (total_threads > 1 || !local.pool) local = {total_threads, std::move(*created)};
    return raw;
}

auto default_thread_pool(std::size_t total_threads) -> Result<std::shared_ptr<ThreadPool>> {
    if (total_threads == 0) total_threads = thread_count();
    if (total_threads == 0 || total_threads > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "CPU thread count exceeds supported range"});
    if (total_threads == 1) return std::shared_ptr<ThreadPool>{};
    auto borrowed = borrow_thread_pool(total_threads);
    if (!borrowed) return std::unexpected(std::move(borrowed.error()));
    // PoolCache retains strong ownership for the lifetime of the process; wrap with a no-op deleter when needed,
    // or return a non-owning shared_ptr alias so Executable never destroys global worker threads.
    return std::shared_ptr<ThreadPool>(std::shared_ptr<ThreadPool>{}, *borrowed);
}

} // namespace

auto set_thread_count(std::size_t total_threads) noexcept -> void {
    ensure_interactive_qos();
    configured_thread_count.store(total_threads, std::memory_order_relaxed);
}

auto thread_count() noexcept -> std::size_t {
    const auto configured = configured_thread_count.load(std::memory_order_relaxed);
    return configured == 0 ? default_thread_count() : configured;
}

auto reserve_thread_pool(std::size_t total_threads) -> Result<void> {
    if (total_threads == 0) total_threads = thread_count();
    if (total_threads <= 1) return {};
    auto pool = borrow_thread_pool(total_threads);
    if (!pool) return std::unexpected(std::move(pool.error()));
    return {};
}

auto parallel_for(std::size_t size, std::function<void(std::size_t)> body) -> Result<void> {
    if (size == 0) return {};
    const auto threads = thread_count();
    if (size == 1 || threads <= 1) {
        for (std::size_t index = 0; index < size; ++index) body(index);
        return {};
    }
    auto pool = borrow_thread_pool(threads);
    if (!pool) return std::unexpected(std::move(pool.error()));
    (*pool)->parallel_for(size, body);
    return {};
}

auto parallel_tasks(std::size_t tasks, void* context, void (*body)(void*, std::size_t)) -> Result<void> {
    if (tasks == 0) return {};
    const auto threads = thread_count();
    if (tasks == 1 || threads <= 1) {
        for (std::size_t index = 0; index < tasks; ++index) body(context, index);
        return {};
    }
    auto pool = borrow_thread_pool(threads);
    if (!pool) return std::unexpected(std::move(pool.error()));
    (*pool)->run_tasks(tasks, context, body);
    return {};
}

auto parallel_ranges(std::size_t items, std::size_t min_grain, void* context,
                     void (*body)(void*, std::size_t, std::size_t)) -> Result<void> {
    struct Payload {
        void* ctx;
        void (*fn)(void*, std::size_t, std::size_t);
    } payload{context, body};
    return parallel_ranges_with_worker(items, min_grain, 0, &payload,
                                       [](void* raw, const WorkerContext&, std::size_t begin, std::size_t end) {
                                           const auto p = *static_cast<const Payload*>(raw);
                                           p.fn(p.ctx, begin, end);
                                       });
}

auto parallel_ranges_with_worker(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes, void* context,
                                 void (*body)(void*, const WorkerContext&, std::size_t, std::size_t)) -> Result<void> {
    if (items == 0) return {};
    min_grain = std::max<std::size_t>(1, min_grain);
    const auto threads = thread_count();
    auto pool = borrow_thread_pool(threads);
    if (!pool) return std::unexpected(std::move(pool.error()));
    (*pool)->run_ranges(items, min_grain, scratch_bytes, context, body);
    return {};
}

auto parallel_team(std::size_t max_threads, std::size_t scratch_bytes, void* context,
                   void (*body)(void*, const WorkerContext&, PhaseBarrier&)) -> Result<void> {
    if (max_threads == 0) return {};
    const auto threads = thread_count();
    const auto team = std::min(max_threads, threads);
    auto pool = borrow_thread_pool(threads);
    if (!pool) return std::unexpected(std::move(pool.error()));
    (*pool)->run_team(team, scratch_bytes, context, body);
    return {};
}

auto supported_arch_flags() noexcept -> std::uint64_t { return ::ynn::get_supported_arch_flags(); }

auto supported_arch_names() -> std::string {
    const auto flags = supported_arch_flags();
    std::string result;
    const auto append = [&](std::uint64_t flag, const char* name) {
        if ((flags & flag) == 0) return;
        if (!result.empty()) result += ',';
        result += name;
    };
#ifdef YNN_ARCH_ARM
    append(::ynn::arch_flag::neon, "neon");
    append(::ynn::arch_flag::neonfma, "fma");
    append(::ynn::arch_flag::neondot, "dotprod");
    append(::ynn::arch_flag::neonfp16, "fp16");
    append(::ynn::arch_flag::neonfp16arith, "fp16arith");
    append(::ynn::arch_flag::neonbf16, "bf16");
    append(::ynn::arch_flag::neonfp8, "fp8");
    append(::ynn::arch_flag::neonfp8dot4, "fp8dot4");
    append(::ynn::arch_flag::neoni8mm, "i8mm");
    append(::ynn::arch_flag::sme, "sme");
    append(::ynn::arch_flag::sme2, "sme2");
    append(::ynn::arch_flag::sve, "sve");
#endif
    return result.empty() ? "generic" : result;
}

auto check_status(ynn_status status, const char* operation) -> Result<void> {
    if (status == ynn_status_success) {
        return {};
    }
    return std::unexpected(Error{
        ErrorCode::RUNTIME,
        std::string(operation) + " failed with YNNPACK status " + std::to_string(status),
    });
}

Graph::Graph(ynn_subgraph_t graph) : graph_(graph) {}

Graph::Graph(Graph&& other) noexcept : graph_(other.release()) {}

auto Graph::operator=(Graph&& other) noexcept -> Graph& {
    if (this != &other) {
        if (graph_ != nullptr) {
            ynn_delete_subgraph(graph_);
        }
        graph_ = other.release();
    }
    return *this;
}

Graph::~Graph() {
    if (graph_ != nullptr) {
        ynn_delete_subgraph(graph_);
    }
}

auto Graph::create(std::uint32_t external_value_count, std::uint32_t flags) -> Result<Graph> {
    ynn_subgraph_t graph = nullptr;
    if (auto status = check_status(ynn_create_subgraph(external_value_count, flags, &graph), "create graph"); !status) {
        return std::unexpected(std::move(status.error()));
    }
    return Graph(graph);
}

auto Graph::get() const noexcept -> ynn_subgraph_t { return graph_; }

auto Graph::release() noexcept -> ynn_subgraph_t { return std::exchange(graph_, nullptr); }

auto Graph::compile(std::size_t total_threads) && -> Result<Executable> {
    auto thread_pool = default_thread_pool(total_threads);
    if (!thread_pool) return std::unexpected(std::move(thread_pool.error()));
    ynn_threadpool_t handle = nullptr;
    if (*thread_pool) {
        auto resolved = (*thread_pool)->ynn_handle();
        if (!resolved) return std::unexpected(std::move(resolved.error()));
        handle = *resolved;
    }
    if (auto status = check_status(ynn_optimize_subgraph(graph_, handle, 0), "optimize graph"); !status) {
        return std::unexpected(std::move(status.error()));
    }
    ynn_runtime_t runtime = nullptr;
    if (auto status = check_status(ynn_create_runtime(graph_, handle, 0, &runtime), "create runtime"); !status) {
        return std::unexpected(std::move(status.error()));
    }
    return Executable(release(), runtime, std::move(*thread_pool));
}

Executable::Executable(ynn_subgraph_t graph, ynn_runtime_t runtime, std::shared_ptr<ThreadPool> thread_pool)
    : graph_(graph), runtime_(runtime), thread_pool_(std::move(thread_pool)) {}

Executable::Executable(Executable&& other) noexcept
    : graph_(std::exchange(other.graph_, nullptr)),
      runtime_(std::exchange(other.runtime_, nullptr)),
      thread_pool_(std::move(other.thread_pool_)) {}

auto Executable::operator=(Executable&& other) noexcept -> Executable& {
    if (this != &other) {
        if (runtime_ != nullptr) ynn_delete_runtime(runtime_);
        if (graph_ != nullptr) ynn_delete_subgraph(graph_);
        graph_ = std::exchange(other.graph_, nullptr);
        runtime_ = std::exchange(other.runtime_, nullptr);
        thread_pool_ = std::move(other.thread_pool_);
    }
    return *this;
}

Executable::~Executable() {
    if (runtime_ != nullptr) ynn_delete_runtime(runtime_);
    if (graph_ != nullptr) ynn_delete_subgraph(graph_);
}

auto Executable::reshape() -> Result<void> { return check_status(ynn_reshape_runtime(runtime_), "reshape runtime"); }

auto Executable::bind(std::uint32_t external_id, void* data) -> Result<void> {
    return check_status(ynn_set_external_value_data(runtime_, external_id, data), "bind external value");
}

auto Executable::bind(std::uint32_t external_id, tensor::Tensor& tensor) -> Result<void> {
    if (!tensor.defined() || tensor.device().kind != tensor::DeviceKind::CPU || !tensor.is_contiguous()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "YNNPACK binding requires a contiguous CPU tensor"});
    }
    auto bytes = tensor.host_bytes();
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    return bind(external_id, bytes->data());
}

auto Executable::bind(std::uint32_t external_id, const tensor::Tensor& tensor) -> Result<void> {
    if (!tensor.defined() || tensor.device().kind != tensor::DeviceKind::CPU || !tensor.is_contiguous()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "YNNPACK binding requires a contiguous CPU tensor"});
    }
    auto bytes = tensor.host_bytes();
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    return bind(external_id, const_cast<std::byte*>(bytes->data()));
}

auto Executable::invoke() -> Result<void> { return check_status(ynn_invoke_runtime(runtime_), "invoke runtime"); }

auto Executable::shape(std::uint32_t external_id) const -> Result<std::vector<std::size_t>> {
    std::array<std::size_t, YNN_MAX_TENSOR_RANK> dimensions{};
    std::size_t rank = dimensions.size();
    if (auto status = check_status(ynn_get_external_value_shape(runtime_, external_id, &rank, dimensions.data()),
                                   "get external shape");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    return std::vector<std::size_t>(dimensions.begin(), dimensions.begin() + static_cast<std::ptrdiff_t>(rank));
}

} // namespace kidi::runtime::ynn