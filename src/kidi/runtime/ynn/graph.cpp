#include "kidi/runtime/ynn/graph.h"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "slinky/base/thread_pool_impl.h"
#include "ynnpack/base/arch.h"

namespace kidi::runtime::ynn {
namespace {

class Scheduler {
public:
    explicit Scheduler(int worker_count) : implementation_(worker_count) {}

    ~Scheduler() { implementation_.work_until_idle(); }

    static auto interface() noexcept -> const ynn_scheduler* {
        static const ynn_scheduler value = {
            .num_threads =
                [](void* context) { return static_cast<Scheduler*>(context)->implementation_.thread_count(); },
            .schedule =
                [](void* context, void* task_context, void (*task)(void*)) {
                    auto& scheduler = *static_cast<Scheduler*>(context);
                    scheduler.implementation_.enqueue([task_context, task] { task(task_context); });
                },
        };
        return &value;
    }

private:
    slinky::thread_pool_impl implementation_;
};

} // namespace

class ThreadPool {
public:
    static auto create(std::size_t total_threads) -> Result<std::shared_ptr<ThreadPool>> {
        auto result = std::shared_ptr<ThreadPool>(new ThreadPool(static_cast<int>(total_threads - 1)));
        auto status =
            check_status(ynn_create_threadpool(Scheduler::interface(), &result->scheduler_, 0, &result->handle_),
                         "create thread pool");
        if (!status) return std::unexpected(std::move(status.error()));
        return result;
    }

    ~ThreadPool() {
        if (handle_ != nullptr) ynn_delete_threadpool(handle_);
    }

    auto get() const noexcept -> ynn_threadpool_t { return handle_; }

private:
    explicit ThreadPool(int worker_count) : scheduler_(worker_count) {}

    Scheduler scheduler_;
    ynn_threadpool_t handle_ = nullptr;
};

namespace {

std::atomic<std::size_t> configured_thread_count = 0;

auto default_thread_pool(std::size_t total_threads) -> Result<std::shared_ptr<ThreadPool>> {
    struct PoolCache {
        std::mutex mutex;
        std::unordered_map<std::size_t, std::weak_ptr<ThreadPool>> pools;
    };
    static PoolCache cache;

    if (total_threads == 0) total_threads = thread_count();
    if (total_threads > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "CPU thread count exceeds supported range"});
    std::scoped_lock lock(cache.mutex);
    if (auto existing = cache.pools[total_threads].lock()) return existing;
    auto created = ThreadPool::create(total_threads);
    if (!created) return std::unexpected(std::move(created.error()));
    cache.pools[total_threads] = *created;
    return *created;
}

} // namespace

auto set_thread_count(std::size_t total_threads) noexcept -> void {
    configured_thread_count.store(total_threads, std::memory_order_relaxed);
}

auto thread_count() noexcept -> std::size_t {
    const auto configured = configured_thread_count.load(std::memory_order_relaxed);
    return configured == 0 ? std::max(1U, std::thread::hardware_concurrency()) : configured;
}

auto reserve_thread_pool(std::size_t total_threads) -> Result<void> {
    static std::mutex mutex;
    static std::unordered_map<std::size_t, std::shared_ptr<ThreadPool>> reservations;
    if (total_threads == 0) total_threads = thread_count();
    std::scoped_lock lock(mutex);
    if (reservations.contains(total_threads)) return {};
    auto pool = default_thread_pool(total_threads);
    if (!pool) return std::unexpected(std::move(pool.error()));
    reservations.emplace(total_threads, std::move(*pool));
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
    if (auto status = check_status(ynn_optimize_subgraph(graph_, (*thread_pool)->get(), 0), "optimize graph");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    ynn_runtime_t runtime = nullptr;
    if (auto status = check_status(ynn_create_runtime(graph_, (*thread_pool)->get(), 0, &runtime), "create runtime");
        !status) {
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