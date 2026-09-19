#include "kidi/runtime/ynn.h"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "slinky/base/thread_pool_impl.h"
#include "ynnpack/base/arch.h"

namespace kidi::runtime {
namespace {

class Scheduler {
public:
    explicit Scheduler(int worker_count) : implementation_(worker_count) {}

    ~Scheduler() { implementation_.work_until_idle(); }

    [[nodiscard]] static const ynn_scheduler* interface() noexcept {
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

class YnnThreadPool {
public:
    static Result<std::shared_ptr<YnnThreadPool>> create(std::size_t total_threads) {
        auto result = std::shared_ptr<YnnThreadPool>(new YnnThreadPool(static_cast<int>(total_threads - 1)));
        auto status =
            check_ynn_status(ynn_create_threadpool(Scheduler::interface(), &result->scheduler_, 0, &result->handle_),
                             "create thread pool");
        if (!status) return std::unexpected(std::move(status.error()));
        return result;
    }

    ~YnnThreadPool() {
        if (handle_ != nullptr) ynn_delete_threadpool(handle_);
    }

    [[nodiscard]] ynn_threadpool_t get() const noexcept { return handle_; }

private:
    explicit YnnThreadPool(int worker_count) : scheduler_(worker_count) {}

    Scheduler scheduler_;
    ynn_threadpool_t handle_ = nullptr;
};

namespace {

std::atomic<std::size_t> configured_thread_count = 0;

Result<std::shared_ptr<YnnThreadPool>> default_thread_pool() {
    struct PoolCache {
        std::mutex mutex;
        std::unordered_map<std::size_t, std::weak_ptr<YnnThreadPool>> pools;
    };
    static PoolCache cache;

    const auto total_threads = ynn_thread_count();
    std::scoped_lock lock(cache.mutex);
    if (auto existing = cache.pools[total_threads].lock()) return existing;
    auto created = YnnThreadPool::create(total_threads);
    if (!created) return std::unexpected(std::move(created.error()));
    cache.pools[total_threads] = *created;
    return *created;
}

} // namespace

void set_ynn_thread_count(std::size_t total_threads) noexcept {
    configured_thread_count.store(total_threads, std::memory_order_relaxed);
}

std::size_t ynn_thread_count() noexcept {
    const auto configured = configured_thread_count.load(std::memory_order_relaxed);
    return configured == 0 ? std::max(1U, std::thread::hardware_concurrency()) : configured;
}

std::uint64_t ynn_supported_arch_flags() noexcept { return ynn::get_supported_arch_flags(); }

std::string ynn_supported_arch_names() {
    const auto flags = ynn_supported_arch_flags();
    std::string result;
    const auto append = [&](std::uint64_t flag, const char* name) {
        if ((flags & flag) == 0) return;
        if (!result.empty()) result += ',';
        result += name;
    };
#ifdef YNN_ARCH_ARM
    append(ynn::arch_flag::neon, "neon");
    append(ynn::arch_flag::neonfma, "fma");
    append(ynn::arch_flag::neondot, "dotprod");
    append(ynn::arch_flag::neonfp16, "fp16");
    append(ynn::arch_flag::neonfp16arith, "fp16arith");
    append(ynn::arch_flag::neonbf16, "bf16");
    append(ynn::arch_flag::neonfp8, "fp8");
    append(ynn::arch_flag::neonfp8dot4, "fp8dot4");
    append(ynn::arch_flag::neoni8mm, "i8mm");
    append(ynn::arch_flag::sme, "sme");
    append(ynn::arch_flag::sme2, "sme2");
    append(ynn::arch_flag::sve, "sve");
#endif
    return result.empty() ? "generic" : result;
}

Result<void> check_ynn_status(ynn_status status, const char* operation) {
    if (status == ynn_status_success) {
        return {};
    }
    return std::unexpected(Error{
        ErrorCode::RUNTIME,
        std::string(operation) + " failed with YNNPACK status " + std::to_string(status),
    });
}

YnnGraph::YnnGraph(ynn_subgraph_t graph) : graph_(graph) {}

YnnGraph::YnnGraph(YnnGraph&& other) noexcept : graph_(other.release()) {}

YnnGraph& YnnGraph::operator=(YnnGraph&& other) noexcept {
    if (this != &other) {
        if (graph_ != nullptr) {
            ynn_delete_subgraph(graph_);
        }
        graph_ = other.release();
    }
    return *this;
}

YnnGraph::~YnnGraph() {
    if (graph_ != nullptr) {
        ynn_delete_subgraph(graph_);
    }
}

Result<YnnGraph> YnnGraph::create(std::uint32_t external_value_count, std::uint32_t flags) {
    ynn_subgraph_t graph = nullptr;
    if (auto status = check_ynn_status(ynn_create_subgraph(external_value_count, flags, &graph), "create graph");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    return YnnGraph(graph);
}

ynn_subgraph_t YnnGraph::get() const noexcept { return graph_; }

ynn_subgraph_t YnnGraph::release() noexcept { return std::exchange(graph_, nullptr); }

Result<YnnExecutable> YnnGraph::compile() && {
    auto thread_pool = default_thread_pool();
    if (!thread_pool) return std::unexpected(std::move(thread_pool.error()));
    if (auto status = check_ynn_status(ynn_optimize_subgraph(graph_, (*thread_pool)->get(), 0), "optimize graph");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    ynn_runtime_t runtime = nullptr;
    if (auto status =
            check_ynn_status(ynn_create_runtime(graph_, (*thread_pool)->get(), 0, &runtime), "create runtime");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    return YnnExecutable(release(), runtime, std::move(*thread_pool));
}

YnnExecutable::YnnExecutable(ynn_subgraph_t graph, ynn_runtime_t runtime, std::shared_ptr<YnnThreadPool> thread_pool)
    : graph_(graph), runtime_(runtime), thread_pool_(std::move(thread_pool)) {}

YnnExecutable::YnnExecutable(YnnExecutable&& other) noexcept
    : graph_(std::exchange(other.graph_, nullptr)),
      runtime_(std::exchange(other.runtime_, nullptr)),
      thread_pool_(std::move(other.thread_pool_)),
      external_shapes_(std::move(other.external_shapes_)),
      reshape_needed_(other.reshape_needed_) {}

YnnExecutable& YnnExecutable::operator=(YnnExecutable&& other) noexcept {
    if (this != &other) {
        if (runtime_ != nullptr) ynn_delete_runtime(runtime_);
        if (graph_ != nullptr) ynn_delete_subgraph(graph_);
        graph_ = std::exchange(other.graph_, nullptr);
        runtime_ = std::exchange(other.runtime_, nullptr);
        thread_pool_ = std::move(other.thread_pool_);
        external_shapes_ = std::move(other.external_shapes_);
        reshape_needed_ = other.reshape_needed_;
    }
    return *this;
}

YnnExecutable::~YnnExecutable() {
    if (runtime_ != nullptr) ynn_delete_runtime(runtime_);
    if (graph_ != nullptr) ynn_delete_subgraph(graph_);
}

Result<void> YnnExecutable::set_shape(std::uint32_t external_id, std::span<const std::size_t> dimensions) {
    if (const auto existing = external_shapes_.find(external_id);
        existing != external_shapes_.end() && std::ranges::equal(existing->second, dimensions)) {
        return {};
    }
    auto status =
        check_ynn_status(ynn_set_external_value_shape(runtime_, external_id, dimensions.size(), dimensions.data()),
                         "set external shape");
    if (status) {
        external_shapes_.insert_or_assign(external_id, std::vector<std::size_t>(dimensions.begin(), dimensions.end()));
        reshape_needed_ = true;
    }
    return status;
}

Result<void> YnnExecutable::reshape() {
    if (!reshape_needed_) return {};
    auto status = check_ynn_status(ynn_reshape_runtime(runtime_), "reshape runtime");
    if (status) reshape_needed_ = false;
    return status;
}

Result<void> YnnExecutable::bind(std::uint32_t external_id, void* data) {
    return check_ynn_status(ynn_set_external_value_data(runtime_, external_id, data), "bind external value");
}

Result<void> YnnExecutable::invoke() { return check_ynn_status(ynn_invoke_runtime(runtime_), "invoke runtime"); }

Result<std::vector<std::size_t>> YnnExecutable::shape(std::uint32_t external_id) const {
    std::array<std::size_t, YNN_MAX_TENSOR_RANK> dimensions{};
    std::size_t rank = dimensions.size();
    if (auto status = check_ynn_status(ynn_get_external_value_shape(runtime_, external_id, &rank, dimensions.data()),
                                       "get external shape");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    return std::vector<std::size_t>(dimensions.begin(), dimensions.begin() + static_cast<std::ptrdiff_t>(rank));
}

Result<std::int32_t> YnnExecutable::concurrency() const {
    std::int32_t result = 0;
    std::size_t result_size = sizeof(result);
    if (auto status =
            check_ynn_status(ynn_query_runtime(runtime_, ynn_runtime_property_concurrency, &result, &result_size),
                             "query runtime concurrency");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    return result;
}

} // namespace kidi::runtime