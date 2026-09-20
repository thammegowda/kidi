#include "kidi/runtime/mps/command_batch.h"
#include "kidi/tensor/tensor.h"
#include "kidi/tensor/metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <utility>
#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace kidi::runtime::mps {
namespace {
auto command_queue() -> id<MTLCommandQueue> {
    static id<MTLCommandQueue> queue = [MTLCreateSystemDefaultDevice() newCommandQueue];
    return queue;
}
auto scatter_pipeline() -> id<MTLComputePipelineState> {
    static id<MTLComputePipelineState> pipeline = [] {
        NSString* source = @R"metal(
            #include <metal_stdlib>
            using namespace metal;
            kernel void scatter_bytes(device uchar* destination [[buffer(0)]],
                                      device const uchar* updates [[buffer(1)]],
                                      device const int* indices [[buffer(2)]],
                                      constant ulong4& sizes [[buffer(3)]],
                                      device atomic_int* invalid [[buffer(4)]],
                                      uint element [[thread_position_in_grid]]) {
                if (element >= sizes.w) return;
                for (ulong index = 0; index < sizes.z; ++index) {
                    if (indices[index] < 0 || ulong(indices[index]) >= sizes.x) {
                        if (element == 0) atomic_store_explicit(invalid, 1, memory_order_relaxed);
                        return;
                    }
                }
                const ulong row = element / sizes.y;
                const ulong index = row % sizes.z;
                for (ulong later = index + 1; later < sizes.z; ++later)
                    if (indices[later] == indices[index]) return;
                const ulong batch = row / sizes.z;
                const ulong offset = (batch * sizes.x + ulong(indices[index])) * sizes.y + element % sizes.y;
                destination[offset] = updates[element];
            }
        )metal";
        auto device = command_queue().device;
        NSError* error = nil;
        auto library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) return (id<MTLComputePipelineState>)nil;
        return [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"scatter_bytes"] error:&error];
    }();
    return pipeline;
}
} // namespace
struct CommandBatch::Impl {
    MPSCommandBuffer* buffer;
    bool finished = false;
    struct Completions {
        std::mutex mutex;
        std::condition_variable condition;
        std::size_t pending = 0;
        std::string error;
    };
    std::shared_ptr<Completions> completions = std::make_shared<Completions>();
    struct Ticket {
        std::shared_ptr<Completions> owner;
        std::atomic_bool completed{false};
        explicit Ticket(std::shared_ptr<Completions> state) : owner(std::move(state)) {}
    };
    std::vector<std::shared_ptr<Ticket>> tickets;
    std::size_t next_ticket = 0;
    id<MTLBuffer> scatter_error = nil;
};
CommandBatch::CommandBatch(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CommandBatch::CommandBatch(CommandBatch&&) noexcept = default;
auto CommandBatch::operator=(CommandBatch&& other) noexcept -> CommandBatch& {
    if (this != &other) {
        if (impl_ && !impl_->finished) {
            auto ignored = finish();
            (void)ignored;
        }
        impl_ = std::move(other.impl_);
    }
    return *this;
}
CommandBatch::~CommandBatch() {
    if (impl_ && !impl_->finished) {
        auto ignored = finish();
        (void)ignored;
    }
}
auto CommandBatch::create() -> Result<CommandBatch> {
    auto queue = command_queue();
    if (!queue) return std::unexpected(Error{ErrorCode::UNSUPPORTED, "no Metal command queue"});
    auto impl = std::make_unique<Impl>();
    impl->buffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
    if (!impl->buffer) return std::unexpected(Error{ErrorCode::RUNTIME, "create Metal command batch"});
    return CommandBatch(std::move(impl));
}
auto CommandBatch::native_handle() const noexcept -> void* { return (__bridge void*)impl_->buffer; }
auto CommandBatch::restart() -> Result<void> {
    if (!impl_->finished) return {};
    auto status = finish();
    if (!status) return status;
    impl_->buffer = [MPSCommandBuffer commandBufferFromCommandQueue:command_queue()];
    if (!impl_->buffer) return std::unexpected(Error{ErrorCode::RUNTIME, "restart Metal command batch"});
    impl_->finished = false;
    impl_->next_ticket = 0;
    if (impl_->scatter_error) *static_cast<std::int32_t*>(impl_->scatter_error.contents) = 0;
    return {};
}
auto CommandBatch::copy_(const tensor::Tensor& source, tensor::Tensor& destination) -> Result<void> {
    if (impl_->finished || source.dtype() != destination.dtype() ||
        !std::ranges::equal(source.shape(), destination.shape()))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid in-place Metal copy"});
    auto writable = destination.host_bytes();
    if (!writable) return std::unexpected(std::move(writable.error()));
    auto from = tensor::metal_buffer(source), to = tensor::metal_buffer(destination);
    if (!from) return std::unexpected(std::move(from.error()));
    if (!to) return std::unexpected(std::move(to.error()));
    if (from->handle == to->handle && from->offset_bytes < to->offset_bytes + destination.nbytes() &&
        to->offset_bytes < from->offset_bytes + source.nbytes())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "Metal copy requires distinct source storage"});
    auto encoder = [impl_->buffer blitCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create in-place copy encoder"});
    [encoder copyFromBuffer:(__bridge id<MTLBuffer>)from->handle
               sourceOffset:from->offset_bytes
                   toBuffer:(__bridge id<MTLBuffer>)to->handle
          destinationOffset:to->offset_bytes
                       size:source.nbytes()];
    [encoder endEncoding];
    return {};
}
auto CommandBatch::scatter_(tensor::Tensor& destination, const tensor::Tensor& updates, const tensor::Tensor& indices)
    -> Result<void> {
    if (impl_->finished || destination.dimensions() != 3 || updates.dimensions() != 3 ||
        destination.dtype() != updates.dtype() || indices.dtype() != tensor::DType::I32 || indices.dimensions() != 1 ||
        destination.size(0) != updates.size(0) || destination.size(2) != updates.size(2) ||
        updates.size(1) != indices.numel() || updates.nbytes() > std::numeric_limits<std::uint32_t>::max())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid in-place Metal scatter"});
    auto writable = destination.host_bytes();
    if (!writable) return std::unexpected(std::move(writable.error()));
    auto target = tensor::metal_buffer(destination), source = tensor::metal_buffer(updates),
         slots = tensor::metal_buffer(indices);
    if (!target) return std::unexpected(std::move(target.error()));
    if (!source) return std::unexpected(std::move(source.error()));
    if (!slots) return std::unexpected(std::move(slots.error()));
    if (!updates.nbytes()) return {};
    auto pipeline = scatter_pipeline();
    if (!pipeline) return std::unexpected(Error{ErrorCode::RUNTIME, "create Metal in-place scatter pipeline"});
    tensor::Tensor update_snapshot, index_snapshot;
    if (source->handle == target->handle) {
        auto snapshot =
            tensor::Tensor::empty({updates.shape().begin(), updates.shape().end()}, updates.dtype(), updates.device());
        if (!snapshot) return std::unexpected(std::move(snapshot.error()));
        update_snapshot = std::move(*snapshot);
        auto copied = copy_(updates, update_snapshot);
        if (!copied) return copied;
        source = tensor::metal_buffer(update_snapshot);
        if (!source) return std::unexpected(std::move(source.error()));
    }
    if (slots->handle == target->handle) {
        auto snapshot =
            tensor::Tensor::empty({indices.shape().begin(), indices.shape().end()}, indices.dtype(), indices.device());
        if (!snapshot) return std::unexpected(std::move(snapshot.error()));
        index_snapshot = std::move(*snapshot);
        auto copied = copy_(indices, index_snapshot);
        if (!copied) return copied;
        slots = tensor::metal_buffer(index_snapshot);
        if (!slots) return std::unexpected(std::move(slots.error()));
    }
    if (!impl_->scatter_error) {
        impl_->scatter_error = [pipeline.device newBufferWithLength:sizeof(std::int32_t)
                                                            options:MTLResourceStorageModeShared];
        if (!impl_->scatter_error) return std::unexpected(Error{ErrorCode::RUNTIME, "allocate scatter error flag"});
        *static_cast<std::int32_t*>(impl_->scatter_error.contents) = 0;
    }
    const std::array<std::uint64_t, 4> sizes{destination.size(1),
                                             updates.size(2) * tensor::element_size(updates.dtype()), indices.numel(),
                                             updates.nbytes()};
    auto encoder = [impl_->buffer computeCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create scatter encoder"});
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)target->handle offset:target->offset_bytes atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)source->handle offset:source->offset_bytes atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)slots->handle offset:slots->offset_bytes atIndex:2];
    [encoder setBytes:sizes.data() length:sizeof(sizes) atIndex:3];
    [encoder setBuffer:impl_->scatter_error offset:0 atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(updates.nbytes(), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
    [encoder endEncoding];
    return {};
}
auto CommandBatch::track_completion() -> std::function<void(std::string)> {
    if (impl_->next_ticket == impl_->tickets.size())
        impl_->tickets.push_back(std::make_shared<Impl::Ticket>(impl_->completions));
    auto ticket = impl_->tickets[impl_->next_ticket++];
    ticket->completed.store(false);
    {
        std::scoped_lock lock(ticket->owner->mutex);
        ++ticket->owner->pending;
    }
    return [ticket = std::move(ticket)](std::string error) {
        if (ticket->completed.exchange(true)) return;
        auto& completions = *ticket->owner;
        {
            std::scoped_lock lock(completions.mutex);
            if (completions.error.empty() && !error.empty()) completions.error = std::move(error);
            --completions.pending;
        }
        completions.condition.notify_all();
    };
}
auto CommandBatch::finish() -> Result<void> {
    if (!impl_->finished) {
        [impl_->buffer commit];
        [impl_->buffer waitUntilCompleted];
        impl_->finished = true;
    }
    std::unique_lock lock(impl_->completions->mutex);
    impl_->completions->condition.wait(lock, [&] { return impl_->completions->pending == 0; });
    if (!impl_->completions->error.empty())
        return std::unexpected(Error{ErrorCode::RUNTIME, impl_->completions->error});
    if (impl_->buffer.status == MTLCommandBufferStatusError)
        return std::unexpected(
            Error{ErrorCode::RUNTIME,
                  "Metal command batch failed: " + std::string(impl_->buffer.error.localizedDescription.UTF8String)});
    if (impl_->scatter_error && *static_cast<const std::int32_t*>(impl_->scatter_error.contents))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cache scatter index out of bounds"});
    return {};
}
} // namespace kidi::runtime::mps