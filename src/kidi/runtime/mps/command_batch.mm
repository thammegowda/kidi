#include "kidi/runtime/mps/command_batch.h"
#include "kidi/tensor/tensor.h"
#include "kidi/tensor/metal.h"
#include "command_batch_metal_source.h"

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
        auto device = command_queue().device;
        NSError* error = nil;
        auto library =
            [device newLibraryWithSource:[NSString stringWithUTF8String:KIDI_COMMAND_BATCH_METAL_SOURCE]
                                 options:nil
                                   error:&error];
        if (!library) return (id<MTLComputePipelineState>)nil;
        return [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"scatter_bytes"] error:&error];
    }();
    return pipeline;
}
} // namespace
struct CommandBatch::Impl {
    MPSCommandBuffer* buffer;
    id<MTLComputeCommandEncoder> compute = nil;
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
    id<MTLBuffer> index_error = nil;
    void end_compute() noexcept {
        if (compute != nil) {
            [compute endEncoding];
            compute = nil;
        }
    }
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
auto CommandBatch::native_handle() const noexcept -> void* {
    impl_->end_compute();
    return (__bridge void*)impl_->buffer;
}
auto CommandBatch::compute_encoder() -> Result<void*> {
    if (impl_->finished) return std::unexpected(Error{ErrorCode::RUNTIME, "Metal command batch already finished"});
    if (impl_->compute == nil) {
        impl_->compute = [impl_->buffer computeCommandEncoder];
        if (!impl_->compute) return std::unexpected(Error{ErrorCode::RUNTIME, "create Metal compute encoder"});
    }
    return (__bridge void*)impl_->compute;
}
auto CommandBatch::restart() -> Result<void> {
    if (!impl_->finished) return {};
    auto status = finish();
    if (!status) return status;
    impl_->buffer = [MPSCommandBuffer commandBufferFromCommandQueue:command_queue()];
    if (!impl_->buffer) return std::unexpected(Error{ErrorCode::RUNTIME, "restart Metal command batch"});
    impl_->compute = nil;
    impl_->finished = false;
    impl_->next_ticket = 0;
    if (impl_->index_error) *static_cast<std::int32_t*>(impl_->index_error.contents) = 0;
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
    impl_->end_compute();
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
auto CommandBatch::copy_slice_(const tensor::Tensor& source, tensor::Tensor& destination, std::size_t outer,
                               std::size_t source_bytes, std::size_t destination_bytes, std::size_t offset_bytes)
    -> Result<void> {
    if (impl_->finished || source.device() != tensor::Device::apple_gpu() || source.dtype() != destination.dtype() ||
        !source.is_contiguous() || !destination.is_contiguous() || !outer || !source_bytes || !destination_bytes ||
        offset_bytes > destination_bytes || source_bytes > destination_bytes - offset_bytes ||
        outer > destination.nbytes() / destination_bytes || outer > source.nbytes() / source_bytes)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal contiguous slice copy"});
    auto writable = destination.host_bytes();
    if (!writable) return std::unexpected(std::move(writable.error()));
    auto from = tensor::metal_buffer(source), to = tensor::metal_buffer(destination);
    if (!from) return std::unexpected(std::move(from.error()));
    if (!to) return std::unexpected(std::move(to.error()));
    tensor::Tensor snapshot;
    if (from->handle == to->handle) {
        auto temporary =
            tensor::Tensor::empty({source.shape().begin(), source.shape().end()}, source.dtype(), source.device());
        if (!temporary) return std::unexpected(std::move(temporary.error()));
        snapshot = std::move(*temporary);
        auto status = copy_(source, snapshot);
        if (!status) return status;
        from = tensor::metal_buffer(snapshot);
        if (!from) return std::unexpected(std::move(from.error()));
    }
    impl_->end_compute();
    auto encoder = [impl_->buffer blitCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create slice copy encoder"});
    for (std::size_t block = 0; block < outer; ++block)
        [encoder copyFromBuffer:(__bridge id<MTLBuffer>)from->handle
                   sourceOffset:from->offset_bytes + block * source_bytes
                       toBuffer:(__bridge id<MTLBuffer>)to->handle
              destinationOffset:to->offset_bytes + block * destination_bytes + offset_bytes
                           size:source_bytes];
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
    auto error = index_error_buffer();
    if (!error) return std::unexpected(std::move(error.error()));
    const std::uint64_t row_bytes = updates.size(2) * tensor::element_size(updates.dtype());
    const std::uint64_t rows = updates.size(0) * indices.numel();
    const std::uint32_t copy_mode =
        (row_bytes % 16 == 0 && target->offset_bytes % 16 == 0 && source->offset_bytes % 16 == 0) ? 2u
        : (row_bytes % 4 == 0 && target->offset_bytes % 4 == 0 && source->offset_bytes % 4 == 0)  ? 1u
                                                                                                  : 0u;
    const struct {
        std::uint64_t capacity, row_bytes, indices, rows;
        std::uint32_t copy_mode;
    } geometry{destination.size(1), row_bytes, indices.numel(), rows, copy_mode};
    auto encoder_handle = compute_encoder();
    if (!encoder_handle) return std::unexpected(std::move(encoder_handle.error()));
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)*encoder_handle;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)target->handle offset:target->offset_bytes atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)source->handle offset:source->offset_bytes atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)slots->handle offset:slots->offset_bytes atIndex:2];
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)*error offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return {};
}
auto CommandBatch::index_error_buffer() -> Result<void*> {
    if (!impl_->index_error) {
        impl_->index_error = [command_queue().device newBufferWithLength:sizeof(std::int32_t)
                                                                 options:MTLResourceStorageModeShared];
        if (!impl_->index_error) return std::unexpected(Error{ErrorCode::RUNTIME, "allocate Metal index error flag"});
        *static_cast<std::int32_t*>(impl_->index_error.contents) = 0;
    }
    return (__bridge void*)impl_->index_error;
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
        impl_->end_compute();
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
    if (impl_->index_error) {
        const auto error = *static_cast<const std::int32_t*>(impl_->index_error.contents);
        if (error)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error == 2
                                                                          ? "Metal expert ID out of bounds"
                                                                          : "cache scatter index out of bounds"});
    }
    return {};
}
} // namespace kidi::runtime::mps