#include "kidi/runtime/mps/graph.h"
#include "kidi/runtime/mps/command_batch.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "kidi/tensor/metal.h"

namespace kidi::runtime::mps {
namespace {

auto ns_string(std::string_view value) -> NSString* {
    if (value.empty()) return nil;
    return [[NSString alloc] initWithBytes:value.data() length:value.size() encoding:NSUTF8StringEncoding];
}

auto mps_shape(std::span<const std::int64_t> shape) -> MPSShape* {
    NSMutableArray<NSNumber*>* result = [NSMutableArray arrayWithCapacity:shape.size()];
    for (const auto extent : shape) [result addObject:@(extent)];
    return result;
}

auto mps_axes(std::span<const std::int64_t> axes) -> NSArray<NSNumber*>* {
    NSMutableArray<NSNumber*>* result = [NSMutableArray arrayWithCapacity:axes.size()];
    for (const auto axis : axes) [result addObject:@(axis)];
    return result;
}

auto mps_type(tensor::DType dtype) -> Result<MPSDataType> {
    switch (dtype) {
        case tensor::DType::BOOL:
            return MPSDataTypeBool;
        case tensor::DType::I8:
            return MPSDataTypeInt8;
        case tensor::DType::I16:
            return MPSDataTypeInt16;
        case tensor::DType::I32:
            return MPSDataTypeInt32;
        case tensor::DType::I64:
            return MPSDataTypeInt64;
        case tensor::DType::U8:
            return MPSDataTypeUInt8;
        case tensor::DType::U16:
            return MPSDataTypeUInt16;
        case tensor::DType::U32:
            return MPSDataTypeUInt32;
        case tensor::DType::U64:
            return MPSDataTypeUInt64;
        case tensor::DType::F16:
            return MPSDataTypeFloat16;
        case tensor::DType::BF16:
            return MPSDataTypeBFloat16;
        case tensor::DType::F32:
            return MPSDataTypeFloat32;
        default:
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "MPSGraph does not support kidi dtype " +
                                                                     std::string(tensor::to_string(dtype))});
    }
}

auto exception_error(NSException* exception, std::string_view operation) -> Error {
    return Error{
        ErrorCode::RUNTIME,
        std::string(operation) + ": " + (exception.reason == nil ? "MPSGraph exception" : exception.reason.UTF8String)};
}

} // namespace

struct Graph::Impl {
    id<MTLDevice> device;
    MPSGraph* graph;
    NSMutableArray<MPSGraphTensor*>* tensors;
    std::vector<std::optional<std::pair<std::vector<std::int64_t>, tensor::DType>>> specs;
    std::vector<tensor::Tensor> retained_constants;
    std::optional<Error> error;

    auto add(MPSGraphTensor* tensor,
             std::optional<std::pair<std::vector<std::int64_t>, tensor::DType>> spec = std::nullopt) -> Value {
        if (tensor == nil) {
            fail(Error{ErrorCode::RUNTIME, "MPSGraph returned a null tensor"});
            return {};
        }
        if (tensors.count >= std::numeric_limits<std::uint32_t>::max()) {
            fail(Error{ErrorCode::RUNTIME, "MPSGraph value count exceeds the supported range"});
            return {};
        }
        [tensors addObject:tensor];
        specs.push_back(std::move(spec));
        return Value(static_cast<std::uint32_t>(tensors.count - 1));
    }

    auto get(Value value) -> MPSGraphTensor* {
        if (!value.valid() || value.id_ >= tensors.count) {
            fail(Error{ErrorCode::INVALID_ARGUMENT, "invalid MPSGraph value"});
            return nil;
        }
        return tensors[value.id_];
    }

    auto fail(Error next) -> void {
        if (!error) error = std::move(next);
    }
};

struct Executable::Impl {
    struct BindingView {
        tensor::MetalBufferView buffer;
        tensor::DType dtype;
        std::vector<std::int64_t> shape;
    };
    struct Bindings {
        std::vector<BindingView> inputs, outputs;
        NSArray<MPSGraphTensorData*>* input_data;
        NSArray<MPSGraphTensorData*>* output_data;
    };
    std::vector<Bindings> bindings;
    std::shared_ptr<void> graph;
    MPSGraphExecutable* executable;
    NSArray<MPSGraphType*>* input_types;
    std::vector<std::size_t> input_order;
    std::vector<std::size_t> output_order;
};

auto Value::valid() const noexcept -> bool { return id_ != INVALID; }

Executable::Executable(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Executable::Executable(Executable&&) noexcept = default;
auto Executable::operator=(Executable&&) noexcept -> Executable& = default;
Executable::~Executable() = default;

Graph::Graph(std::shared_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Graph::Graph(Graph&&) noexcept = default;
auto Graph::operator=(Graph&&) noexcept -> Graph& = default;
Graph::~Graph() = default;

auto Graph::create() -> Result<Graph> {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "no Metal device is available"});
        }
        auto impl = std::make_shared<Impl>();
        impl->device = device;
        impl->graph = [MPSGraph new];
        impl->graph.options = MPSGraphOptionsNone;
        impl->tensors = [NSMutableArray array];
        return Graph(std::move(impl));
    }
}

auto Graph::placeholder(std::span<const std::int64_t> shape, tensor::DType dtype, std::string_view name) -> Value {
    if (impl_->error) return {};
    auto type = mps_type(dtype);
    if (!type) {
        impl_->fail(std::move(type.error()));
        return {};
    }
    @try {
        auto* value = [impl_->graph placeholderWithShape:mps_shape(shape) dataType:*type name:ns_string(name)];
        return impl_->add(value, std::pair{std::vector<std::int64_t>(shape.begin(), shape.end()), dtype});
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph placeholder"));
        return {};
    }
}

auto Graph::constant(const tensor::Tensor& value, std::string_view) -> Value {
    if (impl_->error) return {};
    if (!value.defined() || !value.is_contiguous() || !value.is_host_accessible()) {
        impl_->fail(
            Error{ErrorCode::INVALID_ARGUMENT, "MPSGraph constants require contiguous host-accessible tensors"});
        return {};
    }
    auto type = mps_type(value.dtype());
    auto bytes = value.host_bytes();
    if (!type) {
        impl_->fail(std::move(type.error()));
        return {};
    }
    if (!bytes) {
        impl_->fail(std::move(bytes.error()));
        return {};
    }
    impl_->retained_constants.push_back(value);
    @try {
        NSData* data = [NSData dataWithBytesNoCopy:const_cast<std::byte*>(bytes->data())
                                            length:bytes->size()
                                      freeWhenDone:NO];
        return impl_->add([impl_->graph constantWithData:data shape:mps_shape(value.shape()) dataType:*type]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph constant"));
        return {};
    }
}

auto Graph::scalar(double value, tensor::DType dtype) -> Value {
    if (impl_->error) return {};
    auto type = mps_type(dtype);
    if (!type) {
        impl_->fail(std::move(type.error()));
        return {};
    }
    @try {
        return impl_->add([impl_->graph constantWithScalar:value shape:mps_shape({}) dataType:*type]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph scalar"));
        return {};
    }
}

#define KIDI_MPS_UNARY(method_name, selector, label)                                           \
    auto Graph::method_name(Value value, std::string_view name) -> Value {                     \
        if (impl_->error) return {};                                                           \
        @try {                                                                                 \
            return impl_->add([impl_->graph selector:impl_->get(value) name:ns_string(name)]); \
        } @catch (NSException * exception) {                                                   \
            impl_->fail(exception_error(exception, label));                                    \
            return {};                                                                         \
        }                                                                                      \
    }

KIDI_MPS_UNARY(exp, exponentWithTensor, "create MPSGraph exponent")
KIDI_MPS_UNARY(log, logarithmWithTensor, "create MPSGraph logarithm")
KIDI_MPS_UNARY(erf, erfWithTensor, "create MPSGraph erf")

#undef KIDI_MPS_UNARY

#define KIDI_MPS_BINARY(method_name, selector, label)                                  \
    auto Graph::method_name(Value left, Value right, std::string_view name) -> Value { \
        if (impl_->error) return {};                                                   \
        @try {                                                                         \
            return impl_->add([impl_->graph selector:impl_->get(left)                  \
                                     secondaryTensor:impl_->get(right)                 \
                                                name:ns_string(name)]);                \
        } @catch (NSException * exception) {                                           \
            impl_->fail(exception_error(exception, label));                            \
            return {};                                                                 \
        }                                                                              \
    }

KIDI_MPS_BINARY(add, additionWithPrimaryTensor, "create MPSGraph addition")
KIDI_MPS_BINARY(subtract, subtractionWithPrimaryTensor, "create MPSGraph subtraction")
KIDI_MPS_BINARY(multiply, multiplicationWithPrimaryTensor, "create MPSGraph multiplication")

#undef KIDI_MPS_BINARY

auto Graph::cast(Value value, tensor::DType dtype, std::string_view name) -> Value {
    if (impl_->error) return {};
    auto type = mps_type(dtype);
    if (!type) {
        impl_->fail(std::move(type.error()));
        return {};
    }
    @try {
        return impl_->add([impl_->graph castTensor:impl_->get(value) toType:*type name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph cast"));
        return {};
    }
}

auto Graph::matmul(Value left, Value right, bool transpose_left, bool transpose_right, std::string_view name) -> Value {
    if (impl_->error) return {};
    @try {
        auto* left_tensor = impl_->get(left);
        auto* right_tensor = impl_->get(right);
        if (transpose_left) {
            const auto rank = left_tensor.shape.count;
            left_tensor = [impl_->graph transposeTensor:left_tensor dimension:rank - 2 withDimension:rank - 1 name:nil];
        }
        if (transpose_right) {
            const auto rank = right_tensor.shape.count;
            right_tensor = [impl_->graph transposeTensor:right_tensor
                                               dimension:rank - 2
                                           withDimension:rank - 1
                                                    name:nil];
        }
        return impl_->add([impl_->graph matrixMultiplicationWithPrimaryTensor:left_tensor
                                                              secondaryTensor:right_tensor
                                                                         name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph matrix multiplication"));
        return {};
    }
}

auto Graph::reshape(Value value, std::span<const std::int64_t> shape, std::string_view name) -> Value {
    if (impl_->error) return {};
    @try {
        return impl_->add([impl_->graph reshapeTensor:impl_->get(value)
                                            withShape:mps_shape(shape)
                                                 name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph reshape"));
        return {};
    }
}

auto Graph::transpose(Value value, std::span<const std::int64_t> permutation, std::string_view name) -> Value {
    if (impl_->error) return {};
    @try {
        return impl_->add([impl_->graph transposeTensor:impl_->get(value)
                                            permutation:mps_axes(permutation)
                                                   name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph transpose"));
        return {};
    }
}

auto Graph::slice(Value value, std::int64_t axis, std::int64_t start, std::int64_t length, std::string_view name)
    -> Value {
    if (impl_->error) return {};
    @try {
        return impl_->add([impl_->graph sliceTensor:impl_->get(value)
                                          dimension:static_cast<NSUInteger>(axis)
                                              start:start
                                             length:length
                                               name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph slice"));
        return {};
    }
}

auto Graph::concat(std::span<const Value> values, std::int64_t axis, std::string_view name) -> Value {
    if (impl_->error) return {};
    @try {
        NSMutableArray<MPSGraphTensor*>* tensors = [NSMutableArray arrayWithCapacity:values.size()];
        for (const auto value : values) [tensors addObject:impl_->get(value)];
        return impl_->add([impl_->graph concatTensors:tensors dimension:axis name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph concatenation"));
        return {};
    }
}

auto Graph::gather(Value values, Value indices, std::int64_t axis, std::string_view name) -> Value {
    if (impl_->error) return {};
    @try {
        return impl_->add([impl_->graph gatherWithUpdatesTensor:impl_->get(values)
                                                  indicesTensor:impl_->get(indices)
                                                           axis:static_cast<NSUInteger>(axis)
                                                batchDimensions:0
                                                           name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph gather"));
        return {};
    }
}

auto Graph::scatter(Value data, Value updates, Value indices, std::int64_t axis, std::string_view name) -> Value {
    if (impl_->error) return {};
    @try {
        return impl_->add([impl_->graph scatterWithDataTensor:impl_->get(data)
                                                updatesTensor:impl_->get(updates)
                                                indicesTensor:impl_->get(indices)
                                                         axis:axis
                                                         mode:MPSGraphScatterModeSet
                                                         name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph scatter"));
        return {};
    }
}

auto Graph::reduce_sum(Value value, std::span<const std::int64_t> axes, std::string_view name) -> Value {
    if (impl_->error) return {};
    @try {
        return impl_->add([impl_->graph reductionSumWithTensor:impl_->get(value)
                                                          axes:mps_axes(axes)
                                                          name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph reduction sum"));
        return {};
    }
}

auto Graph::reduce_max(Value value, std::span<const std::int64_t> axes, std::string_view name) -> Value {
    if (impl_->error) return {};
    @try {
        return impl_->add([impl_->graph reductionMaximumWithTensor:impl_->get(value)
                                                              axes:mps_axes(axes)
                                                              name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph reduction maximum"));
        return {};
    }
}

auto Graph::softmax(Value value, std::int64_t axis, std::string_view name) -> Value {
    if (impl_->error) return {};
    @try {
        return impl_->add([impl_->graph softMaxWithTensor:impl_->get(value) axis:axis name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph softmax"));
        return {};
    }
}

auto Graph::normalize(Value value, Value mean_value, Value variance_value, Value scale, Value bias, float epsilon,
                      std::string_view name) -> Value {
    if (impl_->error) return {};
    @try {
        return impl_->add([impl_->graph normalizationWithTensor:impl_->get(value)
                                                     meanTensor:impl_->get(mean_value)
                                                 varianceTensor:impl_->get(variance_value)
                                                    gammaTensor:impl_->get(scale)
                                                     betaTensor:impl_->get(bias)
                                                        epsilon:epsilon
                                                           name:ns_string(name)]);
    } @catch (NSException* exception) {
        impl_->fail(exception_error(exception, "create MPSGraph normalization"));
        return {};
    }
}

auto Graph::compile(std::span<const Value> inputs, std::span<const Value> outputs) const -> Result<Executable> {
    if (impl_->error) return std::unexpected(*impl_->error);
    @autoreleasepool {
        @try {
            NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds = [NSMutableDictionary dictionary];
            NSMutableArray<MPSGraphTensor*>* input_tensors = [NSMutableArray arrayWithCapacity:inputs.size()];
            for (const auto input : inputs) {
                auto* graph_tensor = impl_->get(input);
                if (impl_->error) return std::unexpected(*impl_->error);
                if (input.id_ >= impl_->specs.size() || !impl_->specs[input.id_]) {
                    return std::unexpected(
                        Error{ErrorCode::INVALID_ARGUMENT, "MPSGraph compile input is not a placeholder"});
                }
                const auto& [shape, dtype] = *impl_->specs[input.id_];
                auto type = mps_type(dtype);
                if (!type) return std::unexpected(std::move(type.error()));
                feeds[graph_tensor] = [[MPSGraphShapedType alloc] initWithShape:mps_shape(shape) dataType:*type];
                [input_tensors addObject:graph_tensor];
            }
            NSMutableArray<MPSGraphTensor*>* output_tensors = [NSMutableArray arrayWithCapacity:outputs.size()];
            for (const auto output : outputs) {
                auto* graph_tensor = impl_->get(output);
                if (impl_->error) return std::unexpected(*impl_->error);
                [output_tensors addObject:graph_tensor];
            }
            MPSGraphCompilationDescriptor* descriptor = [MPSGraphCompilationDescriptor new];
            descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
            descriptor.waitForCompilationCompletion = YES;
            MPSGraphDevice* graph_device = [MPSGraphDevice deviceWithMTLDevice:impl_->device];
            MPSGraphExecutable* executable = [impl_->graph compileWithDevice:graph_device
                                                                       feeds:feeds
                                                               targetTensors:output_tensors
                                                            targetOperations:nil
                                                       compilationDescriptor:descriptor];
            if (executable == nil) {
                return std::unexpected(Error{ErrorCode::RUNTIME, "MPSGraph compilation returned no executable"});
            }
            executable.options = MPSGraphOptionsNone;
            auto result = std::make_unique<Executable::Impl>();
            result->graph = impl_;
            result->executable = executable;
            NSMutableArray<MPSGraphType*>* ordered_types = [NSMutableArray array];
            for (MPSGraphTensor* feed in executable.feedTensors) {
                const auto found = [input_tensors indexOfObjectIdenticalTo:feed];
                if (found == NSNotFound) {
                    return std::unexpected(Error{ErrorCode::RUNTIME, "MPSGraph reordered an unknown input"});
                }
                result->input_order.push_back(found);
                [ordered_types addObject:feeds[feed]];
            }
            result->input_types = ordered_types;
            for (MPSGraphTensor* target in executable.targetTensors) {
                const auto found = [output_tensors indexOfObjectIdenticalTo:target];
                if (found == NSNotFound) {
                    return std::unexpected(Error{ErrorCode::RUNTIME, "MPSGraph reordered an unknown output"});
                }
                result->output_order.push_back(found);
            }
            return Executable(std::move(result));
        } @catch (NSException* exception) {
            return std::unexpected(exception_error(exception, "compile MPSGraph"));
        }
    }
}

auto Executable::output_shapes() const -> Result<std::vector<std::vector<std::int64_t>>> {
    @autoreleasepool {
        @try {
            auto* types = [impl_->executable getOutputTypesWithDevice:nil
                                                           inputTypes:impl_->input_types
                                                compilationDescriptor:nil];
            if (types.count != impl_->output_order.size())
                return std::unexpected(Error{ErrorCode::RUNTIME, "MPSGraph output type count mismatch"});
            std::vector<std::vector<std::int64_t>> shapes(types.count);
            for (std::size_t index = 0; index < types.count; ++index) {
                for (NSNumber* extent in types[index].shape) {
                    if (extent.longLongValue < 0)
                        return std::unexpected(
                            Error{ErrorCode::UNSUPPORTED, "MPSGraph stage output must have a concrete shape"});
                    shapes[impl_->output_order[index]].push_back(extent.longLongValue);
                }
            }
            return shapes;
        } @catch (NSException* exception) {
            return std::unexpected(exception_error(exception, "query MPSGraph output shapes"));
        }
    }
}

auto Executable::encode(CommandBatch& batch, TensorInputs inputs, std::span<tensor::Tensor> outputs) const
    -> Result<void> {
    std::function<void(std::string)> completion;
    if (inputs.size() != impl_->input_order.size() || outputs.size() != impl_->output_order.size()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "MPSGraph binding count mismatch"});
    }
    @autoreleasepool {
        @try {
            const auto matches = [](TensorInputs current, const std::vector<Impl::BindingView>& previous) {
                if (current.size() != previous.size()) return false;
                for (std::size_t index = 0; index < current.size(); ++index) {
                    if (current[index].dtype() != previous[index].dtype ||
                        !std::ranges::equal(current[index].shape(), previous[index].shape))
                        return false;
                    auto buffer = tensor::metal_buffer(current[index]);
                    const auto& old_buffer = previous[index].buffer;
                    if (!buffer || buffer->handle != old_buffer.handle ||
                        buffer->offset_bytes != old_buffer.offset_bytes)
                        return false;
                }
                return true;
            };
            auto found = std::find_if(impl_->bindings.begin(), impl_->bindings.end(), [&](const auto& entry) {
                return matches(inputs, entry.inputs) &&
                       matches(std::span<const tensor::Tensor>(outputs), entry.outputs);
            });
            if (found == impl_->bindings.end()) {
                NSMutableArray<MPSGraphTensorData*>* input_data = [NSMutableArray arrayWithCapacity:inputs.size()];
                for (const auto index : impl_->input_order) {
                    const auto& input = inputs[index];
                    auto buffer = tensor::metal_buffer(input);
                    auto type = mps_type(input.dtype());
                    if (!buffer) return std::unexpected(std::move(buffer.error()));
                    if (!type) return std::unexpected(std::move(type.error()));
                    id<MTLBuffer> metal_buffer = (__bridge id<MTLBuffer>)buffer->handle;
                    if (buffer->offset_bytes != 0) {
                        return std::unexpected(
                            Error{ErrorCode::UNSUPPORTED, "MPSGraph does not accept tensor storage offsets"});
                    }
                    [input_data addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:metal_buffer
                                                                                  shape:mps_shape(input.shape())
                                                                               dataType:*type]];
                }
                NSMutableArray<MPSGraphTensorData*>* output_data = [NSMutableArray arrayWithCapacity:outputs.size()];
                for (const auto index : impl_->output_order) {
                    auto& output = outputs[index];
                    auto buffer = tensor::metal_buffer(output);
                    auto type = mps_type(output.dtype());
                    if (!buffer) return std::unexpected(std::move(buffer.error()));
                    if (!type) return std::unexpected(std::move(type.error()));
                    id<MTLBuffer> metal_buffer = (__bridge id<MTLBuffer>)buffer->handle;
                    if (buffer->offset_bytes != 0) {
                        return std::unexpected(
                            Error{ErrorCode::UNSUPPORTED, "MPSGraph does not accept tensor storage offsets"});
                    }
                    [output_data addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:metal_buffer
                                                                                   shape:mps_shape(output.shape())
                                                                                dataType:*type]];
                }
                if (impl_->bindings.size() == 64) impl_->bindings.erase(impl_->bindings.begin());
                std::vector<Impl::BindingView> input_views, output_views;
                input_views.reserve(inputs.size());
                output_views.reserve(outputs.size());
                for (std::size_t index = 0; index < inputs.size(); ++index) {
                    auto buffer = tensor::metal_buffer(inputs[index]);
                    if (!buffer) return std::unexpected(std::move(buffer.error()));
                    input_views.push_back(
                        {*buffer, inputs[index].dtype(), {inputs[index].shape().begin(), inputs[index].shape().end()}});
                }
                for (const auto& output : outputs) {
                    auto buffer = tensor::metal_buffer(output);
                    if (!buffer) return std::unexpected(std::move(buffer.error()));
                    output_views.push_back({*buffer, output.dtype(), {output.shape().begin(), output.shape().end()}});
                }
                impl_->bindings.push_back({std::move(input_views), std::move(output_views), input_data, output_data});
                found = impl_->bindings.end() - 1;
            }
            MPSGraphExecutableExecutionDescriptor* descriptor = [MPSGraphExecutableExecutionDescriptor new];
            descriptor.waitUntilCompleted = NO;
            completion = batch.track_completion();
            descriptor.completionHandler = ^(NSArray<MPSGraphTensorData*>*, NSError* error) {
              completion(error ? std::string(error.localizedDescription.UTF8String) : std::string{});
            };
            MPSCommandBuffer* command_buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
            NSArray<MPSGraphTensorData*>* results = [impl_->executable encodeToCommandBuffer:command_buffer
                                                                                 inputsArray:found->input_data
                                                                                resultsArray:found->output_data
                                                                         executionDescriptor:descriptor];
            if (results.count != outputs.size()) {
                completion("MPSGraph returned an unexpected result count");
                return std::unexpected(Error{ErrorCode::RUNTIME, "MPSGraph returned an unexpected result count"});
            }
            return {};
        } @catch (NSException* exception) {
            if (completion) completion("MPSGraph encode raised an exception");
            return std::unexpected(exception_error(exception, "execute MPSGraph"));
        }
    }
}

} // namespace kidi::runtime::mps