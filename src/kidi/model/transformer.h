#pragma once
#include <memory>
#include "kidi/layers/transformer.h"
#include "kidi/model/package.h"
#include "kidi/inference/profile.h"

namespace kidi::model {
struct EncoderState {
    std::vector<layers::KeyValue> layers;
    tensor::Tensor mask;
};
struct DecoderState {
    // Cache storage is updated in place; copying this state shares its tensors, not an independent history.
    std::vector<layers::KeyValue> layers;
    tensor::Tensor mask, index, embedding;
    std::size_t position = 0;
    std::size_t capacity = 0;
};
KIDI_MODULE(Transformer);
class TransformerImpl : public Module {
public:
    TransformerImpl(const Package&, tensor::Device);
    ~TransformerImpl();
    static auto create(const Package&, tensor::Device) -> Result<Transformer>;
    auto encode(std::span<const std::vector<std::int32_t>> sources, inference::InferenceStats* = nullptr)
        -> Result<EncoderState>;
    auto create_state(std::size_t batch, std::size_t capacity) -> Result<DecoderState>;
    auto forward(const EncoderState&, std::span<const std::int32_t> tokens, std::size_t batch,
                 DecoderState* state = nullptr, inference::InferenceStats* = nullptr) -> Result<tensor::Tensor>;

private:
    struct State;
    std::unique_ptr<State> impl_;
};
} // namespace kidi::model