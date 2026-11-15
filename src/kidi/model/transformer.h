#pragma once
#include <memory>
#include "kidi/layers/transformer.h"
#include "kidi/model/model.h"
#include <yaml-cpp/yaml.h>
#include "kidi/inference/profile.h"

namespace kidi::model {
struct EncoderState : EncoderMemory {
    std::vector<layers::KeyValue> layers;
    tensor::Tensor mask;
};
struct DecoderState : EncoderDecoderState {
    // Cache storage is updated in place; copying this state shares its tensors, not an independent history.
    std::vector<layers::KeyValue> layers;
    tensor::Tensor mask, index, embedding;
    std::size_t position = 0;
    std::size_t capacity = 0;
    auto sequence_position() const noexcept -> std::size_t override { return position; }
    auto sequence_capacity() const noexcept -> std::size_t override { return capacity; }
};
KIDI_MODULE(Transformer);
class TransformerImpl : public Seq2SeqModel {
public:
    explicit TransformerImpl(const YAML::Node& config);
    ~TransformerImpl() override;
    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto create(const YAML::Node& config) -> Result<Transformer>;
    static auto checkpoint_dtype(const checkpoint::Weights& weights) -> Result<tensor::DType>;
    auto state_mapping_specs() const -> std::vector<checkpoint::StateMappingSpec> override;
    auto hidden_size() const noexcept -> std::int32_t override;
    auto source_vocabulary_size() const noexcept -> std::int32_t override;
    auto vocabulary_size() const noexcept -> std::int32_t override;
    auto max_positions() const noexcept -> std::size_t override;
    auto preparation_ns() const noexcept -> std::uint64_t override;
    auto release_workspaces() -> void override;

    auto encode_text(std::span<const std::vector<std::int32_t>> sources, inference::InferenceStats* = nullptr)
        -> Result<std::shared_ptr<EncoderMemory>> override;
    auto create_decoder_state(std::shared_ptr<const EncoderMemory> memory, std::size_t batch, std::size_t capacity)
        -> Result<std::unique_ptr<LanguageModelState>> override;
    auto decode_step(const EncoderMemory& memory, std::span<const std::int32_t> tokens, std::size_t batch,
                     LanguageModelState* state = nullptr, inference::InferenceStats* = nullptr)
        -> Result<tensor::Tensor> override;

private:
    struct State;
    std::unique_ptr<State> impl_;
};
} // namespace kidi::model