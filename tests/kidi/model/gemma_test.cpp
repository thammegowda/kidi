#include "kidi/model/gemma.h"

#include <cmath>
#include <filesystem>
#include <iostream>

auto main() -> int {
    using namespace kidi;
    try {
        for (const auto* fixture : {"gemma4", "gemma4-qat"}) {
            const auto directory = std::filesystem::path(KIDI_GEMMA_FIXTURE).parent_path() / fixture;
            const auto config = YAML::LoadFile((directory / "model.yaml").string())["model"];
            auto checkpoint = ops::require(model::Weights::load(directory / "model.safetensors"));
            auto reference = ops::require(model::Weights::load(directory / "reference.safetensors"));
            const auto token_tensor = ops::require(reference.tensor("tokens"));
            const auto expected = ops::require(reference.tensor("logits"));
            const auto tokens = ops::require(token_tensor.data<std::int32_t>());
            const auto values = ops::require(expected.data<float>());
            auto invalid_config = YAML::Clone(config);
            invalid_config.remove("global_head_dim");
            if (model::Gemma4Impl::create(invalid_config)) return 1;
            std::vector devices{tensor::Device::cpu()};
#if defined(__APPLE__)
            devices.push_back(tensor::Device::apple_gpu());
#endif
            for (auto device : devices) {
                const ModuleScope construction(tensor::DType::F32, false, device);
                auto model = ops::require(model::Gemma4Impl::create(config));
                ops::require(model->set_checkpoint(checkpoint));
                auto full = ops::require(model->create_state(8));
                auto prefill = ops::require(model->forward(tokens, full, true));
                const auto prefill_values = ops::require(prefill.data<float>());
                if (prefill_values.size() != values.size()) return 1;
                for (std::size_t index = 0; index < values.size(); ++index)
                    if (!std::isfinite(prefill_values[index]) ||
                        std::abs(prefill_values[index] - values[index]) > 2e-4F) {
                        std::cerr << "Gemma reference mismatch at " << index << ": " << prefill_values[index]
                                  << " != " << values[index] << '\n';
                        return 1;
                    }
                auto incremental = ops::require(model->create_state(8));
                for (std::size_t position = 0; position < tokens.size(); ++position) {
                    auto output = ops::require(model->forward(std::span(tokens).subspan(position, 1), incremental));
                    const auto actual = ops::require(output.data<float>());
                    for (std::size_t token = 0; token < actual.size(); ++token)
                        if (!std::isfinite(actual[token]) ||
                            std::abs(actual[token] - values[position * actual.size() + token]) > 2e-4F) {
                            std::cerr << "Gemma cached logits differ at " << position << ':' << token << '\n';
                            return 1;
                        }
                }
                if (incremental.position != tokens.size() || full.position != tokens.size()) return 1;
                auto chunked = ops::require(model->create_state(8));
                ops::require(model->prefill(tokens.first(3), chunked));
                const auto tail = ops::require(model->forward(tokens.subspan(3), chunked, true));
                const auto tail_values = ops::require(tail.data<float>());
                const auto reference_tail = values.last(tail_values.size());
                for (std::size_t index = 0; index < tail_values.size(); ++index)
                    if (!std::isfinite(tail_values[index]) ||
                        std::abs(tail_values[index] - reference_tail[index]) > 2e-4F)
                        return 1;
                const std::array<std::int32_t, 1> invalid{-1};
                if (model->forward(invalid, incremental) || incremental.position != tokens.size()) return 1;
                if (model->set_checkpoint(checkpoint, 4, 3)) return 1;
                if (config["quantization_config"]) {
                    if (model->set_checkpoint(checkpoint, 8, 4)) return 1;
                    ops::require(model->set_checkpoint(checkpoint, 0, 128, true));
                    auto packed_state = ops::require(model->create_state(8));
                    const auto packed_output = ops::require(model->forward(tokens, packed_state, true));
                    const auto packed_values = ops::require(packed_output.data<float>());
                    for (std::size_t index = 0; index < values.size(); ++index)
                        if (!std::isfinite(packed_values[index]) ||
                            std::abs(packed_values[index] - values[index]) > 2e-4F)
                            return 1;
                    continue;
                }
                ops::require(model->set_checkpoint(checkpoint, 8, 4));
                auto quantized_state = ops::require(model->create_state(8));
                const auto quantized = ops::require(model->forward(tokens.first(1), quantized_state));
                const auto quantized_values = ops::require(quantized.data<float>());
                for (std::size_t index = 0; index < quantized_values.size(); ++index)
                    if (!std::isfinite(quantized_values[index]) ||
                        std::abs(quantized_values[index] - values[index]) > 0.01F)
                        return 1;
                ops::require(model->set_checkpoint(checkpoint));
                auto restored_state = ops::require(model->create_state(8));
                const auto restored = ops::require(model->forward(tokens.first(1), restored_state));
                const auto restored_values = ops::require(restored.data<float>());
                for (std::size_t index = 0; index < restored_values.size(); ++index)
                    if (!std::isfinite(restored_values[index]) ||
                        std::abs(restored_values[index] - values[index]) > 2e-4F)
                        return 1;
                auto extended_config = YAML::Clone(config);
                extended_config["max_position_embeddings"] = 384;
                auto extended = ops::require(model::Gemma4Impl::create(extended_config));
                ops::require(extended->set_checkpoint(checkpoint));
                std::vector<std::int32_t> sequence(259);
                for (std::size_t index = 0; index < sequence.size(); ++index)
                    sequence[index] = tokens[index % tokens.size()];
                auto prefix_state = ops::require(extended->create_state(384));
                const auto prefix = ops::require(extended->forward(sequence, prefix_state));
                auto history_state = ops::require(extended->create_state(384));
                history_state.crop_local_attention = false;
                ops::require(extended->prefill(std::span(sequence).first(258), history_state));
                const auto history = ops::require(extended->forward(std::span(sequence).last(1), history_state));
                const auto history_values = ops::require(history.data<float>());
                auto bucket_state = ops::require(extended->create_state(384));
                ops::require(extended->prefill(std::span(sequence).first(128), bucket_state));
                ops::require(extended->prefill(std::span(sequence).subspan(128, 130), bucket_state));
                const auto bucket = ops::require(extended->forward(std::span(sequence).last(1), bucket_state));
                const auto prefix_values = ops::require(prefix.data<float>()),
                           bucket_values = ops::require(bucket.data<float>());
                for (std::size_t index = 0; index < bucket_values.size(); ++index)
                    if (!std::isfinite(bucket_values[index]) ||
                        std::abs(bucket_values[index] - prefix_values[index]) > 2e-4F ||
                        std::abs(bucket_values[index] - history_values[index]) > 2e-4F)
                        return 1;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}