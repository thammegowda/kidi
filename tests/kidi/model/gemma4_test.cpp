#include "kidi/model/gemma4.h"

#include <cmath>
#include <filesystem>
#include <iostream>

auto main() -> int {
    using namespace kidi;
    try {
        for (const auto* fixture : {"gemma4", "gemma4-qat"}) {
            const auto directory = std::filesystem::path(KIDI_GEMMA4_FIXTURE).parent_path() / fixture;
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
                auto selected_state = ops::require(model->create_state(8));
                for (std::size_t position = 0; position < tokens.size(); ++position) {
                    auto output = ops::require(model->forward(std::span(tokens).subspan(position, 1), incremental));
                    const auto actual = ops::require(output.data<float>());
                    const auto selected =
                        ops::require(model->forward_token(std::span(tokens).subspan(position, 1), selected_state));
                    if (selected != std::ranges::max_element(actual) - actual.begin()) return 1;
                    for (std::size_t token = 0; token < actual.size(); ++token)
                        if (!std::isfinite(actual[token]) ||
                            std::abs(actual[token] - values[position * actual.size() + token]) > 2e-4F) {
                            std::cerr << "Gemma cached logits differ at " << position << ':' << token << '\n';
                            return 1;
                        }
                }
                if (incremental.position != tokens.size() || full.position != tokens.size()) return 1;
                for (const std::size_t batch_size : {2, 4}) {
                    std::vector<model::Gemma4State> batch, serial;
                    std::vector<std::size_t> order;
                    for (std::size_t row = 0; row < batch_size; ++row) {
                        auto source = ops::require(model->create_state(8));
                        ops::require(model->prefill(tokens.first(row + 1), source));
                        batch.push_back(ops::require(model->fork_state(source, row + 1, 8)));
                        serial.push_back(ops::require(model->fork_state(source, row + 1, 8)));
                        order.push_back(row);
                    }
                    std::array duplicate{&batch[0], &batch[0]};
                    if (model->forward_batch(std::array{tokens[0], tokens[1]}, duplicate) || batch[0].position != 1)
                        return 1;
                    for (std::size_t step = 0; step < 3; ++step) {
                        if (step == 1) std::ranges::reverse(order);
                        if (step == 2) order.pop_back();
                        std::vector<model::Gemma4State*> states;
                        std::vector<std::int32_t> input_ids;
                        std::vector<float> expected;
                        for (auto row : order) {
                            states.push_back(&batch[row]);
                            input_ids.push_back(tokens[(row + step) % tokens.size()]);
                            const auto output = ops::require(
                                model->forward(std::span<const std::int32_t>(&input_ids.back(), 1), serial[row]));
                            const auto row_values = ops::require(output.data<float>());
                            expected.insert(expected.end(), row_values.begin(), row_values.end());
                        }
                        const auto output = ops::require(model->forward_batch(input_ids, states));
                        const auto actual = ops::require(output.data<float>());
                        if (actual.size() != expected.size()) return 1;
                        for (std::size_t index = 0; index < actual.size(); ++index)
                            if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 2e-4F) {
                                std::cerr << "batched Gemma mismatch " << fixture << ' ' << tensor::to_string(device)
                                          << " batch " << batch_size << " step " << step << " index " << index << '\n';
                                return 1;
                            }
                        for (auto row : order)
                            if (batch[row].position != serial[row].position) return 1;
                    }
                }
                auto chunked = ops::require(model->create_state(8));
                ops::require(model->prefill(tokens.first(3), chunked));
                auto complete_prefix = ops::require(model->create_state(8));
                ops::require(model->forward(tokens.first(3), complete_prefix, true));
                if (chunked.position != complete_prefix.position) return 1;
                for (std::size_t producer = 0; producer < chunked.layers.size(); ++producer)
                    if (ops::require(chunked.layers[producer].key.copy_to_host()) !=
                            ops::require(complete_prefix.layers[producer].key.copy_to_host()) ||
                        ops::require(chunked.layers[producer].value.copy_to_host()) !=
                            ops::require(complete_prefix.layers[producer].value.copy_to_host())) {
                        std::cerr << "cache-only prefill differs from full evaluation at producer " << producer << '\n';
                        return 1;
                    }
                auto snapshot = ops::require(model->fork_state(chunked, 3, 3));
                const auto saved_key = ops::require(snapshot.layers[0].key.copy_to_host());
                auto resumed = ops::require(model->fork_state(snapshot, 3, 8));
                const auto resumed_output = ops::require(model->forward(tokens.subspan(3), resumed, true));
                const auto tail = ops::require(model->forward(tokens.subspan(3), chunked, true));
                const auto tail_values = ops::require(tail.data<float>());
                if (!std::ranges::equal(ops::require(resumed_output.data<float>()), tail_values) ||
                    saved_key != ops::require(snapshot.layers[0].key.copy_to_host()) ||
                    model->fork_state(snapshot, 4, 8) || model->fork_state(snapshot, 3, 2))
                    return 1;
                const auto reference_tail = values.last(tail_values.size());
                for (std::size_t index = 0; index < tail_values.size(); ++index)
                    if (!std::isfinite(tail_values[index]) ||
                        std::abs(tail_values[index] - reference_tail[index]) > 2e-4F)
                        return 1;
                const std::array<std::int32_t, 1> invalid{-1};
                if (model->forward(invalid, incremental) || incremental.position != tokens.size()) return 1;
                if (model->set_checkpoint(checkpoint, 4, 3)) return 1;
                ops::require(model->set_checkpoint(checkpoint));
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
                } else {
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
                }
                for (const std::size_t length : {259, 1027}) {
                    const auto capacity = ((length + 127) / 128) * 128;
                    auto extended_config = YAML::Clone(config);
                    extended_config["max_position_embeddings"] = capacity;
                    auto extended = ops::require(model::Gemma4Impl::create(extended_config));
                    ops::require(extended->set_checkpoint(checkpoint));
                    std::vector<std::int32_t> sequence(length);
                    for (std::size_t index = 0; index < sequence.size(); ++index)
                        sequence[index] = tokens[index % tokens.size()];
                    auto prefix_state = ops::require(extended->create_state(capacity));
                    const auto prefix = ops::require(extended->forward(sequence, prefix_state));
                    auto oracle_state = ops::require(extended->create_state(capacity));
                    const auto all_logits = ops::require(extended->forward(sequence, oracle_state, true));
                    const auto last_values = ops::require(prefix.data<float>());
                    const auto oracle_values = ops::require(all_logits.data<float>()).last(last_values.size());
                    for (std::size_t index = 0; index < last_values.size(); ++index)
                        if (!std::isfinite(last_values[index]) ||
                            std::abs(last_values[index] - oracle_values[index]) > 2e-4F)
                            return 1;
                    auto history_state = ops::require(extended->create_state(capacity));
                    history_state.crop_local_attention = false;
                    ops::require(extended->prefill(std::span(sequence).first(length - 1), history_state));
                    const auto history = ops::require(extended->forward(std::span(sequence).last(1), history_state));
                    const auto history_values = ops::require(history.data<float>());
                    auto bucket_state = ops::require(extended->create_state(capacity));
                    ops::require(extended->prefill(std::span(sequence).first(128), bucket_state));
                    ops::require(extended->prefill(std::span(sequence).subspan(128, length - 129), bucket_state));
                    const auto bucket = ops::require(extended->forward(std::span(sequence).last(1), bucket_state));
                    const auto prefix_values = ops::require(prefix.data<float>()),
                               bucket_values = ops::require(bucket.data<float>());
                    for (std::size_t index = 0; index < bucket_values.size(); ++index)
                        if (!std::isfinite(bucket_values[index]) ||
                            std::abs(bucket_values[index] - prefix_values[index]) > 2e-4F ||
                            std::abs(bucket_values[index] - history_values[index]) > 2e-4F)
                            return 1;
                    auto larger_state = ops::require(extended->create_state(capacity));
                    const std::size_t chunk = length == 259 ? 256 : 512;
                    for (std::size_t offset = 0; offset < length - 3; offset += chunk)
                        ops::require(extended->prefill(std::span(sequence).subspan(offset, chunk), larger_state));
                    const auto larger = ops::require(extended->forward(std::span(sequence).last(3), larger_state));
                    const auto larger_values = ops::require(larger.data<float>());
                    for (std::size_t index = 0; index < larger_values.size(); ++index)
                        if (!std::isfinite(larger_values[index]) ||
                            std::abs(larger_values[index] - prefix_values[index]) > 2e-4F)
                            return 1;
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}