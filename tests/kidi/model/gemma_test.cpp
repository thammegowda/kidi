#include "kidi/model/gemma.h"

#include <cmath>
#include <filesystem>
#include <iostream>

auto main() -> int {
    using namespace kidi;
    try {
        const auto directory = std::filesystem::path(KIDI_GEMMA_FIXTURE);
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
                if (!std::isfinite(prefill_values[index]) || std::abs(prefill_values[index] - values[index]) > 2e-4F) {
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
                if (!std::isfinite(tail_values[index]) || std::abs(tail_values[index] - reference_tail[index]) > 2e-4F)
                    return 1;
            const std::array<std::int32_t, 1> invalid{-1};
            if (model->forward(invalid, incremental) || incremental.position != tokens.size()) return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}