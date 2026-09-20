#include "kidi/layers/transformer.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kidi/model/weights.h"
#include "kidi/tensor/tensor.h"

namespace {

auto write_weights(const std::filesystem::path& path) -> void {
    std::string header =
        R"({"norm.weight":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"norm.bias":{"dtype":"F32","shape":[4],"data_offsets":[16,32]},"ff.w_1.weight":{"dtype":"F32","shape":[4,3],"data_offsets":[32,80]},"ff.w_1.bias":{"dtype":"F32","shape":[3],"data_offsets":[80,92]},"ff.w_2.weight":{"dtype":"F32","shape":[3,4],"data_offsets":[92,140]},"ff.w_2.bias":{"dtype":"F32","shape":[4],"data_offsets":[140,156]}})";
    while (header.size() % 8 != 0) header.push_back(' ');

    std::ofstream output(path, std::ios::binary);
    std::uint64_t header_size = header.size();
    if constexpr (std::endian::native == std::endian::big) header_size = std::byteswap(header_size);
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    constexpr std::array NORMALIZATION_WEIGHT = {1.5F, 0.5F, 2.0F, -1.0F};
    constexpr std::array NORMALIZATION_BIAS = {0.1F, -0.2F, 0.3F, 0.4F};
    constexpr std::array FIRST_WEIGHT = {
        0.25F, -1.0F, 0.6F, -0.5F, 0.5F, -0.2F, 1.0F, 0.125F, -0.4F, 0.75F, 0.25F, 0.9F,
    };
    constexpr std::array FIRST_BIAS = {0.2F, -0.1F, 0.3F};
    constexpr std::array SECOND_WEIGHT = {
        0.3F, 1.0F, -0.4F, 0.25F, -0.7F, 0.1F, 0.8F, 0.5F, 0.2F, -0.5F, 0.6F, -1.0F,
    };
    constexpr std::array SECOND_BIAS = {-0.2F, 0.4F, 0.05F, -0.3F};
    const auto write_values = [&output](const auto& values) {
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(values[0])));
    };
    write_values(NORMALIZATION_WEIGHT);
    write_values(NORMALIZATION_BIAS);
    write_values(FIRST_WEIGHT);
    write_values(FIRST_BIAS);
    write_values(SECOND_WEIGHT);
    write_values(SECOND_BIAS);
}

auto near(float left, float right) -> bool { return std::abs(left - right) < 2.0e-5F; }

} // namespace

auto main() -> int {
    using namespace kidi;
    try {
        const auto path = std::filesystem::temp_directory_path() / "kidi-transformer-builder-test.safetensors";
        write_weights(path);
        auto weights = ops::require(model::Weights::load(path));
        layers::LayerNorm norm = std::make_shared<layers::LayerNormImpl>(weights, "norm", 1e-5F);
        layers::Linear first = std::make_shared<layers::LinearImpl>(weights, "ff.w_1", model::WeightEncoding::F32);
        layers::Linear second = std::make_shared<layers::LinearImpl>(weights, "ff.w_2", model::WeightEncoding::F32);
        ModuleList<layers::LinearImpl> projections = std::make_shared<ModuleListImpl<layers::LinearImpl>>();
        projections->push_back(first);
        projections->push_back(second);
        ModuleMap<> modules = std::make_shared<ModuleMapImpl<>>();
        modules->insert("projections", projections);
        modules->insert("norm", norm);
        const auto snapshot = modules->state_dict();
        if (snapshot.size() != 6 || !snapshot.contains("projections.0.weight") || !snapshot.contains("norm.bias"))
            return 1;
        const std::array input_values{-1.F, 0.5F, 2.F, 3.5F, 4.F, -2.F, 1.F, 0.25F};
        const std::array normalized_expected{-1.9124613F, -0.4236068F, 1.1944273F, -0.9416408F,
                                             2.3279426F,  -0.8552770F, 0.4747319F, 0.6621111F};
        const std::array linear_expected{0.4218847F, 1.5145510F, -2.0880032F, 2.1809392F, -2.6307118F, 2.2738280F};
        const std::array gelu_expected{0.279898256F, 1.416187644F,  -0.038417075F,
                                       2.149120331F, -0.011207701F, 2.247702360F};
        const std::array output_expected{-1.115045309F, 0.840725541F, 1.047940612F, 0.516485453F,
                                         0.902121961F,  1.424148321F, 0.530007124F, -2.016026020F};
        std::vector devices{tensor::Device::cpu()};
#if defined(__APPLE__)
        devices.push_back(tensor::Device::apple_gpu());
#endif
        for (auto device : devices) {
            ops::Context context(device);
            auto input =
                ops::require(tensor::Tensor::from_host({1, 2, 4}, std::span<const float>(input_values), device));
            auto normalized = norm->forward(context, input);
            const auto residual_result = norm->forward_residual(context, input, input);
            auto separate_sum = context.add(input, input);
            auto separate_norm = norm->forward(context, separate_sum);
            const auto& first_impl = *first;
            auto projected = first_impl.forward(context, normalized);
            auto activated = context.gelu(projected);
            auto output = second->forward(context, activated);
            context.synchronize();
            for (std::size_t result = 0; result < 2; ++result) {
                const auto fused = ops::require(residual_result[result].data<float>());
                const auto separate = ops::require((result == 0 ? separate_sum : separate_norm).data<float>());
                for (std::size_t index = 0; index < fused.size(); ++index)
                    if (!near(fused[index], separate[index])) return 1;
            }
            const std::array actual{normalized, projected, activated, output};
            const std::array<std::span<const float>, 4> expected{normalized_expected, linear_expected, gelu_expected,
                                                                 output_expected};
            for (std::size_t result = 0; result < actual.size(); ++result) {
                auto values = ops::require(actual[result].data<float>());
                for (std::size_t index = 0; index < values.size(); ++index)
                    if (!near(values[index], expected[result][index])) {
                        std::cerr << "eager primitive mismatch " << result << ' ' << index << '\n';
                        return 1;
                    }
            }
            auto replacement = snapshot;
            replacement["projections.0.weight"] = ops::require(tensor::Tensor::zeros({4, 3}, tensor::DType::F32));
            auto invalid = replacement;
            invalid["projections.1.bias"] = ops::require(tensor::Tensor::zeros({1}, tensor::DType::F32));
            if (modules->load_state_dict(invalid)) return 1;
            auto unchanged = first->forward(context, normalized);
            context.synchronize();
            auto original_values = ops::require(unchanged.data<float>());
            for (std::size_t index = 0; index < original_values.size(); ++index)
                if (!near(original_values[index], linear_expected[index])) return 1;
            ops::require(modules->load_state_dict(replacement));
            auto external_weight = ops::require(replacement.at("projections.0.weight").data<float>());
            std::fill(external_weight.begin(), external_weight.end(), 7.F);
            auto changed = projections->at(0)->forward(context, normalized);
            context.synchronize();
            auto changed_values = ops::require(changed.data<float>());
            const std::array bias{0.2F, -0.1F, 0.3F};
            for (std::size_t index = 0; index < changed_values.size(); ++index)
                if (!near(changed_values[index], bias[index % 3])) return 1;
            ops::require(modules->load_state_dict(snapshot));
            auto restored = first_impl.forward(context, normalized);
            context.synchronize();
            auto restored_values = ops::require(restored.data<float>());
            for (std::size_t index = 0; index < restored_values.size(); ++index)
                if (!near(restored_values[index], linear_expected[index])) return 1;
        }
        auto missing = snapshot;
        missing.erase("norm.bias");
        if (modules->load_state_dict(missing)) return 1;
        ops::require(modules->load_state_dict(missing, false));
        auto extra = snapshot;
        extra.emplace("unknown", snapshot.begin()->second);
        if (modules->load_state_dict(extra)) return 1;
        bool cycle_rejected = false;
        try {
            modules->insert("cycle", modules);
        } catch (const std::invalid_argument&) {
            cycle_rejected = true;
        }
        if (!cycle_rejected) return 1;
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
