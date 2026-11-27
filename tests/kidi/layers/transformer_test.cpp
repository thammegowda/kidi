#include "kidi/layers/transformer.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <thread>

#include "kidi/checkpoint/weights.h"
#include "kidi/tensor/tensor.h"

#include <gtest/gtest.h>

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

constexpr float TOLERANCE = 2.0e-5F;

class DeferredParameter : public kidi::Module {
public:
    explicit DeferredParameter(bool allocate) {
        register_parameter("weight", weight_, {4, 3}, kidi::tensor::DType::F32, allocate);
        register_parameter("tied", tied_, {4, 3}, kidi::tensor::DType::F32, false);
        tie_parameter("tied", "weight");
    }

private:
    kidi::tensor::Tensor weight_;
    kidi::tensor::Tensor tied_;
};

TEST(TransformerTest, DefaultModuleDevice) {
    using namespace kidi;
    const auto default_device = module_device;
    EXPECT_EQ(default_device, default_module_device());
    for (const auto& backend : tensor::BackendRegistry::instance().backends())
        if (backend.device_kind == tensor::DeviceKind::A_GPU && backend.execution_available)
            EXPECT_EQ(default_device, tensor::Device::apple_gpu()) << "an available Apple GPU is the default device";
    EXPECT_TRUE(core::default_module_device().capabilities().compute.supports(tensor::TransformerSupport::GENERAL));
}

TEST(TransformerTest, ModuleScopes) {
    using namespace kidi;
    const auto default_device = module_device;
    const ModuleScope cpu_parameters(tensor::Device::cpu());
    {
        const ModuleScope indexed(tensor::DType::F32, false, tensor::Device::cuda(7));
        layers::Linear pending(4, 3);
        EXPECT_EQ(pending->device(), tensor::Device::cuda(7));
        EXPECT_FALSE(pending->state_dict().at("weight").defined());
        {
            const ModuleScope nested(tensor::Device::cpu());
            EXPECT_EQ(module_device, tensor::Device::cpu());
            EXPECT_FALSE(allocate_parameters);
        }
        EXPECT_EQ(module_device, pending->device());
        EXPECT_FALSE(allocate_parameters);
    }
    EXPECT_EQ(module_device, tensor::Device::cpu());
    EXPECT_TRUE(allocate_parameters);
    layers::Linear scoped{nullptr};
    {
        const ModuleScope outer(tensor::DType::BF16, false, default_device);
        scoped = layers::Linear(4, 3);
        EXPECT_EQ(scoped->device(), default_device);
        EXPECT_FALSE(scoped->state_dict().at("weight").defined());
        {
            const ModuleScope inner(tensor::DType::F32, true, tensor::Device::cpu());
            layers::FeedForward nested(4, 8);
            for (const auto& [name, value] : nested->state_dict()) {
                SCOPED_TRACE(name);
                EXPECT_TRUE(value.defined());
                EXPECT_EQ(value.dtype(), tensor::DType::F32);
                EXPECT_EQ(value.device(), tensor::Device::cpu());
            }
        }
        EXPECT_EQ(module_dtype, tensor::DType::BF16);
        EXPECT_FALSE(allocate_parameters);
        EXPECT_EQ(module_device, default_device);
        bool isolated = false;
        std::thread worker([&] {
            isolated = module_dtype == tensor::DType::F32 && allocate_parameters && module_device == default_device;
            const ModuleScope local(tensor::DType::I8, true, tensor::Device::cpu());
            layers::LinearImpl child(4, 3, true);
            const auto state = child.state_dict();
            isolated = isolated && state.at("weight").dtype() == tensor::DType::I8 &&
                       state.at("weight").size(0) == 3 && state.at("weight").size(1) == 4 &&
                       state.at("scale").defined() && child.device() == tensor::Device::cpu();
        });
        worker.join();
        EXPECT_TRUE(isolated) << "module scopes are thread-local";
        EXPECT_EQ(module_dtype, tensor::DType::BF16);
        EXPECT_FALSE(allocate_parameters);
        EXPECT_EQ(module_device, default_device);
        EXPECT_THROW(
            {
                const ModuleScope failure(tensor::DType::F16, true, tensor::Device::cpu());
                layers::LinearImpl invalid(4, 3, true);
            },
            ops::Failure);
        EXPECT_EQ(module_dtype, tensor::DType::BF16);
        EXPECT_FALSE(allocate_parameters);
        EXPECT_EQ(module_device, default_device);
    }
    EXPECT_EQ(module_dtype, tensor::DType::F32);
    EXPECT_TRUE(allocate_parameters);
    EXPECT_EQ(module_device, tensor::Device::cpu());
    StateDict scoped_state{{"weight", ops::require(tensor::Tensor::zeros({4, 3}, tensor::DType::BF16))},
                           {"bias", ops::require(tensor::Tensor::zeros({3}, tensor::DType::F32))}};
    ops::require(scoped->set_state(scoped_state));
    EXPECT_EQ(scoped->state_dict().at("weight").device(), default_device);
    ops::require(scoped->set_state(scoped_state, true, true));
    EXPECT_EQ(scoped->state_dict().at("weight").device(), default_device);
    {
        const ModuleScope selected(default_device);
        layers::EmbeddingImpl allocated(3, 4, 8);
        EXPECT_EQ(allocated.state_dict().at("weight").device(), default_device);
    }
}

TEST(TransformerTest, ModuleHolder) {
    using namespace kidi;
    const ModuleScope cpu_parameters(tensor::Device::cpu());
    layers::Linear scratch(4, 3);
    auto shared = scratch;
    layers::Linear empty(nullptr);
    EXPECT_TRUE(static_cast<bool>(scratch));
    EXPECT_FALSE(static_cast<bool>(empty));
    EXPECT_EQ(shared.get(), scratch.get());
    EXPECT_EQ(&*scratch, scratch.ptr().get());
    auto moved = std::move(shared);
    EXPECT_FALSE(static_cast<bool>(shared));
    EXPECT_EQ(moved.get(), scratch.get());
    auto scratch_state = scratch->state_dict();
    auto scratch_values = ops::require(scratch_state.at("weight").data<float>());
    std::mt19937 generator(7);
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    for (auto& value : scratch_values) value = distribution(generator);
    ops::Context scratch_context;
    const std::array scratch_input{1.F, 2.F, 3.F, 4.F};
    auto scratch_tensor = ops::require(tensor::Tensor::from_host({1, 4}, std::span<const float>(scratch_input)));
    auto scratch_output = scratch->forward(scratch_context, scratch_tensor);
    scratch_context.synchronize();
    const auto scratch_result = ops::require(scratch_output.data<float>());
    for (std::size_t channel = 0; channel < 3; ++channel) {
        float expected = 0;
        for (std::size_t input = 0; input < 4; ++input)
            expected += scratch_input[input] * scratch_values[input * 3 + channel];
        EXPECT_NEAR(scratch_result[channel], expected, TOLERANCE) << "channel " << channel;
    }
}

TEST(TransformerTest, UnboundLinearRejects) {
    using namespace kidi;
    const ModuleScope cpu_parameters(tensor::Device::cpu());
    const ModuleScope construction(tensor::DType::F32, false);
    ops::Context scratch_context;
    const std::array scratch_input{1.F, 2.F, 3.F, 4.F};
    auto scratch_tensor = ops::require(tensor::Tensor::from_host({1, 4}, std::span<const float>(scratch_input)));
    layers::Linear unbound(4, 3);
    EXPECT_THROW(unbound->forward(scratch_context, scratch_tensor), ops::Failure);
}

class ModulesTest : public ::testing::Test {
protected:
    void SetUp() override {
        using namespace kidi;
        write_weights(path);
        projections->push_back(first);
        projections->push_back(second);
        modules->insert("projections", projections);
        modules->insert("norm", norm);
        const std::array mappings{
            checkpoint::StateMappingSpec{{R"(ff\.w_1\.(weight|bias))"}, "projections.0.$1"},
            checkpoint::StateMappingSpec{{R"(ff\.w_2\.(weight|bias))"}, "projections.1.$1"}};
        mapped.emplace(ops::require(checkpoint::Weights::load(path, mappings)));
        ops::require(modules->set_state(ops::require(mapped->state_dict())));
        snapshot = modules->state_dict();
    }

    void TearDown() override { std::filesystem::remove(path); }

    std::filesystem::path path = std::filesystem::temp_directory_path() / "kidi-transformer-builder-test.safetensors";
    const kidi::ModuleScope cpu_parameters{kidi::tensor::Device::cpu()};
    const kidi::ModuleScope construction{kidi::tensor::DType::F32, false};
    kidi::layers::LayerNorm norm{4, 1e-5F};
    kidi::layers::Linear first{4, 3};
    kidi::layers::Linear second{3, 4};
    kidi::ModuleList<kidi::layers::LinearImpl> projections;
    kidi::ModuleMap<> modules;
    std::optional<kidi::checkpoint::Weights> mapped;
    kidi::StateDict snapshot;
};

TEST_F(ModulesTest, DeferredParameters) {
    using namespace kidi;
    auto weights = ops::require(checkpoint::Weights::load(path));
    ModuleList<DeferredParameter> deferred;
    deferred->push_back(ModuleHolder<DeferredParameter>(false));
    deferred->push_back(ModuleHolder<DeferredParameter>(false));
    EXPECT_FALSE(deferred->state_dict().at("0.weight").defined());
    const StateDict initial{{"0.weight", ops::require(weights.tensor("ff.w_1.weight"))},
                            {"1.weight", ops::require(weights.tensor("ff.w_1.weight"))}};
    auto bad_initial = initial;
    bad_initial["1.weight"] = ops::require(tensor::Tensor::zeros({1}, tensor::DType::F32));
    EXPECT_FALSE(deferred->set_state(bad_initial));
    EXPECT_FALSE(deferred->state_dict().at("0.weight").defined());
    ops::require(deferred->set_state(initial));
    const auto bound = deferred->state_dict();
    EXPECT_EQ(ops::require(bound.at("0.weight").host_bytes()).data(),
              ops::require(initial.at("0.weight").host_bytes()).data());
    ops::require(deferred->set_state(initial, true, true));
    const auto copied = deferred->state_dict();
    EXPECT_EQ(ops::require(copied.at("0.weight").host_bytes()).data(),
              ops::require(copied.at("0.tied").host_bytes()).data());
    EXPECT_NE(ops::require(copied.at("0.weight").host_bytes()).data(),
              ops::require(initial.at("0.weight").host_bytes()).data());
    DeferredParameter allocated(true);
    EXPECT_TRUE(allocated.state_dict().at("weight").defined());
}

TEST_F(ModulesTest, BuildsModuleTree) {
    EXPECT_EQ(projections->at(0).get(), first.get());
    EXPECT_EQ(modules->at("norm").get(), norm.get());
    EXPECT_EQ(snapshot.size(), 6U);
    EXPECT_TRUE(snapshot.contains("projections.0.weight"));
    EXPECT_TRUE(snapshot.contains("norm.bias"));
}

TEST_F(ModulesTest, ForwardAndStateSwaps) {
    using namespace kidi;
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
        auto input = ops::require(tensor::Tensor::from_host({1, 2, 4}, std::span<const float>(input_values), device));
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
                EXPECT_NEAR(fused[index], separate[index], TOLERANCE)
                    << "fused residual mismatch " << result << ' ' << index;
        }
        const std::array actual{normalized, projected, activated, output};
        const std::array<std::span<const float>, 4> expected{normalized_expected, linear_expected, gelu_expected,
                                                             output_expected};
        for (std::size_t result = 0; result < actual.size(); ++result) {
            auto values = ops::require(actual[result].data<float>());
            for (std::size_t index = 0; index < values.size(); ++index)
                EXPECT_NEAR(values[index], expected[result][index], TOLERANCE)
                    << "eager primitive mismatch " << result << ' ' << index;
        }
        auto replacement = snapshot;
        replacement["projections.0.weight"] = ops::require(tensor::Tensor::zeros({4, 3}, tensor::DType::F32));
        auto invalid = replacement;
        invalid["projections.1.bias"] = ops::require(tensor::Tensor::zeros({1}, tensor::DType::F32));
        EXPECT_FALSE(modules->set_state(invalid, true, true));
        auto unchanged = first->forward(context, normalized);
        context.synchronize();
        auto original_values = ops::require(unchanged.data<float>());
        for (std::size_t index = 0; index < original_values.size(); ++index)
            EXPECT_NEAR(original_values[index], linear_expected[index], TOLERANCE) << "unchanged index " << index;
        ops::require(modules->set_state(replacement, true, true));
        auto external_weight = ops::require(replacement.at("projections.0.weight").data<float>());
        std::fill(external_weight.begin(), external_weight.end(), 7.F);
        auto changed = projections->at(0)->forward(context, normalized);
        context.synchronize();
        auto changed_values = ops::require(changed.data<float>());
        const std::array bias{0.2F, -0.1F, 0.3F};
        for (std::size_t index = 0; index < changed_values.size(); ++index)
            EXPECT_NEAR(changed_values[index], bias[index % 3], TOLERANCE) << "changed index " << index;
        ops::require(modules->set_state(snapshot, true, true));
        auto restored = first_impl.forward(context, normalized);
        context.synchronize();
        auto restored_values = ops::require(restored.data<float>());
        for (std::size_t index = 0; index < restored_values.size(); ++index)
            EXPECT_NEAR(restored_values[index], linear_expected[index], TOLERANCE) << "restored index " << index;
    }
}

TEST_F(ModulesTest, RejectsInvalidState) {
    using namespace kidi;
    auto missing = snapshot;
    missing.erase("norm.bias");
    EXPECT_FALSE(modules->set_state(missing, true, true));
    ops::require(modules->set_state(missing, false, true));
    auto extra = snapshot;
    extra.emplace("unknown", snapshot.begin()->second);
    EXPECT_FALSE(modules->set_state(extra, true, true));
    EXPECT_THROW(modules->insert("cycle", modules), std::invalid_argument);
}

} // namespace
