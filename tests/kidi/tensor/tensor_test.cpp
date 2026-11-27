#include <algorithm>
#include <bit>
#include <array>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "kidi/tensor/backend.h"
#include "kidi/tensor/external.h"
#include "kidi/tensor/tensor.h"

#include <gtest/gtest.h>

namespace {

using kidi::tensor::BackendRegistry;
using kidi::tensor::Device;
using kidi::tensor::DeviceKind;
using kidi::tensor::DType;
using kidi::tensor::Tensor;

/// Serves rows of a three-row table whose row `r` holds bytes `10 * r + column`.
class GeneratedRows final : public kidi::tensor::RowSource {
public:
    auto gather(std::span<const std::int32_t> rows, std::span<std::byte> destination) const
        -> kidi::Result<void> override {
        const auto width = destination.size() / rows.size();
        for (std::size_t index = 0; index < rows.size(); ++index)
            for (std::size_t column = 0; column < width; ++column)
                destination[index * width + column] = static_cast<std::byte>(10 * rows[index] + column);
        ++calls;
        return {};
    }
    mutable int calls = 0;
};

constexpr std::array<float, 6> VALUES = {1, 2, 3, 4, 5, 6};

auto values_tensor() -> Tensor {
    auto tensor = Tensor::from_host({2, 3}, std::span<const float>(VALUES));
    if (!tensor) throw std::runtime_error(tensor.error().message);
    return *std::move(tensor);
}

TEST(TensorTest, DeviceCapabilitiesAndPolicies) {
    static_assert(&Device::cuda().capabilities() == &Device::cuda(7).capabilities());
    static_assert(&Device::cuda().policy() == &Device::cuda(7).policy());
    static_assert(Device::cpu().capabilities().supports_audio_transformer());
    static_assert(Device::cuda().capabilities().supports_audio_transformer());
    static_assert(Device::web_gpu().capabilities().supports_audio_transformer());
    static_assert(!Device::apple_gpu().capabilities().supports_audio_transformer());
    static_assert(!Device::vulkan().capabilities().supports_audio_transformer());
    static_assert(!Device::qualcomm_npu().capabilities().supports_audio_transformer());
    static_assert(Device::web_gpu().capabilities().memory.deferred());
    static_assert(!Device::web_gpu().capabilities().memory.requires_staging());
    static_assert(Device::cuda().capabilities().memory.requires_staging());
    static_assert(Device::cpu().capabilities().compute.host());
    static_assert(!Device::cuda().capabilities().compute.host());
    static_assert(Device::cuda().capabilities().sampling.diffusion);
    static_assert(Device::cuda().capabilities().compute.embedding_sum);
    static_assert(Device::web_gpu().capabilities().compute.embedding_sum);
    static_assert(Device::cuda().capabilities().supports_diffusion_feedback());
    static_assert(Device::web_gpu().capabilities().supports_diffusion_feedback());
    static_assert(!Device::cpu().capabilities().supports_diffusion_feedback());
    static_assert(!Device::apple_gpu().capabilities().supports_diffusion_feedback());
    static_assert(!Device::vulkan().capabilities().supports_diffusion_feedback());
    constexpr kidi::tensor::DeviceCapabilities DIFFUSION_ONLY{.sampling = {.diffusion = true}};
    constexpr kidi::tensor::DeviceCapabilities EMBEDDINGS_ONLY{.compute = {.embedding_sum = true}};
    static_assert(!DIFFUSION_ONLY.supports_diffusion_feedback());
    static_assert(!EMBEDDINGS_ONLY.supports_diffusion_feedback());
    static_assert(Device::apple_gpu().policy().audio.codec_tokens == kidi::tensor::CodecTokenInput::HOST_STAGED);
    static_assert(Device::cpu().policy().audio.codec_tokens == kidi::tensor::CodecTokenInput::DEVICE);
    static_assert(Device::cuda().policy().audio.codec_tokens == kidi::tensor::CodecTokenInput::DEVICE);
    static_assert(Device::web_gpu().policy().audio.codec_tokens == kidi::tensor::CodecTokenInput::DEVICE);
    static_assert(Device::vulkan().policy().audio.codec_tokens == kidi::tensor::CodecTokenInput::DEVICE);
    static_assert(Device::cuda().policy().transformer.prefill == kidi::tensor::PrefillStrategy::FULL);
    static_assert(Device::apple_gpu().policy().transformer.prefill == kidi::tensor::PrefillStrategy::SHARED_KV_TAIL);
    static_assert(Device::cuda().parameter_device(kidi::tensor::ParameterRole::AUDIO) == Device::cpu());
    static_assert(Device::cuda(7).parameter_device(kidi::tensor::ParameterRole::EMBEDDING) == Device::cuda(7));
    static_assert(Device::cuda(7).parameter_device(kidi::tensor::ParameterRole::PROJECTION) == Device::cuda(7));
    static_assert(!Device::cuda().capabilities().sampling.supports(kidi::tensor::TokenSelection::BATCHED_DEVICE));
    static_assert(Device::apple_gpu().capabilities().sampling.supports(kidi::tensor::TokenSelection::BATCHED_DEVICE));
    static_assert(Device::apple_gpu().capabilities().sampling.supports(kidi::tensor::TokenSelection::DEVICE));
    static_assert(!Device::cpu().capabilities().sampling.supports(kidi::tensor::TokenSelection::DEVICE));
    static_assert(!Device::cuda().capabilities().compute.supports(kidi::tensor::TransformerSupport::GENERAL));
    static_assert(Device::apple_gpu().capabilities().compute.supports(kidi::tensor::TransformerSupport::GENERAL));
    static_assert(Device::apple_gpu().capabilities().compute.supports(kidi::tensor::TransformerSupport::QUANTIZED));
    static_assert(!Device::cuda().capabilities().compute.supports_native_calibrated_packed(8));
    static_assert(!Device::apple_gpu().capabilities().compute.supports_native_calibrated_packed(4));
    static_assert(Device::apple_gpu().capabilities().compute.supports_native_calibrated_packed(8));
    static_assert(Device::web_gpu().capabilities().compute.supports_native_calibrated_packed(2));
    static_assert(!Device::web_gpu().capabilities().compute.supports_native_calibrated_packed(3));
    static_assert(Device::vulkan().capabilities().compute.supports_native_calibrated_packed(4));
    static_assert(Device::apple_gpu().parameter_device(kidi::tensor::ParameterRole::EMBEDDING) == Device::cpu());
    static_assert(Device::apple_gpu().parameter_device(kidi::tensor::ParameterRole::PROJECTION,
                                                       kidi::tensor::ParameterKind::QUANTIZED) == Device::cpu());
    static_assert(Device::apple_gpu().parameter_device(kidi::tensor::ParameterRole::PROJECTION) == Device::apple_gpu());
}

TEST(TensorTest, BackendRegistry) {
    const auto backends = BackendRegistry::instance().backends();
    ASSERT_EQ(backends.size(), static_cast<std::size_t>(DeviceKind::COUNT)) << "missing tensor backend family";
    EXPECT_EQ(backends[0].device_kind, DeviceKind::CPU) << "CPU must use the available YNNPACK backend";
    EXPECT_TRUE(backends[0].storage_available) << "CPU must use the available YNNPACK backend";
    EXPECT_TRUE(backends[0].execution_available) << "CPU must use the available YNNPACK backend";
    EXPECT_EQ(backends[0].name, "ynnpack") << "CPU must use the available YNNPACK backend";
}

TEST(TensorTest, HostTensorMetadataAndViews) {
    auto tensor = Tensor::from_host({2, 3}, std::span<const float>(VALUES));
    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;
    EXPECT_TRUE(tensor->defined()) << "unexpected CPU tensor metadata";
    EXPECT_EQ(tensor->dtype(), DType::F32) << "unexpected CPU tensor metadata";
    EXPECT_EQ(tensor->device(), Device::cpu()) << "unexpected CPU tensor metadata";
    EXPECT_EQ(tensor->backend_name(), "ynnpack") << "unexpected CPU tensor metadata";
    EXPECT_TRUE(std::ranges::equal(tensor->shape(), std::array<std::int64_t, 2>{2, 3}))
        << "unexpected CPU tensor metadata";
    EXPECT_TRUE(std::ranges::equal(tensor->strides(), std::array<std::int64_t, 2>{3, 1}))
        << "unexpected CPU tensor metadata";
    EXPECT_EQ(tensor->numel(), VALUES.size()) << "unexpected CPU tensor metadata";
    EXPECT_EQ(tensor->nbytes(), sizeof(VALUES)) << "unexpected CPU tensor metadata";
    EXPECT_TRUE(tensor->is_contiguous()) << "unexpected CPU tensor metadata";
    EXPECT_TRUE(tensor->is_host_accessible()) << "unexpected CPU tensor metadata";

    auto values = tensor->data<float>();
    ASSERT_TRUE(values.has_value()) << "CPU tensor data mismatch";
    EXPECT_TRUE(std::ranges::equal(*values, VALUES)) << "CPU tensor data mismatch";

    auto reshaped = tensor->reshape({3, 2});
    ASSERT_TRUE(reshaped.has_value()) << "reshape metadata mismatch";
    EXPECT_TRUE(std::ranges::equal(reshaped->shape(), std::array<std::int64_t, 2>{3, 2})) << "reshape metadata mismatch";
    EXPECT_TRUE(std::ranges::equal(reshaped->strides(), std::array<std::int64_t, 2>{2, 1}))
        << "reshape metadata mismatch";
    (*values)[0] = 9;
    auto reshaped_values = reshaped->data<float>();
    ASSERT_TRUE(reshaped_values.has_value()) << "reshape must alias storage";
    EXPECT_EQ((*reshaped_values)[0], 9) << "reshape must alias storage";

    auto row = tensor->select(0, 1);
    ASSERT_TRUE(row.has_value()) << "select metadata mismatch";
    EXPECT_TRUE(std::ranges::equal(row->shape(), std::array<std::int64_t, 1>{3})) << "select metadata mismatch";
    EXPECT_EQ(row->storage_offset(), 3) << "select metadata mismatch";
    auto row_values = row->data<float>();
    ASSERT_TRUE(row_values.has_value()) << "selected tensor data mismatch";
    EXPECT_TRUE(std::ranges::equal(*row_values, VALUES | std::views::drop(3))) << "selected tensor data mismatch";
    auto columns = tensor->narrow(1, 1, 2);
    ASSERT_TRUE(columns.has_value()) << "inner-axis narrow must remain a non-contiguous view";
    EXPECT_FALSE(columns->is_contiguous()) << "inner-axis narrow must remain a non-contiguous view";
    EXPECT_FALSE(columns->host_bytes().has_value()) << "inner-axis narrow must remain a non-contiguous view";

    std::array<float, 3> copied_row{};
    const auto row_copy = row->copy_to_host_later(std::as_writable_bytes(std::span(copied_row)));
    EXPECT_TRUE(row_copy.has_value()) << "deferred host copy must preserve a contiguous view's offset";
    EXPECT_TRUE(std::ranges::equal(copied_row, *row_values))
        << "deferred host copy must preserve a contiguous view's offset";
    std::array<float, 4> oversized{};
    EXPECT_FALSE(Tensor{}.copy_to_host_later({}).has_value())
        << "deferred host copies must reject undefined tensors, excess bytes, and non-contiguous views";
    EXPECT_FALSE(row->copy_to_host_later(std::as_writable_bytes(std::span(oversized))).has_value())
        << "deferred host copies must reject undefined tensors, excess bytes, and non-contiguous views";
    EXPECT_FALSE(columns->copy_to_host_later(std::as_writable_bytes(std::span(copied_row))).has_value())
        << "deferred host copies must reject undefined tensors, excess bytes, and non-contiguous views";

    auto same_device = tensor->to(Device::cpu());
    ASSERT_TRUE(same_device.has_value()) << same_device.error().message;
    auto same_values = same_device->data<float>();
    ASSERT_TRUE(same_values.has_value()) << "same-device transfer must retain storage";
    EXPECT_EQ(same_values->data(), values->data()) << "same-device transfer must retain storage";
}

TEST(TensorTest, LargeZeroAllocation) {
    auto large_zero = Tensor::zeros({2 * 1024 * 1024}, DType::U8);
    ASSERT_TRUE(large_zero.has_value()) << large_zero.error().message;
    auto zero_bytes = large_zero->data<std::uint8_t>();
    ASSERT_TRUE(zero_bytes) << "large zero allocation is not initially zero";
    EXPECT_EQ((*zero_bytes)[0], 0) << "large zero allocation is not initially zero";
    EXPECT_EQ((*zero_bytes)[zero_bytes->size() / 2], 0) << "large zero allocation is not initially zero";
    EXPECT_EQ((*zero_bytes)[zero_bytes->size() - 1], 0) << "large zero allocation is not initially zero";
    (*zero_bytes)[0] = 7;
    (*zero_bytes)[zero_bytes->size() - 1] = 9;
    EXPECT_EQ((*zero_bytes)[0], 7) << "large zero allocation is not writable";
    EXPECT_EQ((*zero_bytes)[zero_bytes->size() - 1], 9) << "large zero allocation is not writable";
}

TEST(TensorTest, MetalTransfer) {
    const auto backends = BackendRegistry::instance().backends();
    const auto metal =
        std::ranges::find_if(backends, [](const auto& info) { return info.device_kind == DeviceKind::A_GPU; });
    if (metal == backends.end() || !metal->storage_available) GTEST_SKIP() << "Metal backend unavailable";
    auto tensor = values_tensor();
    auto values = tensor.data<float>();
    ASSERT_TRUE(values.has_value());
    EXPECT_TRUE(metal->execution_available) << "Metal backend must expose MPSGraph execution";
    auto gpu = tensor.to(Device::apple_gpu());
    ASSERT_TRUE(gpu.has_value()) << gpu.error().message;
    EXPECT_EQ(gpu->device(), Device::apple_gpu()) << "unexpected Apple GPU tensor metadata";
    EXPECT_EQ(gpu->backend_name(), "metal-mps") << "unexpected Apple GPU tensor metadata";
    auto round_trip = gpu->to(Device::cpu());
    ASSERT_TRUE(round_trip.has_value()) << round_trip.error().message;
    auto round_trip_values = round_trip->data<float>();
    ASSERT_TRUE(round_trip_values.has_value()) << "Metal tensor transfer changed values";
    EXPECT_TRUE(std::ranges::equal(*round_trip_values, *values)) << "Metal tensor transfer changed values";
}

TEST(TensorTest, HighRankMetadata) {
    auto invalid_shape = Tensor::empty({-1}, DType::F32);
    auto high_rank = Tensor::empty(std::vector<std::int64_t>(10, 1), DType::F32);
    ASSERT_TRUE(high_rank.has_value()) << "high-rank tensor allocation failed";
    auto high_view = high_rank->select(0, 0);
    ASSERT_TRUE(high_view) << "high-rank metadata copy failed";
    EXPECT_EQ(high_view->dimensions(), 9) << "high-rank metadata copy failed";
    EXPECT_TRUE(high_view->is_contiguous()) << "high-rank metadata copy failed";
    auto inline_view = high_view->select(0, 0);
    ASSERT_TRUE(inline_view) << "metadata inline transition failed";
    EXPECT_EQ(inline_view->dimensions(), 8) << "metadata inline transition failed";
    EXPECT_TRUE(inline_view->is_contiguous()) << "metadata inline transition failed";
    EXPECT_FALSE(invalid_shape.has_value()) << "negative shape unexpectedly succeeded";
}

TEST(TensorTest, BackendTransfers) {
    const auto backends = BackendRegistry::instance().backends();
    auto tensor = values_tensor();
    auto values = tensor.data<float>();
    ASSERT_TRUE(values.has_value());
    for (const auto device : {Device::qualcomm_npu(), Device::cuda()}) {
        SCOPED_TRACE(static_cast<int>(device.kind));
        const auto available = std::ranges::any_of(backends, [&](const auto& backend) {
            return backend.device_kind == device.kind && backend.storage_available;
        });
        auto transferred = tensor.to(device);
        if (available) {
            ASSERT_TRUE(transferred) << "available backend transfer failed";
            EXPECT_EQ(transferred->device(), device) << "available backend transfer failed";
            std::array<float, VALUES.size()> copied{};
            const auto copy = transferred->copy_to_host_later(std::as_writable_bytes(std::span(copied)));
            EXPECT_TRUE(copy.has_value()) << "backend-neutral host copy changed values";
            EXPECT_TRUE(std::ranges::equal(copied, *values)) << "backend-neutral host copy changed values";
            const auto restored = transferred->to(Device::cpu());
            ASSERT_TRUE(restored.has_value()) << "backend round-trip transfer failed";
            const auto restored_values = restored->data<float>();
            ASSERT_TRUE(restored_values) << "backend round-trip transfer changed values";
            EXPECT_TRUE(std::ranges::equal(*restored_values, *values)) << "backend round-trip transfer changed values";
        } else {
            ASSERT_FALSE(transferred) << "unavailable backend transfer succeeded";
            EXPECT_EQ(transferred.error().code, kidi::ErrorCode::UNSUPPORTED) << "unavailable backend transfer succeeded";
            EXPECT_FALSE(transferred.error().message.empty()) << "unavailable backend transfer succeeded";
        }
    }
}

TEST(TensorTest, HostBlobs) {
    auto blob_owner = std::make_shared<std::array<float, 2>>(std::array<float, 2>{7, 8});
    auto blob = Tensor::from_blob({2}, DType::F32, std::as_bytes(std::span<const float>(*blob_owner)), blob_owner);
    blob_owner.reset();
    ASSERT_TRUE(blob.has_value()) << "read-only host blob must reject mutable access";
    EXPECT_FALSE(blob->data<float>().has_value()) << "read-only host blob must reject mutable access";
    const auto& read_only_blob = *blob;
    auto blob_values = read_only_blob.data<float>();
    ASSERT_TRUE(blob_values.has_value()) << "read-only host blob lost its external owner";
    EXPECT_EQ((*blob_values)[0], 7) << "read-only host blob lost its external owner";
    EXPECT_EQ((*blob_values)[1], 8) << "read-only host blob lost its external owner";
    auto mutable_owner = std::make_shared<std::array<float, 2>>(std::array<float, 2>{3, 4});
    auto mutable_blob = Tensor::from_mutable_blob(
        {2}, DType::F32, std::as_writable_bytes(std::span<float>(*mutable_owner)), mutable_owner);
    mutable_owner.reset();
    ASSERT_TRUE(mutable_blob.has_value()) << "mutable host blob construction failed";
    auto mutable_values = mutable_blob->data<float>();
    ASSERT_TRUE(mutable_values.has_value()) << "mutable host blob rejected mutable access";
    (*mutable_values)[1] = 9;
    const auto& readable_mutable_blob = *mutable_blob;
    auto updated_values = readable_mutable_blob.data<float>();
    ASSERT_TRUE(updated_values.has_value()) << "mutable host blob lost its external owner or write";
    EXPECT_EQ((*updated_values)[0], 3) << "mutable host blob lost its external owner or write";
    EXPECT_EQ((*updated_values)[1], 9) << "mutable host blob lost its external owner or write";
}

TEST(TensorTest, RowGather) {
    auto rows = std::make_shared<GeneratedRows>();
    const auto table = kidi::tensor::external_tensor({3, 4}, DType::U8, rows);
    std::array<std::byte, 8> gathered{};
    const std::array<std::int32_t, 2> selected{2, 0};
    ASSERT_TRUE(table.has_value()) << "external tables must be shaped CPU tensors without host addresses";
    EXPECT_TRUE(kidi::tensor::is_external(*table))
        << "external tables must be shaped CPU tensors without host addresses";
    EXPECT_EQ(table->nbytes(), 12) << "external tables must be shaped CPU tensors without host addresses";
    EXPECT_EQ(table->device(), Device::cpu()) << "external tables must be shaped CPU tensors without host addresses";
    EXPECT_FALSE(table->host_bytes().has_value())
        << "external tables must be shaped CPU tensors without host addresses";
    EXPECT_FALSE(table->to(Device::vulkan()).has_value())
        << "external tables must be shaped CPU tensors without host addresses";
    EXPECT_TRUE(kidi::tensor::gather_rows(*table, selected, gathered).has_value())
        << "external row gather returned the wrong rows";
    EXPECT_EQ(rows->calls, 1) << "external row gather returned the wrong rows";
    EXPECT_EQ(gathered[0], std::byte{20}) << "external row gather returned the wrong rows";
    EXPECT_EQ(gathered[3], std::byte{23}) << "external row gather returned the wrong rows";
    EXPECT_EQ(gathered[4], std::byte{0}) << "external row gather returned the wrong rows";
    const std::array<std::int32_t, 1> outside{3};
    EXPECT_FALSE(kidi::tensor::gather_rows(*table, outside, std::span(gathered).first(4)).has_value())
        << "row gathers must reject rows outside the table and mismatched destinations";
    EXPECT_FALSE(kidi::tensor::gather_rows(*table, selected, std::span(gathered).first(4)).has_value())
        << "row gathers must reject rows outside the table and mismatched destinations";
    EXPECT_EQ(rows->calls, 1) << "row gathers must reject rows outside the table and mismatched destinations";
    const auto host_table = Tensor::from_host({2, 3}, std::span<const float>(VALUES));
    std::array<std::byte, sizeof(float) * 3> host_row{};
    const std::array<std::int32_t, 1> second{1};
    ASSERT_TRUE(host_table.has_value()) << "host row gather returned the wrong row";
    EXPECT_FALSE(kidi::tensor::is_external(*host_table)) << "host row gather returned the wrong row";
    EXPECT_TRUE(kidi::tensor::gather_rows(*host_table, second, host_row).has_value())
        << "host row gather returned the wrong row";
    EXPECT_EQ((std::bit_cast<std::array<float, 3>>(host_row)[0]), 4) << "host row gather returned the wrong row";
}

} // namespace
