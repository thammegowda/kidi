#include <algorithm>
#include <bit>
#include <array>
#include <cstdint>
#include <iostream>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

#include "kidi/tensor/backend.h"
#include "kidi/tensor/external.h"
#include "kidi/tensor/tensor.h"

namespace {

auto require(bool condition, std::string_view message) -> bool {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

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

} // namespace

auto main() -> int {
    using kidi::tensor::BackendRegistry;
    using kidi::tensor::Device;
    using kidi::tensor::DeviceKind;
    using kidi::tensor::DType;
    using kidi::tensor::Tensor;

    const auto backends = BackendRegistry::instance().backends();
    if (!require(backends.size() == static_cast<std::size_t>(DeviceKind::COUNT), "missing tensor backend family"))
        return 1;
    if (!require(backends[0].device_kind == DeviceKind::CPU && backends[0].storage_available &&
                     backends[0].execution_available && backends[0].name == "ynnpack",
                 "CPU must use the available YNNPACK backend")) {
        return 1;
    }

    constexpr std::array<float, 6> VALUES = {1, 2, 3, 4, 5, 6};
    auto tensor = Tensor::from_host({2, 3}, std::span<const float>(VALUES));
    if (!require(tensor.has_value(), tensor ? "" : tensor.error().message)) return 1;
    if (!require(tensor->defined() && tensor->dtype() == DType::F32 && tensor->device() == Device::cpu() &&
                     tensor->backend_name() == "ynnpack" &&
                     std::ranges::equal(tensor->shape(), std::array<std::int64_t, 2>{2, 3}) &&
                     std::ranges::equal(tensor->strides(), std::array<std::int64_t, 2>{3, 1}) &&
                     tensor->numel() == VALUES.size() && tensor->nbytes() == sizeof(VALUES) &&
                     tensor->is_contiguous() && tensor->is_host_accessible(),
                 "unexpected CPU tensor metadata")) {
        return 1;
    }

    auto values = tensor->data<float>();
    if (!require(values.has_value() && std::ranges::equal(*values, VALUES), "CPU tensor data mismatch")) return 1;

    auto large_zero = Tensor::zeros({2 * 1024 * 1024}, DType::U8);
    if (!require(large_zero.has_value(), large_zero ? "" : large_zero.error().message)) return 1;
    auto zero_bytes = large_zero->data<std::uint8_t>();
    if (!require(zero_bytes && (*zero_bytes)[0] == 0 && (*zero_bytes)[zero_bytes->size() / 2] == 0 &&
                     (*zero_bytes)[zero_bytes->size() - 1] == 0,
                 "large zero allocation is not initially zero"))
        return 1;
    (*zero_bytes)[0] = 7;
    (*zero_bytes)[zero_bytes->size() - 1] = 9;
    if (!require((*zero_bytes)[0] == 7 && (*zero_bytes)[zero_bytes->size() - 1] == 9,
                 "large zero allocation is not writable"))
        return 1;

    auto reshaped = tensor->reshape({3, 2});
    if (!require(reshaped.has_value() && std::ranges::equal(reshaped->shape(), std::array<std::int64_t, 2>{3, 2}) &&
                     std::ranges::equal(reshaped->strides(), std::array<std::int64_t, 2>{2, 1}),
                 "reshape metadata mismatch")) {
        return 1;
    }
    (*values)[0] = 9;
    auto reshaped_values = reshaped->data<float>();
    if (!require(reshaped_values.has_value() && (*reshaped_values)[0] == 9, "reshape must alias storage")) return 1;

    auto row = tensor->select(0, 1);
    if (!require(row.has_value() && std::ranges::equal(row->shape(), std::array<std::int64_t, 1>{3}) &&
                     row->storage_offset() == 3,
                 "select metadata mismatch")) {
        return 1;
    }
    auto row_values = row->data<float>();
    if (!require(row_values.has_value() && std::ranges::equal(*row_values, VALUES | std::views::drop(3)),
                 "selected tensor data mismatch")) {
        return 1;
    }
    auto columns = tensor->narrow(1, 1, 2);
    if (!require(columns.has_value() && !columns->is_contiguous() && !columns->host_bytes().has_value(),
                 "inner-axis narrow must remain a non-contiguous view")) {
        return 1;
    }

    auto same_device = tensor->to(Device::cpu());
    if (!require(same_device.has_value(), same_device ? "" : same_device.error().message)) return 1;
    auto same_values = same_device->data<float>();
    if (!require(same_values.has_value() && same_values->data() == values->data(),
                 "same-device transfer must retain storage")) {
        return 1;
    }

    const auto metal =
        std::ranges::find_if(backends, [](const auto& info) { return info.device_kind == DeviceKind::A_GPU; });
    if (metal != backends.end() && metal->storage_available) {
        if (!require(metal->execution_available, "Metal backend must expose MPSGraph execution")) return 1;
        auto gpu = tensor->to(Device::apple_gpu());
        if (!require(gpu.has_value(), gpu ? "" : gpu.error().message)) return 1;
        if (!require(gpu->device() == Device::apple_gpu() && gpu->backend_name() == "metal-mps",
                     "unexpected Apple GPU tensor metadata")) {
            return 1;
        }
        auto round_trip = gpu->to(Device::cpu());
        if (!require(round_trip.has_value(), round_trip ? "" : round_trip.error().message)) return 1;
        auto round_trip_values = round_trip->data<float>();
        if (!require(round_trip_values.has_value() && std::ranges::equal(*round_trip_values, *values),
                     "Metal tensor transfer changed values")) {
            return 1;
        }
    }

    auto invalid_shape = Tensor::empty({-1}, DType::F32);
    auto high_rank = Tensor::empty(std::vector<std::int64_t>(10, 1), DType::F32);
    if (!require(high_rank.has_value(), "high-rank tensor allocation failed")) return 1;
    auto high_view = high_rank->select(0, 0);
    if (!require(high_view && high_view->dimensions() == 9 && high_view->is_contiguous(),
                 "high-rank metadata copy failed"))
        return 1;
    auto inline_view = high_view->select(0, 0);
    if (!require(inline_view && inline_view->dimensions() == 8 && inline_view->is_contiguous(),
                 "metadata inline transition failed"))
        return 1;
    if (!require(!invalid_shape.has_value(), "negative shape unexpectedly succeeded")) return 1;
    for (const auto device : {Device::qualcomm_npu(), Device::cuda()}) {
        auto unavailable = tensor->to(device);
        if (!require(!unavailable.has_value() && unavailable.error().code == kidi::ErrorCode::UNSUPPORTED &&
                         unavailable.error().message.find("not implemented") != std::string::npos,
                     "placeholder backend transfer unexpectedly succeeded")) {
            return 1;
        }
    }
    auto blob_owner = std::make_shared<std::array<float, 2>>(std::array<float, 2>{7, 8});
    auto blob = Tensor::from_blob({2}, DType::F32, std::as_bytes(std::span<const float>(*blob_owner)), blob_owner);
    blob_owner.reset();
    if (!require(blob.has_value() && !blob->data<float>().has_value(),
                 "read-only host blob must reject mutable access")) {
        return 1;
    }
    const auto& read_only_blob = *blob;
    auto blob_values = read_only_blob.data<float>();
    if (!require(blob_values.has_value() && (*blob_values)[0] == 7 && (*blob_values)[1] == 8,
                 "read-only host blob lost its external owner")) {
        return 1;
    }

    auto rows = std::make_shared<GeneratedRows>();
    const auto table = kidi::tensor::external_tensor({3, 4}, DType::U8, rows);
    std::array<std::byte, 8> gathered{};
    const std::array<std::int32_t, 2> selected{2, 0};
    if (!require(table.has_value() && kidi::tensor::is_external(*table) && table->nbytes() == 12 &&
                     table->device() == Device::cpu() && !table->host_bytes().has_value() &&
                     !table->to(Device::vulkan()).has_value(),
                 "external tables must be shaped CPU tensors without host addresses")) {
        return 1;
    }
    if (!require(kidi::tensor::gather_rows(*table, selected, gathered).has_value() && rows->calls == 1 &&
                     gathered[0] == std::byte{20} && gathered[3] == std::byte{23} && gathered[4] == std::byte{0},
                 "external row gather returned the wrong rows")) {
        return 1;
    }
    const std::array<std::int32_t, 1> outside{3};
    if (!require(!kidi::tensor::gather_rows(*table, outside, std::span(gathered).first(4)).has_value() &&
                     !kidi::tensor::gather_rows(*table, selected, std::span(gathered).first(4)).has_value() &&
                     rows->calls == 1,
                 "row gathers must reject rows outside the table and mismatched destinations")) {
        return 1;
    }
    const auto host_table = Tensor::from_host({2, 3}, std::span<const float>(VALUES));
    std::array<std::byte, sizeof(float) * 3> host_row{};
    const std::array<std::int32_t, 1> second{1};
    if (!require(host_table.has_value() && !kidi::tensor::is_external(*host_table) &&
                     kidi::tensor::gather_rows(*host_table, second, host_row).has_value() &&
                     std::bit_cast<std::array<float, 3>>(host_row)[0] == 4,
                 "host row gather returned the wrong row")) {
        return 1;
    }
    return 0;
}