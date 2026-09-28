#include "kidi/image/gemma4.h"

#include <tahoma/vision/codec.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace kidi::image {
namespace {
struct Filter {
    std::int32_t start;
    std::vector<double> weights;
};

auto cubic(double value) -> double {
    value = std::abs(value);
    if (value < 1) return ((1.5 * value - 2.5) * value) * value + 1;
    if (value < 2) return ((-0.5 * value + 2.5) * value - 4) * value + 2;
    return 0;
}

auto filters(std::int32_t input, std::int32_t output) -> std::vector<Filter> {
    std::vector<Filter> result;
    const double ratio = static_cast<double>(input) / output;
    const double scale = std::max(1.0, ratio);
    for (std::int32_t index = 0; index < output; ++index) {
        const auto center = (index + 0.5) * ratio;
        const auto start = std::max(0, static_cast<int>(center - 2 * scale + 0.5));
        const auto end = std::min(input, static_cast<int>(center + 2 * scale + 0.5));
        Filter filter{start, {}};
        double total = 0;
        for (auto source = start; source < end; ++source) {
            const auto weight = cubic((source + 0.5 - center) / scale);
            filter.weights.push_back(weight);
            total += weight;
        }
        for (auto& weight : filter.weights) weight /= total;
        result.push_back(std::move(filter));
    }
    return result;
}
} // namespace

auto prepare_gemma4(std::span<const std::uint8_t> encoded, std::int32_t soft_tokens) -> Result<Gemma4Image> {
    constexpr std::array allowed{70, 140, 280, 560, 1120};
    if (encoded.empty() || encoded.size() > 32 * 1024 * 1024 ||
        std::ranges::find(allowed, soft_tokens) == allowed.end())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid image size or Gemma image token budget"});
    try {
        const auto decoded =
            tahoma::vision::decode(encoded, {.max_pixels = 64 * 1024 * 1024, .max_decoded_bytes = 192 * 1024 * 1024});
        if (decoded.width > 16384 || decoded.height > 16384)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "image dimensions exceed 16384 pixels"});
        const auto factor =
            std::sqrt(static_cast<double>(soft_tokens) * 9 * 16 * 16 / (decoded.width * decoded.height));
        auto height = static_cast<std::int32_t>(std::floor(decoded.height * factor / 48)) * 48;
        auto width = static_cast<std::int32_t>(std::floor(decoded.width * factor / 48)) * 48;
        if (!height) {
            height = 48;
            width = static_cast<std::int32_t>(
                std::min<std::int64_t>((decoded.width / decoded.height) * 48, soft_tokens * 48));
        } else if (!width) {
            width = 48;
            height = static_cast<std::int32_t>(
                std::min<std::int64_t>((decoded.height / decoded.width) * 48, soft_tokens * 48));
        }
        const auto horizontal = filters(decoded.width, width), vertical = filters(decoded.height, height);
        std::vector<float> intermediate(static_cast<std::size_t>(decoded.height) * width * 3);
        for (std::int64_t row = 0; row < decoded.height; ++row)
            for (std::int32_t column = 0; column < width; ++column)
                for (std::int32_t channel = 0; channel < 3; ++channel) {
                    double sum = 0;
                    const auto& filter = horizontal[column];
                    for (std::size_t offset = 0; offset < filter.weights.size(); ++offset)
                        sum += filter.weights[offset] *
                               decoded.pixels[row * decoded.row_stride + (filter.start + offset) * 3 + channel];
                    intermediate[(row * width + column) * 3 + channel] = static_cast<float>(sum);
                }
        Gemma4Image result{height / 16, width / 16, std::vector<float>(static_cast<std::size_t>(height) * width * 3)};
        for (std::int32_t row = 0; row < height; ++row)
            for (std::int32_t column = 0; column < width; ++column)
                for (std::int32_t channel = 0; channel < 3; ++channel) {
                    double sum = 0;
                    const auto& filter = vertical[row];
                    for (std::size_t offset = 0; offset < filter.weights.size(); ++offset)
                        sum += filter.weights[offset] *
                               intermediate[((filter.start + offset) * width + column) * 3 + channel];
                    const auto patch = (row / 16) * result.patch_columns + column / 16;
                    const auto pixel = ((row % 16) * 16 + column % 16) * 3 + channel;
                    result.patches[patch * 768 + pixel] =
                        static_cast<float>(std::clamp(std::nearbyint(sum), 0.0, 255.0) / 255.0);
                }
        return result;
    } catch (const tahoma::vision::CodecError& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}
} // namespace kidi::image