#include "kidi/audio/wav.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>

namespace kidi::audio {
namespace {

template <typename Integer>
auto write_little_endian(std::ostream& output, Integer value) -> void {
    std::array<char, sizeof(Integer)> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<char>((value >> (index * 8)) & 0xff);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

} // namespace

auto save_wav(const std::filesystem::path& path, std::span<const float> samples, std::uint32_t sample_rate)
    -> Result<void> {
    if (samples.empty() || !sample_rate)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "WAV output requires samples and a sample rate"});
    if (samples.size() > (std::numeric_limits<std::uint32_t>::max() - 36) / sizeof(std::int16_t))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "WAV output is too large"});
    if (std::ranges::any_of(samples, [](float value) { return !std::isfinite(value); }))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "WAV output contains a non-finite sample"});

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return std::unexpected(Error{ErrorCode::IO, "cannot create WAV output: " + path.string()});

    const auto data_bytes = static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
    output.write("RIFF", 4);
    write_little_endian(output, static_cast<std::uint32_t>(36 + data_bytes));
    output.write("WAVEfmt ", 8);
    write_little_endian(output, std::uint32_t{16});
    write_little_endian(output, std::uint16_t{1});
    write_little_endian(output, std::uint16_t{1});
    write_little_endian(output, sample_rate);
    write_little_endian(output, static_cast<std::uint32_t>(sample_rate * sizeof(std::int16_t)));
    write_little_endian(output, std::uint16_t{sizeof(std::int16_t)});
    write_little_endian(output, std::uint16_t{16});
    output.write("data", 4);
    write_little_endian(output, data_bytes);
    for (const float sample : samples) {
        const auto clipped = std::clamp(sample, -1.F, 1.F);
        const auto encoded = static_cast<std::int16_t>(
            std::lrint(clipped * static_cast<float>(std::numeric_limits<std::int16_t>::max())));
        write_little_endian(output, static_cast<std::uint16_t>(encoded));
    }
    output.close();
    if (!output) return std::unexpected(Error{ErrorCode::IO, "failed to write WAV output: " + path.string()});
    return {};
}

} // namespace kidi::audio
