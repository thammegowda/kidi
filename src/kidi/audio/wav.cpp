#include "kidi/audio/wav.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

namespace kidi::audio {
namespace {

template <typename Integer>
auto little_endian(const char* bytes) -> Integer {
    Integer result = 0;
    for (std::size_t index = 0; index < sizeof(Integer); ++index)
        result |= static_cast<Integer>(static_cast<unsigned char>(bytes[index])) << (index * 8);
    return result;
}

template <typename Integer>
auto write_little_endian(std::ostream& output, Integer value) -> void {
    std::array<char, sizeof(Integer)> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<char>((value >> (index * 8)) & 0xff);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

} // namespace

WavReader::WavReader(std::ifstream input, std::uint32_t sample_rate, std::uint16_t format, std::uint16_t channels,
                     std::uint16_t bits, std::uint64_t data_bytes)
    : input_(std::move(input)),
      sample_rate_(sample_rate),
      format_(format),
      channels_(channels),
      bits_(bits),
      remaining_bytes_(data_bytes),
      sample_count_(data_bytes / (channels * (bits / 8))) {}

auto WavReader::open(const std::filesystem::path& path) -> Result<WavReader> {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return std::unexpected(Error{ErrorCode::IO, "cannot open audio file: " + path.string()});
    const auto length = input.tellg();
    if (length < 0) return std::unexpected(Error{ErrorCode::IO, "cannot size audio file: " + path.string()});
    std::array<char, 12> header{};
    input.seekg(0);
    if (!input.read(header.data(), header.size()) || std::memcmp(header.data(), "RIFF", 4) ||
        std::memcmp(header.data() + 8, "WAVE", 4))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "audio input is not a RIFF/WAVE file"});
    const auto size = static_cast<std::uint64_t>(length);
    std::uint16_t format = 0, channels = 0, bits = 0;
    std::uint32_t sample_rate = 0, data_bytes = 0;
    std::uint64_t data_offset = 0;
    for (std::uint64_t offset = 12; offset + 8 <= size;) {
        std::array<char, 8> chunk{};
        input.seekg(static_cast<std::streamoff>(offset));
        if (!input.read(chunk.data(), chunk.size()))
            return std::unexpected(Error{ErrorCode::IO, "cannot read WAVE chunk"});
        const auto count = little_endian<std::uint32_t>(chunk.data() + 4);
        if (count > size - offset - 8)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "truncated WAVE chunk"});
        if (!std::memcmp(chunk.data(), "fmt ", 4) && count >= 16) {
            std::array<char, 16> description{};
            if (!input.read(description.data(), description.size()))
                return std::unexpected(Error{ErrorCode::IO, "cannot read WAVE format"});
            format = little_endian<std::uint16_t>(description.data());
            channels = little_endian<std::uint16_t>(description.data() + 2);
            sample_rate = little_endian<std::uint32_t>(description.data() + 4);
            bits = little_endian<std::uint16_t>(description.data() + 14);
            if (format == 0xfffe) {
                if (count < 40)
                    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "truncated WAVE extensible format"});
                std::array<char, 24> extension{};
                if (!input.read(extension.data(), extension.size()))
                    return std::unexpected(Error{ErrorCode::IO, "cannot read WAVE extensible format"});
                const auto extra = little_endian<std::uint16_t>(extension.data());
                if (extra < 22 || static_cast<std::uint32_t>(extra) + 18 > count)
                    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid WAVE extensible format length"});
                constexpr std::array<std::uint8_t, 14> SUBFORMAT_SUFFIX{0, 0, 0,    0, 0x10, 0,    0x80,
                                                                        0, 0, 0xaa, 0, 0x38, 0x9b, 0x71};
                if (std::memcmp(extension.data() + 10, SUBFORMAT_SUFFIX.data(), SUBFORMAT_SUFFIX.size()) ||
                    little_endian<std::uint16_t>(extension.data() + 2) != bits)
                    return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported WAVE extensible sample format"});
                format = little_endian<std::uint16_t>(extension.data() + 8);
            }
        } else if (!std::memcmp(chunk.data(), "data", 4)) {
            data_offset = offset + 8;
            data_bytes = count;
        }
        offset += 8 + static_cast<std::uint64_t>(count) + (count & 1);
    }
    if ((format != 1 && format != 3) || !channels || !sample_rate || !data_bytes ||
        (format == 1 && bits != 16 && bits != 24 && bits != 32) || (format == 3 && bits != 32))
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported WAVE sample format"});
    if (data_bytes % (static_cast<std::size_t>(channels) * (bits / 8)))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "misaligned WAVE sample data"});
    input.seekg(static_cast<std::streamoff>(data_offset));
    if (!input) return std::unexpected(Error{ErrorCode::IO, "cannot seek to WAVE sample data"});
    return WavReader(std::move(input), sample_rate, format, channels, bits, data_bytes);
}

auto WavReader::read(std::size_t maximum_samples) -> Result<std::vector<float>> {
    if (!maximum_samples) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "WAVE read size must be positive"});
    const auto bytes_per_sample = bits_ / 8;
    const auto frame_bytes = static_cast<std::size_t>(channels_) * bytes_per_sample;
    const auto count =
        static_cast<std::size_t>(std::min<std::uint64_t>(maximum_samples, remaining_bytes_ / frame_bytes));
    std::vector<char> bytes(count * frame_bytes);
    if (!bytes.empty() && !input_.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
        return std::unexpected(Error{ErrorCode::IO, "cannot read WAVE sample data"});
    remaining_bytes_ -= bytes.size();
    std::vector<float> samples(count);
    for (std::size_t frame = 0; frame < count; ++frame) {
        double sum = 0;
        for (std::size_t channel = 0; channel < channels_; ++channel) {
            const auto* source = bytes.data() + frame * frame_bytes + channel * bytes_per_sample;
            if (format_ == 3) {
                const auto value = std::bit_cast<float>(little_endian<std::uint32_t>(source));
                if (!std::isfinite(value))
                    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "WAVE contains a non-finite sample"});
                sum += value;
            } else {
                std::uint32_t value = 0;
                for (std::size_t byte = 0; byte < static_cast<std::size_t>(bytes_per_sample); ++byte)
                    value |= static_cast<std::uint32_t>(static_cast<unsigned char>(source[byte])) << (byte * 8);
                sum += static_cast<double>(std::bit_cast<std::int32_t>(value << (32 - bits_))) / 2147483648.0;
            }
        }
        samples[frame] = static_cast<float>(sum / channels_);
    }
    return samples;
}

auto load_wav(const std::filesystem::path& path) -> Result<Waveform> {
    auto reader = WavReader::open(path);
    if (!reader) return std::unexpected(reader.error());
    auto samples = reader->read(std::numeric_limits<std::size_t>::max());
    if (!samples) return std::unexpected(samples.error());
    return Waveform{std::move(*samples), reader->sample_rate()};
}

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
