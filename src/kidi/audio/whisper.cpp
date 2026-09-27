#include "kidi/audio/whisper.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numbers>

#include <yaml-cpp/yaml.h>

namespace kidi::audio {
namespace {

constexpr std::size_t SAMPLE_RATE = 16000;
constexpr std::size_t FFT_SIZE = 400;
constexpr std::size_t HOP_LENGTH = 160;
constexpr std::size_t MEL_BINS = 80;
constexpr std::size_t FREQUENCY_BINS = FFT_SIZE / 2 + 1;
constexpr std::size_t SAMPLE_COUNT = 480000;
constexpr std::size_t FRAME_COUNT = 3000;

struct FftCache {
    std::array<float, FFT_SIZE> sine{}, cosine{}, hann{};

    FftCache() {
        for (std::size_t index = 0; index < FFT_SIZE; ++index) {
            const auto angle = 2.0 * std::numbers::pi * index / FFT_SIZE;
            sine[index] = std::sin(angle);
            cosine[index] = std::cos(angle);
            hann[index] = static_cast<float>(0.5 * (1.0 - std::cos(angle)));
        }
    }
};

const FftCache FFT_CACHE;

auto dft(const float* input, int size, float* output) -> void {
    const int step = FFT_SIZE / size;
    for (int frequency = 0; frequency < size; ++frequency) {
        float real = 0, imaginary = 0;
        for (int sample = 0; sample < size; ++sample) {
            const auto index = (frequency * sample * step) % FFT_SIZE;
            real += input[sample] * FFT_CACHE.cosine[index];
            imaginary -= input[sample] * FFT_CACHE.sine[index];
        }
        output[2 * frequency] = real;
        output[2 * frequency + 1] = imaginary;
    }
}

auto fft(float* input, int size, float* output) -> void {
    if (size == 1) {
        output[0] = input[0];
        output[1] = 0;
        return;
    }
    const int half = size / 2;
    if (size != 2 * half) {
        dft(input, size, output);
        return;
    }
    auto* split = input + size;
    for (int index = 0; index < half; ++index) split[index] = input[2 * index];
    auto* even = output + 2 * size;
    fft(split, half, even);
    for (int index = 0; index < half; ++index) split[index] = input[2 * index + 1];
    auto* odd = even + size;
    fft(split, half, odd);
    const int step = FFT_SIZE / size;
    for (int frequency = 0; frequency < half; ++frequency) {
        const auto index = frequency * step;
        const float real = FFT_CACHE.cosine[index], imaginary = -FFT_CACHE.sine[index];
        const float odd_real = odd[2 * frequency], odd_imaginary = odd[2 * frequency + 1];
        output[2 * frequency] = even[2 * frequency] + real * odd_real - imaginary * odd_imaginary;
        output[2 * frequency + 1] = even[2 * frequency + 1] + real * odd_imaginary + imaginary * odd_real;
        output[2 * (frequency + half)] = even[2 * frequency] - real * odd_real + imaginary * odd_imaginary;
        output[2 * (frequency + half) + 1] = even[2 * frequency + 1] - real * odd_imaginary - imaginary * odd_real;
    }
}

} // namespace

auto load_wav(const std::filesystem::path& path) -> Result<Waveform> {
    std::ifstream input(path, std::ios::binary);
    if (!input) return std::unexpected(Error{ErrorCode::IO, "cannot open audio file: " + path.string()});
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size < 0) return std::unexpected(Error{ErrorCode::IO, "cannot size audio file: " + path.string()});
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(bytes.data()), size))
        return std::unexpected(Error{ErrorCode::IO, "cannot read audio file: " + path.string()});
    const auto equal = [&](std::size_t offset, const char* value) {
        return offset + 4 <= bytes.size() && std::memcmp(bytes.data() + offset, value, 4) == 0;
    };
    const auto u16 = [&](std::size_t offset) {
        return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[offset]) |
                                          std::to_integer<std::uint8_t>(bytes[offset + 1]) << 8);
    };
    const auto u32 = [&](std::size_t offset) {
        return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[offset]) |
                                          std::to_integer<std::uint8_t>(bytes[offset + 1]) << 8 |
                                          std::to_integer<std::uint8_t>(bytes[offset + 2]) << 16 |
                                          std::to_integer<std::uint8_t>(bytes[offset + 3]) << 24);
    };
    if (bytes.size() < 12 || !equal(0, "RIFF") || !equal(8, "WAVE"))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "audio input is not a RIFF/WAVE file"});
    std::uint16_t format = 0, channels = 0, bits = 0;
    std::uint32_t sample_rate = 0;
    std::span<const std::byte> data;
    for (std::size_t offset = 12; offset + 8 <= bytes.size();) {
        const auto chunk_size = u32(offset + 4);
        if (offset + 8 + chunk_size > bytes.size())
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "truncated WAVE chunk"});
        if (equal(offset, "fmt ") && chunk_size >= 16) {
            format = u16(offset + 8);
            channels = u16(offset + 10);
            sample_rate = u32(offset + 12);
            bits = u16(offset + 22);
        } else if (equal(offset, "data")) {
            data = std::span(bytes).subspan(offset + 8, chunk_size);
        }
        offset += 8 + chunk_size + (chunk_size & 1);
    }
    if ((format != 1 && format != 3) || !channels || !sample_rate || !bits || data.empty() ||
        (format == 1 && bits != 16 && bits != 24 && bits != 32) || (format == 3 && bits != 32))
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported WAVE sample format"});
    const auto bytes_per_sample = bits / 8;
    const auto frame_bytes = static_cast<std::size_t>(channels) * bytes_per_sample;
    if (data.size() % frame_bytes)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "misaligned WAVE sample data"});
    Waveform result{{}, sample_rate};
    result.samples.resize(data.size() / frame_bytes);
    for (std::size_t frame = 0; frame < result.samples.size(); ++frame) {
        double sum = 0;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const auto offset = frame * frame_bytes + channel * bytes_per_sample;
            if (format == 3) {
                const auto raw = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[offset]) |
                                                            std::to_integer<std::uint8_t>(data[offset + 1]) << 8 |
                                                            std::to_integer<std::uint8_t>(data[offset + 2]) << 16 |
                                                            std::to_integer<std::uint8_t>(data[offset + 3]) << 24);
                sum += std::bit_cast<float>(raw);
            } else {
                std::int32_t value = 0;
                for (std::size_t byte = 0; byte < bytes_per_sample; ++byte)
                    value |= std::to_integer<std::uint8_t>(data[offset + byte]) << (byte * 8);
                const auto shift = 32 - bits;
                value = value << shift >> shift;
                sum += static_cast<double>(value) / static_cast<double>(std::uint64_t{1} << (bits - 1));
            }
        }
        result.samples[frame] = static_cast<float>(sum / channels);
    }
    return result;
}

auto WhisperFeatureExtractor::load(const std::filesystem::path& path) -> Result<WhisperFeatureExtractor> {
    try {
        if (!std::filesystem::is_regular_file(path))
            return std::unexpected(Error{ErrorCode::IO, "preprocessor config does not exist: " + path.string()});
        const auto config = YAML::LoadFile(path.string());
        if (config["feature_extractor_type"].as<std::string>() != "WhisperFeatureExtractor" ||
            config["sampling_rate"].as<std::size_t>() != SAMPLE_RATE ||
            config["feature_size"].as<std::size_t>() != MEL_BINS ||
            config["hop_length"].as<std::size_t>() != HOP_LENGTH || config["n_fft"].as<std::size_t>() != FFT_SIZE ||
            config["n_samples"].as<std::size_t>() != SAMPLE_COUNT ||
            config["nb_max_frames"].as<std::size_t>() != FRAME_COUNT || config["mel_filters"].size() != MEL_BINS)
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported Whisper preprocessing configuration"});
        std::vector<float> filters;
        filters.reserve(MEL_BINS * FREQUENCY_BINS);
        for (const auto& row : config["mel_filters"]) {
            if (row.size() != FREQUENCY_BINS)
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid Whisper mel filter shape"});
            for (const auto& value : row) filters.push_back(value.as<float>());
        }
        return WhisperFeatureExtractor(std::move(filters));
    } catch (const YAML::Exception& error) {
        return std::unexpected(
            Error{ErrorCode::INVALID_MANIFEST, "invalid Whisper preprocessor config: " + std::string(error.what())});
    }
}

auto WhisperFeatureExtractor::extract(std::span<const float> waveform, std::uint32_t sample_rate) const
    -> Result<WhisperFeatures> {
    if (sample_rate != SAMPLE_RATE)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "Whisper requires 16 kHz samples"});
    if (waveform.empty()) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "audio waveform is empty"});
    if (waveform.size() > SAMPLE_COUNT)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "Whisper audio exceeds the 30-second limit"});
    const auto count = waveform.size();
    std::vector<float> samples(SAMPLE_COUNT);
    std::ranges::copy(waveform.first(count), samples.begin());
    std::vector<float> padded(SAMPLE_COUNT + FFT_SIZE);
    std::ranges::copy(samples, padded.begin() + FFT_SIZE / 2);
    for (std::size_t index = 0; index < FFT_SIZE / 2; ++index) {
        padded[FFT_SIZE / 2 - 1 - index] = samples[1 + index];
        padded[FFT_SIZE / 2 + SAMPLE_COUNT + index] = samples[SAMPLE_COUNT - 2 - index];
    }

    WhisperFeatures result{{}, MEL_BINS, FRAME_COUNT};
    result.values.resize(MEL_BINS * FRAME_COUNT);
    std::array<float, FFT_SIZE * 2> fft_input{};
    std::array<float, FFT_SIZE * 8> fft_output{};
    std::array<float, FREQUENCY_BINS> power{};
    for (std::size_t frame = 0; frame < FRAME_COUNT; ++frame) {
        for (std::size_t index = 0; index < FFT_SIZE; ++index)
            fft_input[index] = FFT_CACHE.hann[index] * padded[frame * HOP_LENGTH + index];
        fft(fft_input.data(), FFT_SIZE, fft_output.data());
        for (std::size_t frequency = 0; frequency < FREQUENCY_BINS; ++frequency) {
            const auto real = fft_output[2 * frequency], imaginary = fft_output[2 * frequency + 1];
            power[frequency] = real * real + imaginary * imaginary;
        }
        for (std::size_t mel = 0; mel < MEL_BINS; ++mel) {
            double sum = 0;
            for (std::size_t frequency = 0; frequency < FREQUENCY_BINS; ++frequency)
                sum += power[frequency] * filters_[mel * FREQUENCY_BINS + frequency];
            result.values[mel * FRAME_COUNT + frame] = static_cast<float>(std::log10(std::max(sum, 1e-10)));
        }
    }
    const auto floor = *std::ranges::max_element(result.values) - 8.F;
    for (auto& value : result.values) value = (std::max(value, floor) + 4.F) / 4.F;
    return result;
}

} // namespace kidi::audio