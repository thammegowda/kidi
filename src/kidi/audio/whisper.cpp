#include "kidi/audio/whisper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include <yaml-cpp/yaml.h>

namespace kidi::audio {
namespace {

constexpr std::size_t SAMPLE_RATE = 16000;
constexpr std::size_t FFT_SIZE = 400;
constexpr std::size_t HOP_LENGTH = 160;
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

auto mel_filters(std::size_t bins) -> std::vector<float> {
    const auto log_step = std::log(6.4) / 27.0;
    const auto maximum_mel = 15.0 + std::log(8.0) / log_step;
    std::vector<double> edges(bins + 2);
    for (std::size_t index = 0; index < edges.size(); ++index) {
        const auto mel = maximum_mel * index / (bins + 1);
        edges[index] = mel < 15.0 ? mel * (200.0 / 3.0) : 1000.0 * std::exp(log_step * (mel - 15.0));
    }
    std::vector<float> filters(bins * FREQUENCY_BINS);
    for (std::size_t mel = 0; mel < bins; ++mel)
        for (std::size_t frequency = 0; frequency < FREQUENCY_BINS; ++frequency) {
            const auto hz = static_cast<double>(frequency * SAMPLE_RATE) / FFT_SIZE;
            const auto lower = (hz - edges[mel]) / (edges[mel + 1] - edges[mel]);
            const auto upper = (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]);
            filters[mel * FREQUENCY_BINS + frequency] =
                static_cast<float>(std::max(0.0, std::min(lower, upper)) * 2.0 / (edges[mel + 2] - edges[mel]));
        }
    return filters;
}

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

auto WhisperFeatureExtractor::load(const std::filesystem::path& path) -> Result<WhisperFeatureExtractor> {
    try {
        if (!std::filesystem::is_regular_file(path))
            return std::unexpected(Error{ErrorCode::IO, "preprocessor config does not exist: " + path.string()});
        const auto config = YAML::LoadFile(path.string());
        const auto bins = config["feature_size"].as<std::size_t>();
        if (config["feature_extractor_type"].as<std::string>() != "WhisperFeatureExtractor" ||
            config["sampling_rate"].as<std::size_t>() != SAMPLE_RATE || (bins != 80 && bins != 128) ||
            config["hop_length"].as<std::size_t>() != HOP_LENGTH || config["n_fft"].as<std::size_t>() != FFT_SIZE ||
            config["n_samples"].as<std::size_t>() != SAMPLE_COUNT ||
            config["nb_max_frames"].as<std::size_t>() != FRAME_COUNT)
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported Whisper preprocessing configuration"});
        if (!config["mel_filters"]) return WhisperFeatureExtractor(mel_filters(bins));
        if (config["mel_filters"].size() != bins)
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid Whisper mel filter shape"});
        std::vector<float> filters;
        filters.reserve(bins * FREQUENCY_BINS);
        for (const auto& row : config["mel_filters"]) {
            if (row.size() != FREQUENCY_BINS)
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid Whisper mel filter shape"});
            for (const auto& value : row) {
                const auto filter = value.as<float>();
                if (!std::isfinite(filter) || filter < 0.F)
                    return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid Whisper mel filter value"});
                filters.push_back(filter);
            }
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
    const auto mel_bins = bins();
    WhisperFeatures result{{}, mel_bins, FRAME_COUNT};
    result.values.assign(mel_bins * FRAME_COUNT, -10.F);
    std::array<float, FFT_SIZE * 2> fft_input{};
    std::array<float, FFT_SIZE * 8> fft_output{};
    std::array<float, FREQUENCY_BINS> power{};
    const auto active_frames = std::min(FRAME_COUNT, (count + FFT_SIZE / 2 + HOP_LENGTH - 1) / HOP_LENGTH);
    for (std::size_t frame = 0; frame < active_frames; ++frame) {
        for (std::size_t index = 0; index < FFT_SIZE; ++index) {
            auto source =
                static_cast<std::int64_t>(frame * HOP_LENGTH + index) - static_cast<std::int64_t>(FFT_SIZE / 2);
            if (source < 0) source = -source;
            if (source >= static_cast<std::int64_t>(SAMPLE_COUNT)) source = 2 * SAMPLE_COUNT - 2 - source;
            fft_input[index] =
                FFT_CACHE.hann[index] * (static_cast<std::size_t>(source) < count ? waveform[source] : 0.F);
        }
        fft(fft_input.data(), FFT_SIZE, fft_output.data());
        for (std::size_t frequency = 0; frequency < FREQUENCY_BINS; ++frequency) {
            const auto real = fft_output[2 * frequency], imaginary = fft_output[2 * frequency + 1];
            power[frequency] = real * real + imaginary * imaginary;
        }
        for (std::size_t mel = 0; mel < mel_bins; ++mel) {
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