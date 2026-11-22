#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "kidi/audio/whisper.h"

namespace {
auto create_fixture_directory() -> std::filesystem::path {
    const auto root = std::filesystem::temp_directory_path();
    std::random_device random;
    for (int attempt = 0; attempt < 16; ++attempt) {
        auto candidate = root / ("kidi-whisper-audio-test-" + std::to_string(random()));
        if (std::filesystem::create_directory(candidate)) return candidate;
    }
    throw std::runtime_error("cannot create a unique Whisper test directory");
}

struct PreprocessorFixture {
    const std::filesystem::path directory = create_fixture_directory();
    const std::filesystem::path config = directory / "preprocessor_config.json";

    ~PreprocessorFixture() {
        std::error_code error;
        std::filesystem::remove(config, error);
        if (!error) std::filesystem::remove(directory, error);
        if (error) std::cerr << "failed to clean up Whisper test fixture: " << error.message() << '\n';
    }
};

auto write_config(const std::filesystem::path& path, const YAML::Node& config) -> void {
    std::ofstream output(path);
    output << YAML::Dump(config);
    output.close();
    if (!output) throw std::runtime_error("failed to write Whisper preprocessor config: " + path.string());
}

auto write_preprocessor(const std::filesystem::path& path, int bins, bool embedded) -> void {
    YAML::Node config;
    config["feature_extractor_type"] = "WhisperFeatureExtractor";
    config["sampling_rate"] = 16000;
    config["feature_size"] = bins;
    config["hop_length"] = 160;
    config["n_fft"] = 400;
    config["n_samples"] = 480000;
    config["nb_max_frames"] = 3000;
    if (embedded)
        for (int mel = 0; mel < bins; ++mel)
            for (int frequency = 0; frequency < 201; ++frequency)
                config["mel_filters"][mel].push_back(frequency == mel ? 1.F : 0.F);
    write_config(path, config);
}
} // namespace

auto main() -> int try {
    const PreprocessorFixture fixture;
    const auto& config = fixture.config;
    const std::vector<float> silence(400);
    for (const std::size_t bins : {80, 128}) {
        for (const bool embedded : {true, false}) {
            write_preprocessor(config, static_cast<int>(bins), embedded);
            auto extractor = kidi::audio::WhisperFeatureExtractor::load(config);
            if (!extractor) {
                std::cerr << "failed to load Whisper preprocessor: " << extractor.error().message << '\n';
                return 1;
            }
            if (extractor->bins() != bins) {
                std::cerr << "expected " << bins << " Whisper mel bins, got " << extractor->bins() << '\n';
                return 1;
            }
            auto features = extractor->extract(silence, 16000);
            if (!features || features->bins != bins || features->frames != 3000 ||
                features->values.size() != bins * 3000)
                return 1;
            for (auto value : features->values)
                if (value != -1.5F) return 1;
            for (const std::size_t length : {1, 159, 160, 201, 401}) {
                std::vector<float> signal(length);
                for (std::size_t index = 0; index < length; ++index)
                    signal[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 16.F;
                const auto short_features = extractor->extract(signal, 16000);
                signal.resize(480000, 0.F);
                const auto padded_features = extractor->extract(signal, 16000);
                if (!short_features || !padded_features || short_features->values != padded_features->values) {
                    std::cerr << "short audio features differ from explicit 30-second zero padding\n";
                    return 1;
                }
            }
            if (extractor->extract(silence, 8000) || extractor->extract({}, 16000) ||
                extractor->extract(std::vector<float>(480001), 16000))
                return 1;
        }
    }
    write_preprocessor(config, 64, false);
    if (kidi::audio::WhisperFeatureExtractor::load(config)) return 1;
    write_preprocessor(config, 128, true);
    auto invalid = YAML::LoadFile(config.string());
    invalid["mel_filters"][0] = YAML::Node(YAML::NodeType::Sequence);
    write_config(config, invalid);
    if (kidi::audio::WhisperFeatureExtractor::load(config)) {
        std::cerr << "invalid mel filter shape was accepted\n";
        return 1;
    }
    return 0;
} catch (const std::runtime_error& error) {
    std::cerr << error.what() << '\n';
    return 1;
}