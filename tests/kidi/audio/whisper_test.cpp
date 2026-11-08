#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <yaml-cpp/yaml.h>

#include "kidi/audio/whisper.h"

namespace {
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
    std::ofstream(path) << YAML::Dump(config);
}
} // namespace

auto main() -> int {
    const auto directory = std::filesystem::temp_directory_path() / "kidi-whisper-audio-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto config = directory / "preprocessor_config.json";
    const std::vector<float> silence(400);
    for (const std::size_t bins : {80, 128}) {
        for (const bool embedded : {true, false}) {
            write_preprocessor(config, static_cast<int>(bins), embedded);
            auto extractor = kidi::audio::WhisperFeatureExtractor::load(config);
            if (!extractor || extractor->bins() != bins) {
                std::cerr << "failed to load Whisper preprocessor\n";
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
    std::ofstream(config) << YAML::Dump(invalid);
    if (kidi::audio::WhisperFeatureExtractor::load(config)) {
        std::cerr << "invalid mel filter shape was accepted\n";
        return 1;
    }
    std::filesystem::remove_all(directory);
    return 0;
}