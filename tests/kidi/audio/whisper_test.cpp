#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <yaml-cpp/yaml.h>

#include "kidi/audio/whisper.h"

namespace {
auto write_preprocessor(const std::filesystem::path& path) -> void {
    YAML::Node config;
    config["feature_extractor_type"] = "WhisperFeatureExtractor";
    config["sampling_rate"] = 16000;
    config["feature_size"] = 80;
    config["hop_length"] = 160;
    config["n_fft"] = 400;
    config["n_samples"] = 480000;
    config["nb_max_frames"] = 3000;
    for (int mel = 0; mel < 80; ++mel)
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
    write_preprocessor(config);
    auto extractor = kidi::audio::WhisperFeatureExtractor::load(config);
    if (!extractor) {
        std::cerr << extractor.error().message << '\n';
        return 1;
    }
    const std::vector<float> silence(400);
    auto features = extractor->extract(silence, 16000);
    if (!features || features->bins != 80 || features->frames != 3000 || features->values.size() != 240000) return 1;
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
    std::filesystem::remove_all(directory);
    return 0;
}