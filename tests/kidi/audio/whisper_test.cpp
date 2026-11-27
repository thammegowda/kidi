#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>
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
        if (error) ADD_FAILURE() << "failed to clean up Whisper test fixture: " << error.message();
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

TEST(WhisperMelTest, ExtractsFeatures) {
    const PreprocessorFixture fixture;
    const auto& config = fixture.config;
    const std::vector<float> silence(400);
    for (const std::size_t bins : {80, 128}) {
        for (const bool embedded : {true, false}) {
            SCOPED_TRACE(::testing::Message() << "bins=" << bins << " embedded=" << embedded);
            write_preprocessor(config, static_cast<int>(bins), embedded);
            auto extractor = kidi::audio::WhisperFeatureExtractor::load(config);
            ASSERT_TRUE(extractor.has_value())
                << "failed to load Whisper preprocessor: " << extractor.error().message;
            ASSERT_EQ(extractor->bins(), bins) << "expected " << bins << " Whisper mel bins";
            auto features = extractor->extract(silence, 16000);
            ASSERT_TRUE(features);
            EXPECT_EQ(features->bins, bins);
            EXPECT_EQ(features->frames, 3000U);
            ASSERT_EQ(features->values.size(), bins * 3000);
            for (auto value : features->values) ASSERT_EQ(value, -1.5F);
            // Padding is independent of the filter bank; check it once to keep this test fast.
            const bool check_padding = bins == 80 && embedded;
            for (const std::size_t length : {1, 159, 160, 201, 401}) {
                if (!check_padding) break;
                std::vector<float> signal(length);
                for (std::size_t index = 0; index < length; ++index)
                    signal[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 16.F;
                const auto short_features = extractor->extract(signal, 16000);
                signal.resize(480000, 0.F);
                const auto padded_features = extractor->extract(signal, 16000);
                ASSERT_TRUE(short_features) << "length=" << length;
                ASSERT_TRUE(padded_features) << "length=" << length;
                EXPECT_EQ(short_features->values, padded_features->values)
                    << "short audio features differ from explicit 30-second zero padding (length=" << length << ")";
            }
            EXPECT_FALSE(extractor->extract(silence, 8000));
            EXPECT_FALSE(extractor->extract({}, 16000));
            EXPECT_FALSE(extractor->extract(std::vector<float>(480001), 16000));
        }
    }
}

TEST(WhisperMelTest, RejectsInvalid) {
    const PreprocessorFixture fixture;
    const auto& config = fixture.config;
    write_preprocessor(config, 64, false);
    EXPECT_FALSE(kidi::audio::WhisperFeatureExtractor::load(config));
    write_preprocessor(config, 128, true);
    auto invalid = YAML::LoadFile(config.string());
    invalid["mel_filters"][0] = YAML::Node(YAML::NodeType::Sequence);
    write_config(config, invalid);
    EXPECT_FALSE(kidi::audio::WhisperFeatureExtractor::load(config)) << "invalid mel filter shape was accepted";
}

} // namespace
