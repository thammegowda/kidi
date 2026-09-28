#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <vector>

#include "kidi/audio/whisper.h"
#include "kidi/model/config.h"
#include "kidi/model/whisper.h"

namespace {
using kidi::ops::require;

auto read_f32(const std::filesystem::path& path) -> std::vector<float> {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open reference: " + path.string());
    const auto bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)))
        throw std::runtime_error("invalid reference size: " + path.string());
    std::vector<float> result(static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(result.data()), bytes))
        throw std::runtime_error("cannot read reference: " + path.string());
    return result;
}

auto compare(std::string_view name, std::span<const float> actual, std::span<const float> expected) -> void {
    if (actual.size() != expected.size()) throw std::runtime_error(std::string(name) + " size mismatch");
    double squared = 0, reference = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto difference = actual[index] - expected[index];
        squared += static_cast<double>(difference) * difference;
        reference += static_cast<double>(expected[index]) * expected[index];
    }
    const auto relative_rmse = std::sqrt(squared / reference);
    if (!std::isfinite(relative_rmse) || relative_rmse > 1e-3)
        throw std::runtime_error(std::string(name) + " relative RMSE exceeds tolerance");
}

auto transpose_convolution(std::span<const float> values, std::size_t channels, std::size_t frames)
    -> std::vector<float> {
    if (values.size() != channels * frames) throw std::runtime_error("convolution reference shape mismatch");
    std::vector<float> result(values.size());
    for (std::size_t frame = 0; frame < frames; ++frame)
        for (std::size_t channel = 0; channel < channels; ++channel)
            result[frame * channels + channel] = values[channel * frames + frame];
    return result;
}
} // namespace

auto main(int argc, char** argv) -> int {
    if (argc != 4) {
        std::cerr << "usage: kidi_whisper_quality MODEL WAV REFERENCE_DIR\n";
        return 2;
    }
    try {
        const std::filesystem::path directory(argv[1]), reference(argv[3]);
        auto config = require(kidi::model::load_whisper_config(directory));
        auto extractor =
            require(kidi::audio::WhisperFeatureExtractor::load(config["preprocessor_config_file"].as<std::string>()));
        auto waveform = require(kidi::audio::load_wav(argv[2]));
        auto features = require(extractor.extract(waveform.samples, waveform.sample_rate));
        compare("features", features.values, read_f32(reference / "features.f32"));

        auto weights = require(kidi::model::Weights::load(config["model_file"].as<std::string>()));
        const kidi::ModuleScope construction(kidi::tensor::DType::F32, false, kidi::tensor::Device::cpu());
        auto model = require(kidi::model::WhisperImpl::create(config["model"]));
        require(model->set_checkpoint(weights));
        auto source = require(model->encode(features));
        const auto expected_convolution =
            transpose_convolution(read_f32(reference / "conv2.f32"), config["model"]["d_model"].as<std::size_t>(),
                                  config["model"]["max_source_positions"].as<std::size_t>());
        compare("convolution", require(source.convolution.data<float>()), expected_convolution);
        compare("encoder", require(source.hidden.data<float>()), read_f32(reference / "encoder.f32"));

        auto state = require(model->create_state(32));
        kidi::tensor::Tensor logits;
        for (const std::int32_t token : {50258, 50259, 50359, 50363})
            logits = require(model->forward(source, std::span(&token, 1), state));
        const auto actual_logits = require(logits.data<float>());
        const auto expected_logits = read_f32(reference / "logits.f32");
        compare("logits", actual_logits, expected_logits);
        auto prefixed_state = require(model->create_state(32));
        const std::array<std::int32_t, 3> prefix{50258, 50259, 50359};
        require(model->prefill(source, prefix, prefixed_state));
        const std::int32_t final_prefix = 50363;
        const auto prefixed_logits = require(model->forward(source, std::span(&final_prefix, 1), prefixed_state));
        if (!std::ranges::equal(actual_logits, require(prefixed_logits.data<float>())))
            throw std::runtime_error("Whisper prefix-only decoding changed logits");
        if (std::ranges::max_element(actual_logits) - actual_logits.begin() !=
            std::ranges::max_element(expected_logits) - expected_logits.begin())
            throw std::runtime_error("Whisper top logit differs from reference");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}