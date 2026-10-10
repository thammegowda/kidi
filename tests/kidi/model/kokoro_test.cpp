#include "kidi/model/kokoro.h"

#include <filesystem>
#include <iostream>

auto main() -> int {
    const auto path = std::filesystem::temp_directory_path() / "kidi-kokoro-invalid.safetensors";
    auto voice = kidi::tensor::Tensor::zeros({510, 256}, kidi::tensor::DType::I8);
    auto malformed_scales = kidi::tensor::Tensor::zeros({510, 1}, kidi::tensor::DType::F32);
    if (!voice || !malformed_scales ||
        !kidi::checkpoint::Weights::save(
            path, {{"voices.af_heart.weight", *voice}, {"voices.af_heart.scale", *malformed_scales}})) {
        std::cerr << "cannot create malformed Kokoro fixture\n";
        return 1;
    }
    auto weights = kidi::checkpoint::Weights::load(path);
    std::filesystem::remove(path);
    if (!weights) {
        std::cerr << weights.error().message << '\n';
        return 1;
    }
    YAML::Node config;
    config["type"] = "kokoro";
    config["sample_rate"] = 24000;
    config["weight_group_size"] = 8;
    config["voices"].push_back("af_heart");
    config["config"]["vocab"]["a"] = 1;
    auto model = kidi::model::KokoroImpl::load(config, std::move(*weights));
    if (model || model.error().code != kidi::ErrorCode::INVALID_MANIFEST) {
        std::cerr << "malformed Kokoro voice scales were accepted\n";
        return 1;
    }
    return 0;
}
