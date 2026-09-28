#pragma once

#include "kidi/core/module.h"
#include "kidi/image/gemma4.h"
#include "kidi/model/weights.h"
#include <yaml-cpp/yaml.h>

namespace kidi::model {
KIDI_MODULE(Gemma4Vision);
class Gemma4VisionImpl : public Module {
public:
    Gemma4VisionImpl(const YAML::Node& config, std::int32_t text_width, bool quantized);
    ~Gemma4VisionImpl();
    auto set_checkpoint(const Weights& weights) -> Result<void>;
    auto forward(const image::Gemma4Image& image) -> Result<tensor::Tensor>;

private:
    struct State;
    std::unique_ptr<State> impl_;
};
} // namespace kidi::model