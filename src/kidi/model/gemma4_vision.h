#pragma once

#include "kidi/core/precision.h"
#include "kidi/image/gemma4.h"
#include "kidi/model/model.h"
#include <functional>
#include <yaml-cpp/yaml.h>

namespace kidi::model {
KIDI_MODULE(Gemma4Vision);
class Gemma4VisionImpl : public VisionModel {
public:
    using BlockObserver = std::function<void(std::size_t, const tensor::Tensor&)>;
    using StageObserver = std::function<void(std::size_t, std::string_view, const tensor::Tensor&)>;

    Gemma4VisionImpl(const YAML::Node& config, std::int32_t text_width, bool quantized);
    ~Gemma4VisionImpl() override;
    auto state_mapping_specs() const -> std::vector<checkpoint::StateMappingSpec> override;
    auto output_dim() const noexcept -> std::int32_t override;
    auto set_precision(core::InferencePrecision precision) noexcept -> void override;
    auto release_workspaces() -> void override;
    auto encode_image(const image::Gemma4Image& image) -> Result<tensor::Tensor> override { return forward(image); }
    auto forward(const image::Gemma4Image& image) -> Result<tensor::Tensor>;
    auto forward(const image::Gemma4Image& image, const BlockObserver& observer) -> Result<tensor::Tensor>;
    auto forward(const image::Gemma4Image& image, const BlockObserver& block_observer,
                 const StageObserver& stage_observer, std::size_t stage_layers) -> Result<tensor::Tensor>;

private:
    struct State;
    std::unique_ptr<State> impl_;
};
} // namespace kidi::model