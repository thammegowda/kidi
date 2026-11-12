#include "kidi/model/gemma4.h"
#include "kidi/inference/decoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace {
using namespace kidi;
using ops::require;
auto snapshot(const tensor::Tensor& tensor) -> tensor::Tensor {
    return require(tensor::Tensor::from_bytes({tensor.shape().begin(), tensor.shape().end()}, tensor.dtype(),
                                              require(tensor.copy_to_host())));
}
auto close(const tensor::Tensor& left, const tensor::Tensor& right) -> bool {
    if (!std::ranges::equal(left.shape(), right.shape())) return false;
    const auto a = require(left.data<float>()), b = require(right.data<float>());
    for (std::size_t index = 0; index < a.size(); ++index)
        if (!std::isfinite(a[index]) || !std::isfinite(b[index]) || std::abs(a[index] - b[index]) > 2e-4F) return false;
    return true;
}
auto check_rounds(model::Gemma4Impl& model, std::span<const std::int32_t> prompt, std::size_t vocabulary) -> bool {
    constexpr std::size_t MAXIMUM = 10;
    const inference::SearchOptions options{.vocabulary_size = vocabulary,
                                           .end_id = 1,
                                           .pad_id = 0,
                                           .maximum_steps = MAXIMUM,
                                           .compute_score = false,
                                           .stop_on_eos = false};
    auto reference_state = require(model.create_state(16));
    auto reference = require(inference::GreedyState::create(options));
    require(
        reference.accept({{}, inference::ScoreKind::LOGITS, require(model.forward_token(prompt, reference_state))}));
    while (!reference.finished()) {
        const auto token = reference.token();
        require(reference.accept(
            {{}, inference::ScoreKind::LOGITS, require(model.forward_token(std::span(&token, 1), reference_state))}));
    }
    const auto& expected = reference.result().token_ids;
    for (int mode = 0; mode < 3; ++mode) {
        auto state = require(model.create_state(16));
        auto search = require(inference::GreedyState::create(options));
        const auto first = require(model.forward_features(prompt, state, false));
        require(search.accept({require(first.logits.data<float>())}));
        while (!search.finished()) {
            const auto generated = search.result().decoder_steps;
            const auto count = std::min<std::size_t>(3, MAXIMUM - generated - 1);
            std::vector<std::int32_t> drafts;
            for (std::size_t index = 0; index < count; ++index) {
                const auto token = expected[generated + index];
                drafts.push_back(mode == 0 || (mode == 2 && index == 0)
                                     ? token
                                     : static_cast<std::int32_t>((token + 1) % vocabulary));
            }
            std::vector<std::int32_t> inputs{search.token()};
            inputs.insert(inputs.end(), drafts.begin(), drafts.end());
            const auto position = state.position;
            const auto verification = require(model.forward_features(inputs, state));
            const auto accepted =
                require(search.accept_speculative(drafts, {require(verification.logits.data<float>())}));
            const auto matching = mode == 0 ? count : mode == 1 ? 0 : std::min<std::size_t>(1, count);
            if (accepted.accepted_drafts != matching || !accepted.committed_inputs ||
                accepted.committed_inputs > inputs.size())
                return false;
            require(model.rewind(state, position + accepted.committed_inputs));
        }
        if (search.result().token_ids != expected) return false;
    }
    return true;
}
auto check(const std::filesystem::path& directory) -> bool {
    const auto config = YAML::LoadFile((directory / "model.yaml").string())["model"];
    const auto checkpoint = require(checkpoint::Weights::load(directory / "model.safetensors"));
    const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
    auto model = require(model::Gemma4Impl::create(config));
    require(model->set_checkpoint(checkpoint));
    const std::array<std::int32_t, 3> prompt{2, 5, 7};
    auto prefix = require(model->create_state(16));
    for (const auto& layer : prefix.layers)
        if (layer.key.dtype() == tensor::DType::I8 &&
            (layer.key_quantization.scales.size() != 1 || layer.value_quantization.scales.size() != 1 ||
             layer.key_quantization.scales[0] <= 0 || layer.value_quantization.scales[0] <= 0))
            return false;
    const auto features = require(model->forward_features(prompt, prefix));
    const auto feature_logits = snapshot(features.logits);
    if (features.hidden.dimensions() != 3 || features.hidden.size(1) != prompt.size() ||
        features.hidden.size(2) != config["hidden_size"].as<int>() || features.logits.size(1) != prompt.size() ||
        prefix.position != prompt.size())
        return false;
    auto oracle = require(model->create_state(16));
    if (!close(feature_logits, require(model->forward(prompt, oracle, true)))) return false;
    const std::array<std::int32_t, 4> proposed{3, 8, 6, 9};
    const std::array<std::int32_t, 1> correction{11};
    for (std::size_t committed = 1; committed <= proposed.size(); ++committed) {
        auto speculative = require(model->fork_state(prefix, prompt.size(), 16));
        auto serial = require(model->fork_state(prefix, prompt.size(), 16));
        const auto verification = require(model->forward_features(proposed, speculative));
        if (verification.hidden.size(1) != proposed.size() || verification.logits.size(1) != proposed.size() ||
            speculative.position != prompt.size() + proposed.size())
            return false;
        require(model->rewind(speculative, prompt.size() + committed));
        for (const auto token : std::span(proposed).first(committed))
            require(model->forward(std::span(&token, 1), serial));
        const auto actual = snapshot(require(model->forward(correction, speculative)));
        const auto expected = require(model->forward(correction, serial));
        if (!close(actual, expected) || speculative.position != serial.position) return false;
    }
    const auto position = prefix.position;
    if (model->rewind(prefix, position + 1) || prefix.position != position) return false;
    require(model->rewind(prefix, 0));
    const auto replayed = require(model->forward_features(prompt, prefix, false));
    const auto replayed_logits = snapshot(replayed.logits);
    auto fresh = require(model->create_state(16));
    const auto expected = require(model->forward(prompt, fresh));
    if (replayed.hidden.size(1) != 1 || !close(replayed_logits, expected)) return false;
    const auto one = require(model->forward_features(correction, prefix));
    if (!one.hidden.defined() || one.hidden.size(1) != 1 || one.logits.size(1) != 1) return false;
    model::Gemma4State invalid;
    return !model->rewind(invalid, 0) && !model->forward_features({}, prefix) &&
           check_rounds(*model, prompt, config["vocab_size"].as<std::size_t>());
}
} // namespace

auto main() -> int {
    try {
        const auto directory = std::filesystem::path(KIDI_GEMMA4_FIXTURE);
        if (!check(directory) || !check(directory.parent_path() / "gemma4-qat")) {
            std::cerr << "Gemma 4 MTP features or cache rewind failed\n";
            return 1;
        }
        std::cout << "Gemma 4 MTP features and cache rewind passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
