#include "kidi/inference/synthesizer.h"

#include <cctype>
#include <cmath>
#include <locale>
#include <sstream>

#include "kidi/checkpoint/config.h"
#include "kidi/model/kokoro.h"
#include "kidi/model/omnivoice.h"
#include "kidi/model/registry.h"
#include "kidi/ops/context.h"

namespace kidi::inference {
using ops::require;
namespace {

auto trim(std::string_view value) -> std::string_view {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.remove_suffix(1);
    return value;
}

auto normalize_key(std::string_view value) -> Result<std::string> {
    value = trim(value);
    if (value.empty())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "voice attribute key must not be empty"});
    std::string result;
    result.reserve(value.size());
    for (const auto character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (std::isalnum(byte))
            result.push_back(static_cast<char>(std::tolower(byte)));
        else if (character == '-' || character == '_')
            result.push_back('-');
        else
            return std::unexpected(
                Error{ErrorCode::INVALID_ARGUMENT, "voice attribute keys must use letters, numbers, '-' or '_'"});
    }
    return result;
}

} // namespace

auto parse_synthesis_number(std::string_view value) -> Result<float> {
    if (value.empty()) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "expected a finite number"});
    std::istringstream input{std::string(value)};
    input.imbue(std::locale::classic());
    float result = 0;
    input >> std::noskipws >> result;
    if (!input || !input.eof() || !std::isfinite(result))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "expected a finite number"});
    return result;
}

auto parse_voice_attributes(std::span<const std::string> specifications) -> Result<VoiceAttributes> {
    VoiceAttributes result;
    for (const auto& specification : specifications) {
        const auto assignment = trim(specification);
        const auto separator = assignment.find('=');
        if (separator == std::string_view::npos || separator == 0 || separator + 1 == assignment.size())
            return std::unexpected(
                Error{ErrorCode::INVALID_ARGUMENT, "voice attributes must use nonempty key=value assignments"});
        auto key = normalize_key(assignment.substr(0, separator));
        if (!key) return std::unexpected(key.error());
        const auto value = trim(assignment.substr(separator + 1));
        if (value.empty())
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "voice attribute value must not be empty"});
        if (!result.emplace(*key, value).second)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "duplicate voice attribute: " + *key});
    }
    return result;
}

auto synthesis_elapsed_ns(std::chrono::steady_clock::time_point start) -> std::uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}

struct Synthesizer::Impl {
    std::shared_ptr<SynthesizerModel> model;
};

Synthesizer::Synthesizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Synthesizer::Synthesizer(Synthesizer&&) noexcept = default;
auto Synthesizer::operator=(Synthesizer&&) noexcept -> Synthesizer& = default;
Synthesizer::~Synthesizer() = default;

auto Synthesizer::load(const std::filesystem::path& directory, tensor::Device device) -> Result<Synthesizer> {
    try {
        auto config = require(checkpoint::load_config(directory / "model.yaml"));
        const auto type = config["model"]["type"].as<std::string>();
        if (!model::supports_task(type, model::Task::TTS))
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, type + " does not support the TTS task"});
        auto impl = std::make_unique<Impl>();
        if (type == "omnivoice")
            impl->model = require(model::OmniVoiceImpl::load(std::move(config), device)).ptr();
        else if (type == "kokoro")
            impl->model = require(model::KokoroImpl::load(std::move(config), device)).ptr();
        else
            throw ops::Failure({ErrorCode::UNSUPPORTED, "no TTS loader for model type: " + type});
        return Synthesizer(std::move(impl));
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto Synthesizer::synthesize(std::string_view text, const SynthesisOptions& options) const -> Result<Synthesis> {
    return impl_->model->synthesize(text, options);
}

auto Synthesizer::execution() const -> std::string { return impl_->model->execution(); }

} // namespace kidi::inference
