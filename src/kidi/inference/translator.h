#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/model/transformer.h"
#include "kidi/model/package.h"
#include "kidi/inference/profile.h"

namespace kidi::inference {

enum class InferenceBackend {
    YNNPACK,
    MPS,
};

constexpr auto to_string(InferenceBackend backend) noexcept -> std::string_view {
    switch (backend) {
        case InferenceBackend::YNNPACK:
            return "ynnpack_cpu";
        case InferenceBackend::MPS:
            return "mps_gpu";
    }
}

struct Translation {
    std::string text;
    std::vector<std::int32_t> token_ids;
    float score;
};

struct DecodeOptions {
    std::optional<std::int32_t> beam_size;
    std::optional<std::int32_t> maximum_extra_tokens;
    std::optional<float> length_penalty;
    bool compute_score = true;
};

class Translator {
public:
    Translator(Translator&&) noexcept = default;
    auto operator=(Translator&&) noexcept -> Translator& = default;

    Translator(const Translator&) = delete;
    auto operator=(const Translator&) -> Translator& = delete;

    static auto load(const std::filesystem::path& model_directory, InferenceStats* stats = nullptr)
        -> Result<Translator>;
    static auto load(const std::filesystem::path& model_directory, InferenceBackend backend,
                     InferenceStats* stats = nullptr, std::size_t batch_size = 1) -> Result<Translator>;
    auto translate(std::string_view source, DecodeOptions options = {}, InferenceStats* stats = nullptr)
        -> Result<Translation>;
    auto translate_batch(std::span<const std::string> sources, DecodeOptions options = {},
                         InferenceStats* stats = nullptr) -> Result<std::vector<Translation>>;

private:
    Translator(model::Package package, model::Transformer model, InferenceBackend backend) noexcept;

    model::Package package_;
    model::Transformer model_;
    InferenceBackend backend_;
    std::size_t batch_size_ = 1;
};

} // namespace kidi::inference