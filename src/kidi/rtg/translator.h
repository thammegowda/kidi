#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/rtg/decoder.h"
#include "kidi/rtg/encoder.h"
#include "kidi/rtg/package.h"

namespace kidi::rtg {

struct Translation {
    std::string text;
    std::vector<std::int32_t> token_ids;
    float score;
};

struct DecodeOptions {
    std::optional<std::int32_t> beam_size;
    std::optional<std::int32_t> maximum_extra_tokens;
    std::optional<float> length_penalty;
};

class Translator {
public:
    Translator(Translator&&) noexcept = default;
    Translator& operator=(Translator&&) noexcept = default;

    Translator(const Translator&) = delete;
    Translator& operator=(const Translator&) = delete;

    [[nodiscard]] static Result<Translator> load(const std::filesystem::path& model_directory);
    [[nodiscard]] Result<Translation> translate(std::string_view source, DecodeOptions options = {});

private:
    Translator(Package package, Encoder encoder, Decoder decoder) noexcept;

    Package package_;
    Encoder encoder_;
    Decoder decoder_;
};

} // namespace kidi::rtg