#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
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

class Translator {
public:
    Translator(Translator&&) noexcept = default;
    Translator& operator=(Translator&&) noexcept = default;

    Translator(const Translator&) = delete;
    Translator& operator=(const Translator&) = delete;

    [[nodiscard]] static std::expected<Translator, core::Error> load(const std::filesystem::path& manifest_path);
    [[nodiscard]] std::expected<Translation, core::Error> translate(std::string_view source);

private:
    Translator(Package package, Encoder encoder, Decoder decoder) noexcept;

    Package package_;
    Encoder encoder_;
    Decoder decoder_;
};

} // namespace kidi::rtg