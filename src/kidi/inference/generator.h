#pragma once

#include "kidi/inference/decoder.h"
#include "kidi/model/gemma.h"
#include "kidi/text/tokenizer.h"

namespace kidi::inference {
struct GenerationOptions {
    std::size_t maximum_new_tokens = 0, context_size = 0, prefill_chunk_size = 128;
    bool raw_prompt = false, ignore_eos = false;
};
struct GenerationStats {
    std::uint64_t tokenize_ns = 0, prefill_ns = 0, decode_ns = 0, preparation_ns = 0, time_to_first_token_ns = 0;
    std::size_t prompt_tokens = 0, decode_tokens = 0;
};
struct TextGeneration {
    std::string text;
    Generation generation;
    GenerationStats stats;
};
class Generator {
public:
    static auto load(const std::filesystem::path& directory, tensor::Device device) -> Result<Generator>;
    auto generate(std::string_view prompt, GenerationOptions options = {}) -> Result<TextGeneration>;

private:
    Generator(YAML::Node config, text::Tokenizer tokenizer, model::Gemma4 model, std::array<std::int32_t, 3> special);
    YAML::Node config_;
    text::Tokenizer tokenizer_;
    model::Gemma4 model_;
    std::array<std::int32_t, 3> special_;
};
} // namespace kidi::inference