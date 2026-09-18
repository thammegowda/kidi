#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

#include "kidi/rtg/package.h"

namespace {

constexpr std::string_view TOKENIZER_JSON = R"({
  "version": "1.0",
  "decoder": {"type": "Fuse"},
  "model": {
    "type": "WordPiece",
    "unk_token": "<unk>",
    "continuing_subword_prefix": "",
    "max_input_chars_per_word": 100,
    "fuse_unk": false,
    "vocab": {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "hi": 4}
  }
})";

constexpr std::string_view MANIFEST_YAML = R"(format_version: 1
model_type: rtg_transformer_nmt
weights_file: model.safetensors
tokenizers: {source: tokenizer.src.json, target: tokenizer.tgt.json}
io: {input_format: moses_tokenized, output_format: moses_tokenized}
special_tokens: {pad: 0, unknown: 1, begin: 2, end: 3}
architecture:
  encoder_layers: 1
  decoder_layers: 1
  hidden_size: 4
  feed_forward_size: 8
  attention_heads: 2
  source_vocabulary_size: 5
  target_vocabulary_size: 5
  activation: gelu
  attention_bias: true
  tied_embeddings: one-way
  layer_norm_epsilon: 1.0e-5
  position_encoding: sinusoidal
  maximum_position: 32
limits: {source_tokens: 8, maximum_extra_tokens: 4, maximum_beam_size: 2}
decode: {beam_size: 2, maximum_extra_tokens: 4, length_penalty: 0.6}
)";

void write(const std::filesystem::path& path, std::string_view contents = {}) {
    std::ofstream(path, std::ios::binary) << contents;
}

void write_empty_safetensors(const std::filesystem::path& path) {
    constexpr std::string_view HEADER = "{}      ";
    std::uint64_t header_size = HEADER.size();
    if constexpr (std::endian::native == std::endian::big) {
        header_size = std::byteswap(header_size);
    }
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(HEADER.data(), static_cast<std::streamsize>(HEADER.size()));
}

} // namespace

int main() {
    const auto directory = std::filesystem::temp_directory_path() / "kidi-rtg-package-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    write_empty_safetensors(directory / "model.safetensors");
    write(directory / "tokenizer.src.json", TOKENIZER_JSON);
    write(directory / "tokenizer.tgt.json", TOKENIZER_JSON);
    write(directory / "model.yaml", MANIFEST_YAML);

    auto package = kidi::rtg::Package::load(directory / "model.yaml");
    if (!package || package->source_tokenizer().token_id("<s>") != 2 ||
        package->target_tokenizer().vocabulary_size() != 5) {
        std::cerr << "valid RTG package was rejected\n";
        return 1;
    }

    std::filesystem::remove_all(directory);
    return 0;
}