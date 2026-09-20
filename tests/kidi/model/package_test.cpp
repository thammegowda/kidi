#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

#include "kidi/model/package.h"
#include "kidi/model/transformer.h"
#include "kidi/inference/translator.h"

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
weights_file: model.safetensors
tokenizers: {source: tokenizer.src.json, target: tokenizer.tgt.json}
io: {input_format: moses_tokenized, output_format: moses_tokenized}
model:
  type: rtg_transformer_nmt
  source_tokens: 8
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
decode:
  beam_size: 2
  maximum_extra_tokens: 4
  length_penalty: 0.6
)";

auto write(const std::filesystem::path& path, std::string_view contents = {}) -> void {
    std::ofstream(path, std::ios::binary) << contents;
}

auto write_empty_safetensors(const std::filesystem::path& path) -> void {
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

auto main() -> int {
    const auto directory = std::filesystem::temp_directory_path() / "kidi-rtg-package-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    write_empty_safetensors(directory / "model.safetensors");
    write(directory / "tokenizer.src.json", TOKENIZER_JSON);
    write(directory / "tokenizer.tgt.json", TOKENIZER_JSON);
    write(directory / "model.yaml", MANIFEST_YAML);

    auto package = kidi::model::Package::load(directory);
    if (!package || package->source_tokenizer().token_id("<s>") != 2 ||
        package->target_tokenizer().vocabulary_size() != 5 ||
        package->config()["model"]["source_pad_id"].as<int>() != 0 ||
        package->config()["decode"]["begin_id"].as<int>() != 2 ||
        package->config()["decode"]["end_id"].as<int>() != 3) {
        std::cerr << "valid RTG package was rejected\n";
        return 1;
    }
    if (kidi::model::Package::load(directory / "model.yaml")) {
        std::cerr << "manifest path was accepted as a model directory\n";
        return 1;
    }

    auto model_config = YAML::Clone(package->config()["model"]);
    if (!kidi::model::TransformerImpl::validate_config(model_config)) return 1;
    auto scratch = kidi::model::TransformerImpl::create(model_config);
    if (!scratch) return 1;
    const auto initialized = (*scratch)->state_dict();
    if (!initialized.contains("encoder.0.qkv.weight") || !initialized.at("encoder.0.qkv.weight").defined()) return 1;
    const auto target = initialized.at("target_embedding.weight").host_bytes();
    const auto projection = initialized.at("generator.weight").host_bytes();
    if (!target || !projection || target->data() != projection->data()) return 1;
    for (const auto* key : {"type", "hidden_size", "source_pad_id"}) {
        auto invalid = YAML::Clone(model_config);
        invalid.remove(key);
        auto result = kidi::model::TransformerImpl::create(invalid);
        if (result || result.error().code != kidi::ErrorCode::INVALID_MANIFEST) return 1;
    }
    auto invalid = YAML::Clone(model_config);
    invalid["attention_heads"] = 3;
    if (kidi::model::TransformerImpl::validate_config(invalid)) return 1;
    invalid = YAML::Clone(model_config);
    invalid["hidden_size"] = 0;
    if (kidi::model::TransformerImpl::validate_config(invalid)) return 1;
    invalid = YAML::Clone(model_config);
    invalid["source_pad_id"] = -1;
    if (kidi::model::TransformerImpl::validate_config(invalid)) return 1;
    invalid = YAML::Clone(model_config);
    invalid["layer_norm_epsilon"] = std::numeric_limits<float>::quiet_NaN();
    if (kidi::model::TransformerImpl::validate_config(invalid)) return 1;

    auto invalid_decode = YAML::Load(std::string(MANIFEST_YAML));
    invalid_decode["decode"]["beam_size"] = 0;
    write(directory / "model.yaml", YAML::Dump(invalid_decode));
    auto translator = kidi::inference::Translator::load(directory);
    if (translator || translator.error().code != kidi::ErrorCode::INVALID_ARGUMENT) return 1;
    invalid_decode["decode"]["beam_size"] = 1;
    invalid_decode["decode"]["length_penalty"] = std::numeric_limits<float>::quiet_NaN();
    write(directory / "model.yaml", YAML::Dump(invalid_decode));
    translator = kidi::inference::Translator::load(directory);
    if (translator || translator.error().code != kidi::ErrorCode::INVALID_ARGUMENT) return 1;

    write(directory / "model.yaml", MANIFEST_YAML);
    const std::string original(TOKENIZER_JSON);
    auto reordered = original;
    const std::string vocabulary = R"("<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "hi": 4)";
    reordered.replace(reordered.find(vocabulary), vocabulary.size(),
                      R"("<pad>": 4, "<unk>": 0, "<s>": 1, "</s>": 2, "hi": 3)");
    write(directory / "tokenizer.tgt.json", reordered);
    auto different_ids = kidi::model::Package::load(directory);
    if (!different_ids || different_ids->config()["model"]["source_pad_id"].as<int>() != 0 ||
        different_ids->config()["decode"]["source_end_id"].as<int>() != 3 ||
        different_ids->config()["decode"]["begin_id"].as<int>() != 1 ||
        different_ids->config()["decode"]["end_id"].as<int>() != 2 ||
        different_ids->config()["decode"]["pad_id"].as<int>() != 4)
        return 1;
    auto missing = original;
    missing.replace(missing.find("<s>"), 3, "<missing>");
    write(directory / "tokenizer.tgt.json", missing);
    if (kidi::model::Package::load(directory)) return 1;

    std::filesystem::remove_all(directory);
    return 0;
}