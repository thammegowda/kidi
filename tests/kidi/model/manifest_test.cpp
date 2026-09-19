#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kidi/model/manifest.h"

namespace {

void write_file(const std::filesystem::path& path, std::string_view content = {}) {
    std::ofstream output(path, std::ios::binary);
    output << content;
}

std::string valid_manifest(std::string_view weights_file = "model.safetensors") {
    return "format_version: 1\n"
           "model_type: rtg_transformer_nmt\n"
           "weights_file: " +
           std::string(weights_file) +
           "\n"
           "tokenizers:\n"
           "  source: tokenizer.src.json\n"
           "  target: tokenizer.tgt.json.gz\n"
           "io:\n"
           "  input_format: moses_tokenized\n"
           "  output_format: moses_tokenized\n"
           "special_tokens:\n"
           "  pad: 0\n"
           "  unknown: 1\n"
           "  begin: 2\n"
           "  end: 3\n"
           "architecture:\n"
           "  encoder_layers: 9\n"
           "  decoder_layers: 6\n"
           "  hidden_size: 768\n"
           "  feed_forward_size: 2048\n"
           "  attention_heads: 12\n"
           "  source_vocabulary_size: 512000\n"
           "  target_vocabulary_size: 64000\n"
           "  activation: gelu\n"
           "  attention_bias: true\n"
           "  tied_embeddings: one-way\n"
           "  layer_norm_epsilon: 1.0e-5\n"
           "  position_encoding: sinusoidal\n"
           "  maximum_position: 5000\n"
           "limits:\n"
           "  source_tokens: 160\n"
           "  maximum_extra_tokens: 50\n"
           "  maximum_beam_size: 4\n"
           "decode:\n"
           "  beam_size: 4\n"
           "  maximum_extra_tokens: 50\n"
           "  length_penalty: 0.6\n"
           "weights:\n"
           "  format: safetensors\n"
           "  encoding: BF16\n";
}

} // namespace

int main() {
    const auto directory = std::filesystem::temp_directory_path() / "kidi-manifest-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    write_file(directory / "model.safetensors");
    write_file(directory / "tokenizer.src.json", "{}");
    write_file(directory / "tokenizer.tgt.json.gz");

    write_file(directory / "model.yaml", valid_manifest());
    auto manifest = kidi::model::ModelManifest::load(directory / "model.yaml");
    if (!manifest || manifest->architecture.hidden_size != 768 ||
        manifest->weights_file != directory / "model.safetensors" ||
        manifest->weights.encoding != kidi::model::WeightEncoding::BF16) {
        std::cerr << "valid manifest was rejected\n";
        return 1;
    }

    write_file(directory / "invalid.yaml", valid_manifest("../outside.safetensors"));
    auto invalid = kidi::model::ModelManifest::load(directory / "invalid.yaml");
    if (invalid || invalid.error().code != kidi::ErrorCode::INVALID_MANIFEST) {
        std::cerr << "escaping model path was accepted\n";
        return 1;
    }

    std::filesystem::remove_all(directory);
    return 0;
}