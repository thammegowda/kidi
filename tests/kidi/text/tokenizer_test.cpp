#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <zlib.h>

#include "kidi/text/tokenizer.h"

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
    "vocab": {"<unk>": 0, "hi": 1, "café": 2}
  }
})";

bool write_gzip(const std::filesystem::path& path, std::string_view contents) {
    gzFile output = gzopen(path.c_str(), "wb");
    if (output == nullptr) {
        return false;
    }
    const auto written = gzwrite(output, contents.data(), static_cast<unsigned int>(contents.size()));
    return written == static_cast<int>(contents.size()) && gzclose(output) == Z_OK;
}

} // namespace

int main() {
    const auto directory = std::filesystem::temp_directory_path() / "kidi-tokenizer-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto plain_path = directory / "tokenizer.json";
    const auto gzip_path = directory / "tokenizer.json.gz";
    std::ofstream(plain_path, std::ios::binary) << TOKENIZER_JSON;
    if (!write_gzip(gzip_path, TOKENIZER_JSON)) {
        std::cerr << "failed to create gzip fixture\n";
        return 1;
    }

    auto plain = kidi::text::Tokenizer::load(plain_path);
    auto compressed = kidi::text::Tokenizer::load(gzip_path);
    if (!plain || !compressed || plain->vocabulary_size() != 3 || compressed->vocabulary_size() != 3) {
        std::cerr << "failed to load tokenizer fixtures\n";
        return 1;
    }

    auto plain_ids = plain->encode("café");
    auto compressed_ids = compressed->encode("café");
    if (!plain_ids || !compressed_ids || *plain_ids != *compressed_ids || *plain_ids != std::vector<std::int32_t>{2}) {
        std::cerr << "plain and compressed tokenizers disagree\n";
        return 1;
    }
    auto decoded = compressed->decode(*compressed_ids);
    if (!decoded || *decoded != "café") {
        std::cerr << "UTF-8 tokenizer round trip failed\n";
        return 1;
    }

    std::filesystem::remove_all(directory);
    return 0;
}