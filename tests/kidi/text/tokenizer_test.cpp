#include <array>
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

auto write_gzip(const std::filesystem::path& path, std::string_view contents) -> bool {
    gzFile output = gzopen(path.c_str(), "wb");
    if (output == nullptr) {
        return false;
    }
    const auto written = gzwrite(output, contents.data(), static_cast<unsigned int>(contents.size()));
    return written == static_cast<int>(contents.size()) && gzclose(output) == Z_OK;
}

} // namespace

auto main(int argc, char** argv) -> int {
    const std::array conversation{
        kidi::text::ChatMessage{"system", "Be brief."}, kidi::text::ChatMessage{"user", "Hi\nthere"},
        kidi::text::ChatMessage{"assistant", "Hello"}, kidi::text::ChatMessage{"user", "Again"}};
    if (argc == 2) {
        auto checkpoint = kidi::text::Tokenizer::load(std::filesystem::path(argv[1]) / "tokenizer.json");
        if (!checkpoint) return 1;
        const std::array messages{kidi::text::ChatMessage{"user", "Hello"}};
        auto rendered = checkpoint->format_chat(messages);
        const std::string expected = "<bos><|turn>user\nHello<turn|>\n<|turn>model\n";
        if (!rendered || *rendered != expected) {
            std::cerr << "checkpoint chat mismatch: " << (rendered ? *rendered : rendered.error().message) << '\n';
            return 1;
        }
        auto multi_turn = checkpoint->format_chat(conversation);
        if (!multi_turn || *multi_turn !=
                               "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nHi\nthere<turn|>\n"
                               "<|turn>model\nHello<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n")
            return 1;
    }
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

    if (plain->format_chat(conversation)) return 1;
    std::ofstream(directory / "chat_template.jinja")
        << "{{ bos_token }}{% for message in messages %}{{ message.role }}:{{ message.content }};{% endfor %}"
           "{% if add_generation_prompt %}assistant:{% endif %}{% if enable_thinking %}thinking{% endif %}";
    if (kidi::text::Tokenizer::load(plain_path)) return 1;
    std::ofstream(directory / "tokenizer_config.json") << R"({"bos_token":"<bos>"})";
    auto chat = kidi::text::Tokenizer::load(plain_path);
    if (!chat) return 1;
    auto formatted = chat->format_chat(conversation);
    if (!formatted || *formatted != "<bos>system:Be brief.;user:Hi\nthere;assistant:Hello;user:Again;assistant:")
        return 1;
    if (chat->format_chat({}) || chat->format_chat(std::span(conversation).first(3))) return 1;
    const std::array invalid{kidi::text::ChatMessage{"tool", "ignored"}, kidi::text::ChatMessage{"user", "Hi"}};
    if (chat->format_chat(invalid)) return 1;

    std::filesystem::remove_all(directory);
    return 0;
}