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
    const auto synthesis_template = directory / "synthesis_template.jinja";
    std::ofstream(synthesis_template) << "<lang>{{- language -}}</lang><style>{{- instruction -}}</style>{{- text -}}";
    auto chat = kidi::text::Tokenizer::load(plain_path);
    if (!chat) return 1;
    auto formatted = chat->format_chat(conversation);
    if (!formatted || *formatted != "<bos>system:Be brief.;user:Hi\nthere;assistant:Hello;user:Again;assistant:")
        return 1;
    if (chat->format_chat({}) || chat->format_chat(std::span(conversation).first(3))) return 1;
    const std::array invalid{kidi::text::ChatMessage{"tool", "ignored"}, kidi::text::ChatMessage{"user", "Hi"}};
    if (chat->format_chat(invalid)) return 1;
    const auto simple = nlohmann::ordered_json::parse(R"([{"role":"user","content":"Hi"}])");
    auto structured = chat->format_chat_json(simple, {{"enable_thinking", true}}, false);
    if (!structured || *structured != "<bos>user:Hi;thinking") return 1;
    if (chat->format_chat_json(nlohmann::ordered_json::array()) ||
        chat->format_chat_json(simple, {{"messages", "reserved"}}) ||
        chat->format_chat_json(nlohmann::ordered_json::parse(R"([{"content":"missing role"}])")))
        return 1;
    auto templated = kidi::text::Tokenizer::load(plain_path, {{"synthesis", synthesis_template}});
    if (!templated) return 1;
    auto synthesis = templated->format_template(
        "synthesis", {{"language", "English"}, {"instruction", "female"}, {"text", "Hello"}});
    if (!synthesis || *synthesis != "<lang>English</lang><style>female</style>Hello") return 1;
    if (templated->format_template("missing", {})) return 1;

    std::ofstream(directory / "chat_template.jinja")
        << "{% if enable_thinking is defined %}specified{% else %}default{% endif %}";
    auto defaults = kidi::text::Tokenizer::load(plain_path);
    if (!defaults) return 1;
    auto unspecified = defaults->format_chat_json(simple);
    auto disabled = defaults->format_chat_json(simple, {{"enable_thinking", false}});
    if (!unspecified || *unspecified != "default" || !disabled || *disabled != "specified") return 1;

    std::filesystem::copy_file(KIDI_INKLING_TEMPLATE, directory / "chat_template.jinja",
                               std::filesystem::copy_options::overwrite_existing);
    auto inkling = kidi::text::Tokenizer::load(plain_path);
    if (!inkling) return 1;
    auto history = nlohmann::ordered_json::parse(R"([
      {"role":"system","content":"Be brief."},
      {"role":"user","content":"Weather?"},
      {"role":"assistant","content":null,"reasoning_content":"Look it up.",
       "tool_calls":[{"id":"call1","function":{"name":"weather","arguments":{"city":"Paris"}}}]},
      {"role":"tool","tool_call_id":"call1","content":"Sunny"}
    ])");
    const auto args = nlohmann::ordered_json::parse(R"({
      "tools":[{"type":"function","function":{"name":"weather","description":"Forecast",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
      "reasoning_effort":"low"
    })");
    auto agentic = inkling->format_chat_json(history, args);
    const std::string expected =
        R"(<|message_system|>tool_declare<|content_xml|>[{"description":"Forecast","name":"weather","parameters":{"properties":{"city":{"type":"string"}},"type":"object"},"type":"function"}]<|end_message|>)"
        "<|message_system|><|content_text|>Be brief.<|end_message|>"
        "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
        "<|message_user|><|content_text|>Weather?<|end_message|>"
        "<|message_model|><|content_thinking|>Look it up.<|end_message|>"
        R"(<|message_model|>weather<|content_invoke_tool_json|>{"name":"weather","args":{"city":"Paris"}}<|end_message|>)"
        "<|content_model_end_sampling|><|message_tool|>weather<|content_text|>Sunny<|end_message|>";
    if (!agentic || *agentic != expected) {
        std::cerr << "Inkling structured chat mismatch: " << (agentic ? *agentic : agentic.error().message) << '\n';
        return 1;
    }
    if (inkling->format_chat_json(history, {{"reasoning_effort", "bad"}})) return 1;
    history.push_back({{"role", "system"}, {"content", "Be concise."}});
    if (!inkling->format_chat_json(history, args)) return 1;
    history[2]["tool_calls"][0]["function"]["arguments"] = R"({"city":"Paris"})";
    if (inkling->format_chat_json(history, args)) return 1;

    std::ofstream(directory / "bytes.json") << R"({
      "version":"1.0", "decoder":{"type":"ByteFallback"},
      "model":{"type":"WordLevel", "unk_token":"<unk>",
               "vocab":{"<unk>":0,"<0xE2>":1,"<0x82>":2,"<0xAC>":3,"!":4}}
    })";
    auto bytes = kidi::text::Tokenizer::load(directory / "bytes.json");
    if (!bytes) return 1;
    const std::array<std::int32_t, 4> byte_tokens{1, 2, 3, 4};
    std::string emitted;
    for (std::size_t count = 1; count <= byte_tokens.size(); ++count) {
        auto delta = bytes->decode_delta(std::span(byte_tokens).first(count), emitted);
        if (!delta || (count < 3 && !delta->empty())) return 1;
    }
    auto tail = bytes->decode_delta(byte_tokens, emitted, true);
    auto complete = bytes->decode(byte_tokens);
    if (!tail || !tail->empty() || !complete || emitted != *complete || emitted != "\xE2\x82\xAC!") return 1;
    emitted.clear();
    auto partial = bytes->decode_delta(std::span(byte_tokens).first(1), emitted);
    auto final = bytes->decode_delta(std::span(byte_tokens).first(1), emitted, true);
    if (!partial || !partial->empty() || !final || final->empty()) return 1;

    std::filesystem::remove_all(directory);
    return 0;
}