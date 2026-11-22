#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <zlib.h>

#include "kidi/checkpoint/hub.h"
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

const std::array CONVERSATION{
    kidi::text::ChatMessage{"system", "Be brief."}, kidi::text::ChatMessage{"user", "Hi\nthere"},
    kidi::text::ChatMessage{"assistant", "Hello"}, kidi::text::ChatMessage{"user", "Again"}};

// Checks a real Gemma tokenizer from the default local model hub.
TEST(TokenizerTest, CheckpointChatTemplate) {
    const auto file = kidi::checkpoint::Hub{}.resolve("@google/gemma-4-E2B-it/tokenizer.json");
    if (!file && file.error().code == kidi::ErrorCode::IO) GTEST_SKIP() << file.error().message;
    ASSERT_TRUE(file.has_value()) << file.error().message;
    auto checkpoint = kidi::text::Tokenizer::load(*file);
    ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error().message;
    const std::array messages{kidi::text::ChatMessage{"user", "Hello"}};
    auto rendered = checkpoint->format_chat(messages);
    const std::string expected = "<bos><|turn>user\nHello<turn|>\n<|turn>model\n";
    ASSERT_TRUE(rendered.has_value()) << "checkpoint chat mismatch: " << rendered.error().message;
    EXPECT_EQ(*rendered, expected) << "checkpoint chat mismatch";
    auto multi_turn = checkpoint->format_chat(CONVERSATION);
    ASSERT_TRUE(multi_turn.has_value()) << multi_turn.error().message;
    EXPECT_EQ(*multi_turn, "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nHi\nthere<turn|>\n"
                           "<|turn>model\nHello<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n");
}

class TokenizerFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        directory = std::filesystem::temp_directory_path() /
                    (std::string("kidi-tokenizer-test-") +
                     ::testing::UnitTest::GetInstance()->current_test_info()->name());
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        plain_path = directory / "tokenizer.json";
        gzip_path = directory / "tokenizer.json.gz";
        std::ofstream(plain_path, std::ios::binary) << TOKENIZER_JSON;
        ASSERT_TRUE(write_gzip(gzip_path, TOKENIZER_JSON)) << "failed to create gzip fixture";
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    auto write_chat_template() const -> void {
        std::ofstream(directory / "chat_template.jinja")
            << "{{ bos_token }}{% for message in messages %}{{ message.role }}:{{ message.content }};{% endfor %}"
               "{% if add_generation_prompt %}assistant:{% endif %}{% if enable_thinking %}thinking{% endif %}";
    }

    auto write_tokenizer_config() const -> void {
        std::ofstream(directory / "tokenizer_config.json") << R"({"bos_token":"<bos>"})";
    }

    std::filesystem::path directory, plain_path, gzip_path;
};

TEST_F(TokenizerFileTest, PlainAndCompressed) {
    auto plain = kidi::text::Tokenizer::load(plain_path);
    auto compressed = kidi::text::Tokenizer::load(gzip_path);
    ASSERT_TRUE(plain.has_value()) << "failed to load tokenizer fixtures: " << plain.error().message;
    ASSERT_TRUE(compressed.has_value()) << "failed to load tokenizer fixtures: " << compressed.error().message;
    EXPECT_EQ(plain->vocabulary_size(), 3U) << "failed to load tokenizer fixtures";
    EXPECT_EQ(compressed->vocabulary_size(), 3U) << "failed to load tokenizer fixtures";

    auto plain_ids = plain->encode("café");
    auto compressed_ids = compressed->encode("café");
    ASSERT_TRUE(plain_ids.has_value()) << "plain and compressed tokenizers disagree";
    ASSERT_TRUE(compressed_ids.has_value()) << "plain and compressed tokenizers disagree";
    EXPECT_EQ(*plain_ids, *compressed_ids) << "plain and compressed tokenizers disagree";
    EXPECT_EQ(*plain_ids, std::vector<std::int32_t>{2}) << "plain and compressed tokenizers disagree";
    auto decoded = compressed->decode(*compressed_ids);
    ASSERT_TRUE(decoded.has_value()) << "UTF-8 tokenizer round trip failed: " << decoded.error().message;
    EXPECT_EQ(*decoded, "café") << "UTF-8 tokenizer round trip failed";
}

TEST_F(TokenizerFileTest, TemplateNeedsConfig) {
    auto plain = kidi::text::Tokenizer::load(plain_path);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_FALSE(plain->format_chat(CONVERSATION));
    write_chat_template();
    EXPECT_FALSE(kidi::text::Tokenizer::load(plain_path));
}

TEST_F(TokenizerFileTest, FormatsChat) {
    write_chat_template();
    write_tokenizer_config();
    auto chat = kidi::text::Tokenizer::load(plain_path);
    ASSERT_TRUE(chat.has_value()) << chat.error().message;
    auto formatted = chat->format_chat(CONVERSATION);
    ASSERT_TRUE(formatted.has_value()) << formatted.error().message;
    EXPECT_EQ(*formatted, "<bos>system:Be brief.;user:Hi\nthere;assistant:Hello;user:Again;assistant:");
    EXPECT_FALSE(chat->format_chat(std::span<const kidi::text::ChatMessage>{}));
    EXPECT_FALSE(chat->format_chat(std::span(CONVERSATION).first(3)));
    const std::array invalid{kidi::text::ChatMessage{"tool", "ignored"}, kidi::text::ChatMessage{"user", "Hi"}};
    EXPECT_FALSE(chat->format_chat(invalid));
    const auto simple = nlohmann::ordered_json::parse(R"([{"role":"user","content":"Hi"}])");
    auto structured = chat->format_chat(simple, {{"enable_thinking", true}}, false);
    ASSERT_TRUE(structured.has_value()) << structured.error().message;
    EXPECT_EQ(*structured, "<bos>user:Hi;thinking");
    EXPECT_FALSE(chat->format_chat(nlohmann::ordered_json::array()));
    EXPECT_FALSE(chat->format_chat(simple, {{"messages", "reserved"}}));
    EXPECT_FALSE(chat->format_chat(nlohmann::ordered_json::parse(R"([{"content":"missing role"}])")));
}

TEST_F(TokenizerFileTest, FormatsNamedTemplates) {
    write_chat_template();
    write_tokenizer_config();
    const auto synthesis_template = directory / "synthesis_template.jinja";
    std::ofstream(synthesis_template) << "<lang>{{- language -}}</lang><style>{{- instruction -}}</style>{{- text -}}";
    auto templated = kidi::text::Tokenizer::load(plain_path, {{"synthesis", synthesis_template}});
    ASSERT_TRUE(templated.has_value()) << templated.error().message;
    auto synthesis = templated->format_template(
        "synthesis", {{"language", "English"}, {"instruction", "female"}, {"text", "Hello"}});
    ASSERT_TRUE(synthesis.has_value()) << synthesis.error().message;
    EXPECT_EQ(*synthesis, "<lang>English</lang><style>female</style>Hello");
    EXPECT_FALSE(templated->format_template("missing", {}));
}

TEST_F(TokenizerFileTest, UnspecifiedThinking) {
    write_tokenizer_config();
    std::ofstream(directory / "chat_template.jinja")
        << "{% if enable_thinking is defined %}specified{% else %}default{% endif %}";
    auto defaults = kidi::text::Tokenizer::load(plain_path);
    ASSERT_TRUE(defaults.has_value()) << defaults.error().message;
    const auto simple = nlohmann::ordered_json::parse(R"([{"role":"user","content":"Hi"}])");
    auto unspecified = defaults->format_chat(simple);
    auto disabled = defaults->format_chat(simple, {{"enable_thinking", false}});
    ASSERT_TRUE(unspecified.has_value()) << unspecified.error().message;
    EXPECT_EQ(*unspecified, "default");
    ASSERT_TRUE(disabled.has_value()) << disabled.error().message;
    EXPECT_EQ(*disabled, "specified");
}

TEST_F(TokenizerFileTest, InklingAgenticChat) {
    write_tokenizer_config();
    std::filesystem::copy_file(KIDI_INKLING_TEMPLATE, directory / "chat_template.jinja",
                               std::filesystem::copy_options::overwrite_existing);
    auto inkling = kidi::text::Tokenizer::load(plain_path);
    ASSERT_TRUE(inkling.has_value()) << inkling.error().message;
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
    auto agentic = inkling->format_chat(history, args);
    const std::string expected =
        R"(<|message_system|>tool_declare<|content_xml|>[{"description":"Forecast","name":"weather","parameters":{"properties":{"city":{"type":"string"}},"type":"object"},"type":"function"}]<|end_message|>)"
        "<|message_system|><|content_text|>Be brief.<|end_message|>"
        "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
        "<|message_user|><|content_text|>Weather?<|end_message|>"
        "<|message_model|><|content_thinking|>Look it up.<|end_message|>"
        R"(<|message_model|>weather<|content_invoke_tool_json|>{"name":"weather","args":{"city":"Paris"}}<|end_message|>)"
        "<|content_model_end_sampling|><|message_tool|>weather<|content_text|>Sunny<|end_message|>";
    ASSERT_TRUE(agentic.has_value()) << "Inkling structured chat mismatch: " << agentic.error().message;
    EXPECT_EQ(*agentic, expected) << "Inkling structured chat mismatch";
    EXPECT_FALSE(inkling->format_chat(history, {{"reasoning_effort", "bad"}}));
    history.push_back({{"role", "system"}, {"content", "Be concise."}});
    EXPECT_TRUE(inkling->format_chat(history, args));
    history[2]["tool_calls"][0]["function"]["arguments"] = R"({"city":"Paris"})";
    EXPECT_FALSE(inkling->format_chat(history, args));
}

TEST_F(TokenizerFileTest, ByteFallbackDeltas) {
    std::ofstream(directory / "bytes.json") << R"({
      "version":"1.0", "decoder":{"type":"ByteFallback"},
      "model":{"type":"WordLevel", "unk_token":"<unk>",
               "vocab":{"<unk>":0,"<0xE2>":1,"<0x82>":2,"<0xAC>":3,"!":4}}
    })";
    auto bytes = kidi::text::Tokenizer::load(directory / "bytes.json");
    ASSERT_TRUE(bytes.has_value()) << bytes.error().message;
    const std::array<std::int32_t, 4> byte_tokens{1, 2, 3, 4};
    std::string emitted;
    for (std::size_t count = 1; count <= byte_tokens.size(); ++count) {
        auto delta = bytes->decode_delta(std::span(byte_tokens).first(count), emitted);
        ASSERT_TRUE(delta.has_value()) << "count=" << count << ": " << delta.error().message;
        if (count < 3) EXPECT_TRUE(delta->empty()) << "count=" << count;
    }
    auto tail = bytes->decode_delta(byte_tokens, emitted, true);
    auto complete = bytes->decode(byte_tokens);
    ASSERT_TRUE(tail.has_value()) << tail.error().message;
    EXPECT_TRUE(tail->empty());
    ASSERT_TRUE(complete.has_value()) << complete.error().message;
    EXPECT_EQ(emitted, *complete);
    EXPECT_EQ(emitted, "\xE2\x82\xAC!");
    emitted.clear();
    auto partial = bytes->decode_delta(std::span(byte_tokens).first(1), emitted);
    auto final = bytes->decode_delta(std::span(byte_tokens).first(1), emitted, true);
    ASSERT_TRUE(partial.has_value()) << partial.error().message;
    EXPECT_TRUE(partial->empty());
    ASSERT_TRUE(final.has_value()) << final.error().message;
    EXPECT_FALSE(final->empty());
}

} // namespace
