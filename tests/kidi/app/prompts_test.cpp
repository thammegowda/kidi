#include <string>
#include <yaml-cpp/yaml.h>

#include <gtest/gtest.h>

#include "kidi/app/prompts.h"

namespace {

TEST(PromptsTest, UnifiesDefinitions) {
    auto source = YAML::LoadFile(KIDI_APPLICATION_CONFIG);
    auto prompts = kidi::app::application_prompts(source);
    ASSERT_TRUE(prompts) << prompts.error().message;
    ASSERT_EQ(prompts->size(), 14) << "canonical prompt definitions were not unified";
    EXPECT_FALSE(prompts->contains("chat")) << "canonical prompt definitions were not unified";
    EXPECT_TRUE(prompts->at("system").contains("preserve natural code switching"))
        << "canonical prompt definitions were not unified";
    EXPECT_TRUE(prompts->at("system").contains("untrusted reference material"))
        << "canonical prompt definitions were not unified";
    EXPECT_EQ(prompts->at("translate"), prompts->at("page_translate"))
        << "canonical prompt definitions were not unified";
    EXPECT_TRUE(prompts->at("page_summarize").contains("cumulative summary"))
        << "canonical prompt definitions were not unified";
    EXPECT_NE(prompts->at("summarize"), prompts->at("page_summarize"))
        << "canonical prompt definitions were not unified";
}

TEST(PromptsTest, RuntimeOverrides) {
    auto overrides = YAML::Load(
        "prompts: {system: Native, translate: 'Translate {target_language}', custom: '42'}\n"
        "runtime:\n  wasm:\n    prompts: {system: Browser}\n  native:\n    prompts: {system: Android}\n");
    auto wasm = kidi::app::application_prompts(overrides, "wasm");
    auto native = kidi::app::application_prompts(overrides, "native");
    ASSERT_TRUE(wasm) << "runtime prompt overrides did not preserve the shared map";
    ASSERT_TRUE(native) << "runtime prompt overrides did not preserve the shared map";
    EXPECT_EQ(wasm->at("system"), "Browser") << "runtime prompt overrides did not preserve the shared map";
    EXPECT_EQ(native->at("system"), "Android") << "runtime prompt overrides did not preserve the shared map";
    EXPECT_EQ(native->at("custom"), "42") << "runtime prompt overrides did not preserve the shared map";
    EXPECT_EQ(native->at("translate"), "Translate {target_language}")
        << "runtime prompt overrides did not preserve the shared map";
    EXPECT_EQ(overrides["prompts"]["system"].as<std::string>(), "Native")
        << "runtime prompt overrides did not preserve the shared map";
}

TEST(PromptsTest, RejectsInvalid) {
    for (const auto* yaml : {
             "{}",
             "prompts: []",
             "prompts: {system: Chat}",
             "prompts: {translate: Translation}",
             "prompts: {system: '', translate: Translation}",
             "prompts: {system: [], translate: Translation}",
             "prompts: {system: true, translate: Translation}",
             "prompts: {system: 42, translate: Translation}",
             "prompts: {system: Chat, translate: Translation, chat: Duplicate}",
             "prompts: {system: Chat, translate: Translation, custom: {nested: value}}",
             "prompts: {system: Chat, system: Duplicate, translate: Translation}",
             "prompts: {system: Chat, translate: Translation}\nmodels: {chat: {system_prompt: Old}}",
             "prompts: {system: Chat, translate: Translation}\nmodels: {chat: {translation_prompt: Old}}",
             "prompts: {system: Chat, translate: Translation}\nmodels: {chat: {editor: {prompts: {polish: Old}}}}",
         }) {
        EXPECT_FALSE(kidi::app::application_prompts(YAML::Load(yaml)))
            << "accepted invalid or model-local prompts: " << yaml;
    }
}

TEST(PromptsTest, RejectsInvalidRuntimePrompts) {
    for (const auto* yaml : {
             "prompts: {system: Chat, translate: Translation}\nruntime: {native: {prompts: []}}",
             "prompts: {system: Chat, translate: Translation}\nruntime: {native: []}",
             "prompts: {system: Chat, translate: Translation}\nruntime: {native: {models: {chat: {system_prompt: "
             "Old}}}}",
         }) {
        EXPECT_FALSE(kidi::app::application_prompts(YAML::Load(yaml), "native"))
            << "accepted invalid runtime prompts: " << yaml;
    }
}

} // namespace
