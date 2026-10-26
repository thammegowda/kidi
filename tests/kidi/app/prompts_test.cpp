#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

#include "kidi/app/prompts.h"

namespace {
auto fail(std::string_view message) -> int {
    std::cerr << message << '\n';
    return 1;
}
} // namespace

auto main() -> int {
    auto source = YAML::LoadFile(KIDI_APPLICATION_CONFIG);
    auto prompts = kidi::app::application_prompts(source);
    if (!prompts) return fail(prompts.error().message);
    if (prompts->size() != 14 || prompts->contains("chat") ||
        !prompts->at("system").contains("preserve natural code switching") ||
        !prompts->at("system").contains("untrusted reference material") ||
        prompts->at("translate") != prompts->at("page_translate") ||
        !prompts->at("page_summarize").contains("cumulative summary") ||
        prompts->at("summarize") == prompts->at("page_summarize"))
        return fail("canonical prompt definitions were not unified");
    auto overrides = YAML::Load(
        "prompts: {system: Native, translate: 'Translate {target_language}', custom: '42'}\n"
        "runtime:\n  wasm:\n    prompts: {system: Browser}\n  native:\n    prompts: {system: Android}\n");
    auto wasm = kidi::app::application_prompts(overrides, "wasm");
    auto native = kidi::app::application_prompts(overrides, "native");
    if (!wasm || !native || wasm->at("system") != "Browser" || native->at("system") != "Android" ||
        native->at("custom") != "42" || native->at("translate") != "Translate {target_language}" ||
        overrides["prompts"]["system"].as<std::string>() != "Native")
        return fail("runtime prompt overrides did not preserve the shared map");
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
        if (kidi::app::application_prompts(YAML::Load(yaml))) return fail("accepted invalid or model-local prompts");
    }
    for (const auto* yaml : {
             "prompts: {system: Chat, translate: Translation}\nruntime: {native: {prompts: []}}",
             "prompts: {system: Chat, translate: Translation}\nruntime: {native: []}",
             "prompts: {system: Chat, translate: Translation}\nruntime: {native: {models: {chat: {system_prompt: "
             "Old}}}}",
         }) {
        if (kidi::app::application_prompts(YAML::Load(yaml), "native")) return fail("accepted invalid runtime prompts");
    }
    return 0;
}
