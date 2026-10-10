#include "kidi/inference/synthesizer.h"
#include "kidi/model/omnivoice.h"

#include <array>
#include <cmath>
#include <iostream>

auto main() -> int {
    auto decimal = kidi::inference::parse_synthesis_number("1.25");
    auto exponent = kidi::inference::parse_synthesis_number("2e-2");
    if (!decimal || std::abs(*decimal - 1.25F) > 1e-6F || !exponent || std::abs(*exponent - 0.02F) > 1e-6F) {
        std::cerr << "portable synthesis number parsing failed\n";
        return 1;
    }
    for (const auto* invalid : {"", " 1", "1 ", "1x", "nan", "inf"})
        if (kidi::inference::parse_synthesis_number(invalid)) {
            std::cerr << "invalid synthesis number was accepted: " << invalid << '\n';
            return 1;
        }

    const std::array specifications{
        std::string{"gender=female"}, std::string{"age=young-adult"}, std::string{"pitch=high"},
        std::string{"style=whisper"}, std::string{"accent=british"},
    };
    auto attributes = kidi::inference::parse_voice_attributes(specifications);
    if (!attributes || attributes->size() != 5 || attributes->at("gender") != "female" ||
        attributes->at("age") != "young-adult" || attributes->at("pitch") != "high" ||
        attributes->at("style") != "whisper" || attributes->at("accent") != "british") {
        std::cerr << "generic voice attribute parsing failed\n";
        return 1;
    }
    auto instruction = kidi::model::OmniVoiceImpl::voice_instruction(*attributes, "Hello", "English");
    if (!instruction || *instruction != "female, young adult, high pitch, whisper, british accent") {
        std::cerr << "English OmniVoice instruction failed\n";
        return 1;
    }

    const kidi::inference::VoiceAttributes chinese_attributes{
        {"gender", "male"},
        {"pitch", "high"},
    };
    auto chinese =
        kidi::model::OmniVoiceImpl::voice_instruction(chinese_attributes, "\xE4\xBD\xA0\xE5\xA5\xBD", "Chinese");
    if (!chinese || *chinese != "男，高音调") {
        std::cerr << "Chinese OmniVoice instruction failed\n";
        return 1;
    }
    auto dialect = kidi::model::OmniVoiceImpl::voice_instruction({{"gender", "female"}, {"dialect", "四川话"}},
                                                                 "\xE4\xBD\xA0\xE5\xA5\xBD", "zh");
    if (!dialect || *dialect != "女，四川话") {
        std::cerr << "Chinese OmniVoice dialect instruction failed\n";
        return 1;
    }

    const std::array duplicate_specs{std::string{"gender=female"}, std::string{"gender=male"}};
    if (kidi::inference::parse_voice_attributes(duplicate_specs)) {
        std::cerr << "duplicate voice attribute was accepted\n";
        return 1;
    }
    const std::array malformed_specs{std::string{"gender"}};
    if (kidi::inference::parse_voice_attributes(malformed_specs)) {
        std::cerr << "malformed voice attribute was accepted\n";
        return 1;
    }
    if (kidi::model::OmniVoiceImpl::voice_instruction({{"unknown", "value"}}, "Hello", "English")) {
        std::cerr << "unsupported OmniVoice attribute was accepted\n";
        return 1;
    }
    if (kidi::model::OmniVoiceImpl::voice_instruction({{"accent", "british"}, {"dialect", "四川话"}}, "Hello",
                                                      "English")) {
        std::cerr << "conflicting accent and dialect were accepted\n";
        return 1;
    }
    return 0;
}
