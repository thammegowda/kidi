#include "kidi/inference/synthesizer.h"
#include "kidi/model/omnivoice.h"

#include <array>
#include <cmath>

#include <gtest/gtest.h>

namespace {

TEST(VoiceTest, ParsesSynthesisNumbers) {
    auto decimal = kidi::inference::parse_synthesis_number("1.25");
    auto exponent = kidi::inference::parse_synthesis_number("2e-2");
    ASSERT_TRUE(decimal) << "portable synthesis number parsing failed";
    EXPECT_NEAR(*decimal, 1.25F, 1e-6F) << "portable synthesis number parsing failed";
    ASSERT_TRUE(exponent) << "portable synthesis number parsing failed";
    EXPECT_NEAR(*exponent, 0.02F, 1e-6F) << "portable synthesis number parsing failed";
    for (const auto* invalid : {"", " 1", "1 ", "1x", "nan", "inf"})
        EXPECT_FALSE(kidi::inference::parse_synthesis_number(invalid))
            << "invalid synthesis number was accepted: " << invalid;
}

TEST(VoiceTest, ParsesAttributes) {
    const std::array specifications{
        std::string{"gender=female"}, std::string{"age=young-adult"}, std::string{"pitch=high"},
        std::string{"style=whisper"}, std::string{"accent=british"},
    };
    auto attributes = kidi::inference::parse_voice_attributes(specifications);
    ASSERT_TRUE(attributes) << "generic voice attribute parsing failed";
    ASSERT_EQ(attributes->size(), 5) << "generic voice attribute parsing failed";
    EXPECT_EQ(attributes->at("gender"), "female") << "generic voice attribute parsing failed";
    EXPECT_EQ(attributes->at("age"), "young-adult") << "generic voice attribute parsing failed";
    EXPECT_EQ(attributes->at("pitch"), "high") << "generic voice attribute parsing failed";
    EXPECT_EQ(attributes->at("style"), "whisper") << "generic voice attribute parsing failed";
    EXPECT_EQ(attributes->at("accent"), "british") << "generic voice attribute parsing failed";
    auto instruction = kidi::model::OmniVoiceImpl::voice_instruction(*attributes, "Hello", "English");
    ASSERT_TRUE(instruction) << "English OmniVoice instruction failed";
    EXPECT_EQ(*instruction, "female, young adult, high pitch, whisper, british accent")
        << "English OmniVoice instruction failed";
}

TEST(VoiceTest, BuildsChineseInstructions) {
    const kidi::inference::VoiceAttributes chinese_attributes{
        {"gender", "male"},
        {"pitch", "high"},
    };
    auto chinese =
        kidi::model::OmniVoiceImpl::voice_instruction(chinese_attributes, "\xE4\xBD\xA0\xE5\xA5\xBD", "Chinese");
    ASSERT_TRUE(chinese) << "Chinese OmniVoice instruction failed";
    EXPECT_EQ(*chinese, "男，高音调") << "Chinese OmniVoice instruction failed";
    auto dialect = kidi::model::OmniVoiceImpl::voice_instruction({{"gender", "female"}, {"dialect", "四川话"}},
                                                                 "\xE4\xBD\xA0\xE5\xA5\xBD", "zh");
    ASSERT_TRUE(dialect) << "Chinese OmniVoice dialect instruction failed";
    EXPECT_EQ(*dialect, "女，四川话") << "Chinese OmniVoice dialect instruction failed";
}

TEST(VoiceTest, RejectsInvalidAttributes) {
    const std::array duplicate_specs{std::string{"gender=female"}, std::string{"gender=male"}};
    EXPECT_FALSE(kidi::inference::parse_voice_attributes(duplicate_specs)) << "duplicate voice attribute was accepted";
    const std::array malformed_specs{std::string{"gender"}};
    EXPECT_FALSE(kidi::inference::parse_voice_attributes(malformed_specs)) << "malformed voice attribute was accepted";
    EXPECT_FALSE(kidi::model::OmniVoiceImpl::voice_instruction({{"unknown", "value"}}, "Hello", "English"))
        << "unsupported OmniVoice attribute was accepted";
    EXPECT_FALSE(kidi::model::OmniVoiceImpl::voice_instruction({{"accent", "british"}, {"dialect", "四川话"}}, "Hello",
                                                               "English"))
        << "conflicting accent and dialect were accepted";
}
} // namespace
