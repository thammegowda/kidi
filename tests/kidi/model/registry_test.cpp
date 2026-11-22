#include "kidi/model/registry.h"

#include <gtest/gtest.h>

namespace {
using kidi::model::supports_task;
using kidi::model::Task;
using kidi::model::validate_model_config;

TEST(RegistryTest, SupportsTasks) {
    EXPECT_TRUE(supports_task("rtg_transformer_nmt", Task::TRANSLATION));
    EXPECT_TRUE(supports_task("gemma4_text", Task::TEXT_GENERATION));
    EXPECT_TRUE(supports_task("whisper", Task::SPEECH_RECOGNITION));
    EXPECT_TRUE(supports_task("omnivoice", Task::TTS));
    EXPECT_TRUE(supports_task("kokoro", Task::TTS));
    EXPECT_FALSE(supports_task("gemma4_assistant", Task::TEXT_GENERATION));
    EXPECT_FALSE(supports_task("omnivoice", Task::TEXT_GENERATION));
    EXPECT_FALSE(supports_task("unknown", Task::TTS));
}

TEST(RegistryTest, RejectsInvalidModelConfigs) {
    YAML::Node moe_gemma;
    moe_gemma["type"] = "gemma4_text";
    moe_gemma["enable_moe_block"] = true;
    EXPECT_FALSE(validate_model_config(moe_gemma));

    YAML::Node bad_assistant;
    bad_assistant["type"] = "gemma4_assistant";
    bad_assistant["text_config"]["enable_moe_block"] = true;
    EXPECT_FALSE(validate_model_config(bad_assistant));

    YAML::Node bad_whisper;
    bad_whisper["type"] = "whisper";
    bad_whisper["model_type"] = "whisper";
    EXPECT_FALSE(validate_model_config(bad_whisper));
}
} // namespace
