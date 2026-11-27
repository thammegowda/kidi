#include "kidi/audio/duration.h"

#include <gtest/gtest.h>

namespace {

constexpr std::string_view CHINESE = "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\x96\xE7\x95\x8C\xEF\xBC\x81";

TEST(DurationTest, Weight) {
    const kidi::audio::RuleDurationEstimator estimator;
    auto latin = estimator.weight("a b!");
    ASSERT_TRUE(latin);
    EXPECT_NEAR(*latin, 2.7F, 1e-4F);
    auto chinese = estimator.weight(CHINESE);
    ASSERT_TRUE(chinese);
    EXPECT_NEAR(*chinese, 13.F, 1e-4F);
    auto devanagari = estimator.weight("\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87");
    ASSERT_TRUE(devanagari);
    EXPECT_NEAR(*devanagari, 7.2F, 1e-4F);
    EXPECT_FALSE(estimator.weight("\xFF"));
}

TEST(DurationTest, Estimate) {
    const kidi::audio::RuleDurationEstimator estimator;
    auto estimate = estimator.estimate("Hello from Kidi.", "Nice to meet you.", 25.F);
    ASSERT_TRUE(estimate);
    EXPECT_NEAR(*estimate, 39.496496F, 1e-4F);
    auto chinese_estimate = estimator.estimate(CHINESE, "Nice to meet you.", 25.F);
    ASSERT_TRUE(chinese_estimate);
    EXPECT_NEAR(*chinese_estimate, 38.624964F, 1e-4F);
    EXPECT_FALSE(estimator.estimate("text", "", 25.F));
}

} // namespace
