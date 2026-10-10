#include "kidi/audio/duration.h"

#include <cassert>
#include <cmath>

namespace {
auto close(float left, float right) -> bool { return std::abs(left - right) < 1e-4F; }
} // namespace

int main() {
    const kidi::audio::RuleDurationEstimator estimator;
    auto latin = estimator.weight("a b!");
    assert(latin && close(*latin, 2.7F));
    constexpr std::string_view CHINESE = "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\x96\xE7\x95\x8C\xEF\xBC\x81";
    auto chinese = estimator.weight(CHINESE);
    assert(chinese && close(*chinese, 13.F));
    auto devanagari = estimator.weight("\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87");
    assert(devanagari && close(*devanagari, 7.2F));
    assert(!estimator.weight("\xFF"));

    auto estimate = estimator.estimate("Hello from Kidi.", "Nice to meet you.", 25.F);
    assert(estimate && close(*estimate, 39.496496F));
    auto chinese_estimate = estimator.estimate(CHINESE, "Nice to meet you.", 25.F);
    assert(chinese_estimate && close(*chinese_estimate, 38.624964F));
    assert(!estimator.estimate("text", "", 25.F));
}
