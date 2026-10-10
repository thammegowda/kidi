#include "kidi/audio/duration.h"

#include <cmath>

#include <uni_algo/conv.h>
#include <uni_algo/prop.h>
#include <uni_algo/script.h>

namespace kidi::audio {
namespace {

#define SCRIPT(tag) static_cast<char32_t>(una::locale::script{#tag})

enum class Script : char32_t {
    ARAB = SCRIPT(Arab),
    ARMN = SCRIPT(Armn),
    BALI = SCRIPT(Bali),
    BATK = SCRIPT(Batk),
    BENG = SCRIPT(Beng),
    BOPO = SCRIPT(Bopo),
    BUGI = SCRIPT(Bugi),
    CHAM = SCRIPT(Cham),
    CYRL = SCRIPT(Cyrl),
    DEVA = SCRIPT(Deva),
    ETHI = SCRIPT(Ethi),
    GEOR = SCRIPT(Geor),
    GREK = SCRIPT(Grek),
    GUJR = SCRIPT(Gujr),
    GURU = SCRIPT(Guru),
    HANG = SCRIPT(Hang),
    HANI = SCRIPT(Hani),
    HEBR = SCRIPT(Hebr),
    HIRA = SCRIPT(Hira),
    HRKT = SCRIPT(Hrkt),
    JAVA = SCRIPT(Java),
    KALI = SCRIPT(Kali),
    KANA = SCRIPT(Kana),
    KHMR = SCRIPT(Khmr),
    KNDA = SCRIPT(Knda),
    LANA = SCRIPT(Lana),
    LAOO = SCRIPT(Laoo),
    LATN = SCRIPT(Latn),
    LEPC = SCRIPT(Lepc),
    LIMB = SCRIPT(Limb),
    MLYM = SCRIPT(Mlym),
    MTEI = SCRIPT(Mtei),
    MYMR = SCRIPT(Mymr),
    OLCK = SCRIPT(Olck),
    ORYA = SCRIPT(Orya),
    RJNG = SCRIPT(Rjng),
    SAUR = SCRIPT(Saur),
    SINH = SCRIPT(Sinh),
    SUND = SCRIPT(Sund),
    SYLO = SCRIPT(Sylo),
    SYRC = SCRIPT(Syrc),
    TALE = SCRIPT(Tale),
    TALU = SCRIPT(Talu),
    TAML = SCRIPT(Taml),
    TAVT = SCRIPT(Tavt),
    TELU = SCRIPT(Telu),
    THAI = SCRIPT(Thai),
    TIBT = SCRIPT(Tibt),
    YIII = SCRIPT(Yiii),
};

#undef SCRIPT

auto character_weight(char32_t codepoint) -> float {
    if ((codepoint >= U'A' && codepoint <= U'Z') || (codepoint >= U'a' && codepoint <= U'z')) return 1.F;
    if (codepoint == U' ') return 0.2F;
    if (codepoint == 0x0640) return 0.F;

    const una::codepoint::prop property(codepoint);
    if (property.General_Category_M()) return 0.F;
    if (property.General_Category_P() || property.General_Category_S()) return 0.5F;
    if (property.General_Category_Z()) return 0.2F;
    if (property.General_Category_N()) return 3.5F;

    const auto script = static_cast<Script>(static_cast<char32_t>(una::codepoint::get_script(codepoint)));
    switch (script) {
        case Script::HANI:
        case Script::BOPO:
        case Script::ETHI:
        case Script::YIII:
            return 3.F;
        case Script::HANG:
            return 2.5F;
        case Script::HIRA:
        case Script::KANA:
        case Script::HRKT:
            return 2.2F;
        case Script::DEVA:
        case Script::BENG:
        case Script::GURU:
        case Script::GUJR:
        case Script::ORYA:
        case Script::TAML:
        case Script::TELU:
        case Script::KNDA:
        case Script::MLYM:
        case Script::SINH:
        case Script::TIBT:
        case Script::LIMB:
        case Script::TALE:
        case Script::TALU:
        case Script::BUGI:
        case Script::LANA:
        case Script::BALI:
        case Script::SUND:
        case Script::BATK:
        case Script::LEPC:
        case Script::OLCK:
        case Script::SYLO:
        case Script::SAUR:
        case Script::KALI:
        case Script::RJNG:
        case Script::JAVA:
        case Script::CHAM:
        case Script::TAVT:
        case Script::MTEI:
        case Script::KHMR:
        case Script::MYMR:
            return 1.8F;
        case Script::THAI:
        case Script::LAOO:
        case Script::ARAB:
        case Script::SYRC:
        case Script::HEBR:
            return 1.5F;
        case Script::LATN:
        case Script::CYRL:
        case Script::GREK:
        case Script::ARMN:
        case Script::GEOR:
            return 1.F;
    }
    return 1.F;
}

} // namespace

auto RuleDurationEstimator::weight(std::string_view text) const -> Result<float> {
    una::error error;
    const auto codepoints = una::strict::utf8to32u(text, error);
    if (error) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "TTS text is not valid UTF-8"});
    float result = 0;
    for (const auto codepoint : codepoints) result += character_weight(codepoint);
    return result;
}

auto RuleDurationEstimator::estimate(std::string_view target, std::string_view reference, float reference_duration,
                                     std::optional<float> low_threshold, float boost_strength) const -> Result<float> {
    if (!std::isfinite(reference_duration) || reference_duration <= 0 || reference.empty() ||
        (low_threshold && (!std::isfinite(*low_threshold) || *low_threshold <= 0)) || !std::isfinite(boost_strength) ||
        boost_strength <= 0)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid automatic-duration configuration"});
    const auto reference_weight = weight(reference);
    if (!reference_weight) return std::unexpected(reference_weight.error());
    const auto target_weight = weight(target);
    if (!target_weight) return std::unexpected(target_weight.error());
    if (*reference_weight <= 0)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "automatic-duration reference has no weight"});
    const auto estimate = *target_weight / (*reference_weight / reference_duration);
    if (low_threshold && estimate < *low_threshold)
        return *low_threshold * std::pow(estimate / *low_threshold, 1.F / boost_strength);
    return estimate;
}

} // namespace kidi::audio
