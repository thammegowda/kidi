#include "kidi/model/omnivoice.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>

#include <uni_algo/conv.h>
#include <uni_algo/script.h>

#include "kidi/audio/duration.h"
#include "kidi/text/tokenizer.h"
namespace kidi::model {
using ops::require;

namespace {

auto decoder_shape(const YAML::Node& config) -> layers::HiggsDecoderShape {
    const auto codec = config["codec_config"];
    return {
        .codebooks = config["num_audio_codebook"].as<std::int32_t>(),
        .codebook_size = codec["codebook_size"].as<std::int32_t>(),
        .codebook_width = codec["codebook_dim"].as<std::int32_t>(),
        .latent_width = codec["hidden_size"].as<std::int32_t>(),
        .acoustic_width = codec["acoustic_hidden_size"].as<std::int32_t>(),
        .decoder_width = codec["decoder_hidden_size"].as<std::int32_t>(),
        .upsampling_ratios = codec["upsampling_ratios"].as<std::vector<std::int32_t>>(),
    };
}

auto clean_text(std::string_view text) -> std::string {
    std::string result;
    result.reserve(text.size());
    bool space = false;
    for (const char value : text) {
        if (value == '\r' || value == '\n' || value == '\t' || value == ' ') {
            space = !result.empty();
            continue;
        }
        if (space) result.push_back(' ');
        result.push_back(value);
        space = false;
    }
    while (!result.empty() && result.back() == ' ') result.pop_back();
    return result;
}

auto log_sum_exp(std::span<const float> values) -> float {
    const auto maximum = *std::ranges::max_element(values);
    double sum = 0;
    for (const auto value : values) sum += std::exp(static_cast<double>(value - maximum));
    return maximum + static_cast<float>(std::log(sum));
}

struct Candidate {
    float score;
    std::size_t index;
    std::int32_t token;
};

auto prepare_input(std::span<const std::int32_t> prefix, std::span<const std::int32_t> audio, std::int32_t codebooks)
    -> std::vector<std::int32_t> {
    const auto target = audio.size() / static_cast<std::size_t>(codebooks);
    const auto length = prefix.size() + target;
    std::vector<std::int32_t> result(static_cast<std::size_t>(codebooks) * length);
    for (std::int32_t codebook = 0; codebook < codebooks; ++codebook) {
        std::ranges::copy(prefix, result.begin() + static_cast<std::size_t>(codebook) * length);
        std::ranges::copy(audio.subspan(static_cast<std::size_t>(codebook) * target, target),
                          result.begin() + static_cast<std::size_t>(codebook) * length + prefix.size());
    }
    return result;
}

auto postprocess(std::vector<float> audio, std::uint32_t sample_rate) -> std::vector<float> {
    const auto peak = std::abs(std::ranges::max(audio, {}, [](float value) { return std::abs(value); }));
    if (peak > 1e-6F)
        for (auto& value : audio) value = value / peak * 0.5F;
    const auto padding = static_cast<std::size_t>(sample_rate / 10);
    const auto fade = std::min(padding, audio.size() / 2);
    for (std::size_t index = 0; index < fade; ++index) {
        const auto scale = static_cast<float>(index) / static_cast<float>(fade);
        audio[index] *= scale;
        audio[audio.size() - 1 - index] *= scale;
    }
    std::vector<float> padded(audio.size() + 2 * padding, 0.F);
    std::ranges::copy(audio, padded.begin() + padding);
    return padded;
}

auto trim(std::string_view value) -> std::string {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.remove_suffix(1);
    return std::string(value);
}

auto normalized(std::string_view value) -> std::string {
    auto result = trim(value);
    for (auto& character : result) {
        const auto byte = static_cast<unsigned char>(character);
        if (character == '_' || character == '-')
            character = ' ';
        else if (byte < 0x80)
            character = static_cast<char>(std::tolower(byte));
    }
    std::string collapsed;
    collapsed.reserve(result.size());
    bool space = false;
    for (const auto character : result) {
        if (character == ' ') {
            space = !collapsed.empty();
            continue;
        }
        if (space) collapsed.push_back(' ');
        collapsed.push_back(character);
        space = false;
    }
    return collapsed;
}

template <std::size_t Size>
auto one_of(std::string_view value, const std::array<std::string_view, Size>& choices) -> bool {
    return std::ranges::find(choices, value) != choices.end();
}

auto has_han(std::string_view text) -> bool {
    const auto codepoints = una::utf8to32u(text);
    return std::ranges::any_of(codepoints, [](char32_t codepoint) {
        return una::codepoint::get_script(codepoint) == una::locale::script{"Hani"};
    });
}

auto explicit_language(std::string_view language) -> std::string {
    auto value = normalized(language);
    if (value.empty() || value == "auto" || value == "none") return {};
    return value;
}

auto english_language(std::string_view language) -> bool {
    return language == "en" || language == "english" || language.starts_with("en ");
}

auto chinese_language(std::string_view language) -> bool {
    return language == "zh" || language == "chinese" || language == "mandarin" || language.starts_with("zh ");
}

auto unsupported_value(std::string_view key, std::string_view value) -> std::unexpected<Error> {
    return std::unexpected(
        Error{ErrorCode::INVALID_ARGUMENT, "unsupported OmniVoice " + std::string(key) + ": " + std::string(value)});
}

} // namespace

struct OmniVoiceImpl::State {
    ops::Context context{tensor::Device::cpu()};
    std::int32_t audio_vocabulary, mask_id, codebooks;
    std::uint32_t sample_rate, frame_rate;
    Qwen3 llm;
    layers::TokenEmbedding audio_embeddings;
    layers::Linear audio_heads;
    layers::HiggsDecoder audio_tokenizer;
    YAML::Node package;
    std::optional<text::Tokenizer> tokenizer;

    explicit State(const YAML::Node& config)
        : audio_vocabulary(config["audio_vocab_size"].as<std::int32_t>()),
          mask_id(config["audio_mask_id"].as<std::int32_t>()),
          codebooks(config["num_audio_codebook"].as<std::int32_t>()),
          sample_rate(config["codec_config"]["sample_rate"].as<std::uint32_t>()),
          frame_rate(config["codec_config"]["frame_rate"].as<std::uint32_t>()),
          llm(config["llm_config"]),
          audio_embeddings(codebooks * audio_vocabulary, llm->hidden_size(), 1.F),
          audio_heads(llm->hidden_size(), codebooks * audio_vocabulary, true, false),
          audio_tokenizer(decoder_shape(config)) {}
};

auto OmniVoiceImpl::validate_config(const YAML::Node& config) -> Result<void> {
    try {
        if (!config.IsMap() || config["type"].as<std::string>() != "omnivoice")
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "expected an OmniVoice model"});
        auto qwen = Qwen3Impl::validate_config(config["llm_config"]);
        if (!qwen) return qwen;
        const auto audio_vocabulary = config["audio_vocab_size"].as<std::int32_t>();
        const auto mask = config["audio_mask_id"].as<std::int32_t>();
        const auto codebooks = config["num_audio_codebook"].as<std::int32_t>();
        const auto codec = config["codec_config"];
        const auto ratios = codec["upsampling_ratios"].as<std::vector<std::int32_t>>();
        if (audio_vocabulary <= 1 || mask < 0 || mask >= audio_vocabulary || codebooks <= 0 ||
            codec["codebook_size"].as<std::int32_t>() <= 0 || codec["codebook_dim"].as<std::int32_t>() <= 0 ||
            codec["hidden_size"].as<std::int32_t>() <= 0 || codec["acoustic_hidden_size"].as<std::int32_t>() <= 0 ||
            codec["decoder_hidden_size"].as<std::int32_t>() <= 0 || ratios.empty() ||
            std::ranges::any_of(
                ratios, [](auto value) { return value <= 0; }) ||
            codec["sample_rate"].as<std::uint32_t>() == 0 || codec["frame_rate"].as<std::uint32_t>() == 0)
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid OmniVoice codec configuration"});
        return {};
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, error.what()});
    }
}

OmniVoiceImpl::OmniVoiceImpl(const YAML::Node& config) {
    auto valid = validate_config(config);
    if (!valid) throw ops::Failure(valid.error());
    if (module_dtype != tensor::DType::I8 || module_device != tensor::Device::cpu())
        throw ops::Failure({ErrorCode::UNSUPPORTED, "OmniVoice requires INT8 parameters on the CPU"});
    impl_ = std::make_unique<State>(config);
    register_module("llm", impl_->llm);
    register_module("audio_embeddings", impl_->audio_embeddings);
    register_module("audio_heads", impl_->audio_heads);
    register_module("audio_tokenizer", impl_->audio_tokenizer);
}

OmniVoiceImpl::~OmniVoiceImpl() = default;

auto OmniVoiceImpl::create(const YAML::Node& config) -> Result<OmniVoice> {
    auto valid = validate_config(config);
    if (!valid) return std::unexpected(std::move(valid.error()));
    try {
        return OmniVoice(config);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto OmniVoiceImpl::load(YAML::Node package, tensor::Device device) -> Result<OmniVoice> {
    try {
        if (device != tensor::Device::cpu())
            throw ops::Failure({ErrorCode::UNSUPPORTED, "OmniVoice currently supports the CPU only"});
        auto tokenizer = require(text::Tokenizer::load(package["tokenizer_file"].as<std::string>(),
                                                       {{"synthesis", package["template_file"].as<std::string>()}}));
        auto weights = require(checkpoint::Weights::load(package["weights_file"].as<std::string>()));
        const ModuleScope construction(tensor::DType::I8, false, tensor::Device::cpu());
        auto model = require(create(package["model"]));
        require(model->set_checkpoint(weights));
        if (tokenizer.vocabulary_size() != static_cast<std::size_t>(model->text_vocabulary_size()))
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "OmniVoice tokenizer vocabulary mismatch"});
        model->impl_->package = std::move(package);
        model->impl_->tokenizer.emplace(std::move(tokenizer));
        return model;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto OmniVoiceImpl::voice_instruction(const inference::VoiceAttributes& attributes, std::string_view text,
                                      std::string_view language) -> Result<std::string> {
    static constexpr std::array KEYS{
        std::string_view{"gender"}, std::string_view{"age"},    std::string_view{"pitch"},
        std::string_view{"style"},  std::string_view{"accent"}, std::string_view{"dialect"},
    };
    static constexpr std::array GENDERS{std::string_view{"male"}, std::string_view{"female"}};
    static constexpr std::array AGES{
        std::string_view{"child"},       std::string_view{"teenager"},    std::string_view{"young adult"},
        std::string_view{"middle aged"}, std::string_view{"middle-aged"}, std::string_view{"elderly"},
    };
    static constexpr std::array PITCHES{
        std::string_view{"very low"}, std::string_view{"low"},       std::string_view{"moderate"},
        std::string_view{"high"},     std::string_view{"very high"},
    };
    static constexpr std::array ACCENTS{
        std::string_view{"american"}, std::string_view{"british"},    std::string_view{"australian"},
        std::string_view{"chinese"},  std::string_view{"canadian"},   std::string_view{"indian"},
        std::string_view{"korean"},   std::string_view{"portuguese"}, std::string_view{"russian"},
        std::string_view{"japanese"},
    };
    static const std::set<std::string, std::less<>> DIALECTS{
        "河南话", "陕西话",   "四川话", "贵州话", "云南话", "桂林话",
        "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话",
    };
    static const std::map<std::string, std::string, std::less<>> CHINESE{
        {"male", "男"},           {"female", "女"},
        {"child", "儿童"},        {"teenager", "少年"},
        {"young adult", "青年"},  {"middle-aged", "中年"},
        {"elderly", "老年"},      {"very low pitch", "极低音调"},
        {"low pitch", "低音调"},  {"moderate pitch", "中音调"},
        {"high pitch", "高音调"}, {"very high pitch", "极高音调"},
        {"whisper", "耳语"},
    };

    for (const auto& [key, value] : attributes)
        if (!one_of(key, KEYS))
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "OmniVoice does not support voice attribute: " + key});

    std::map<std::string, std::string, std::less<>> selected;
    for (const auto& [key, raw] : attributes) {
        auto value = normalized(raw);
        if (value == "auto") continue;
        if (key == "gender") {
            if (!one_of(value, GENDERS)) return unsupported_value(key, raw);
        } else if (key == "age") {
            if (!one_of(value, AGES)) return unsupported_value(key, raw);
            if (value == "middle aged") value = "middle-aged";
        } else if (key == "pitch") {
            if (value.ends_with(" pitch")) value.resize(value.size() - std::string_view(" pitch").size());
            if (!one_of(value, PITCHES)) return unsupported_value(key, raw);
            value += " pitch";
        } else if (key == "style") {
            if (value != "whisper") return unsupported_value(key, raw);
        } else if (key == "accent") {
            if (value.ends_with(" accent")) value.resize(value.size() - std::string_view(" accent").size());
            if (!one_of(value, ACCENTS)) return unsupported_value(key, raw);
            value += " accent";
        } else if (key == "dialect") {
            value = trim(raw);
            if (!DIALECTS.contains(value)) return unsupported_value(key, raw);
        }
        selected.emplace(key, std::move(value));
    }

    const auto accent = selected.contains("accent");
    const auto dialect = selected.contains("dialect");
    if (accent && dialect)
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "OmniVoice cannot combine an English accent and Chinese dialect"});
    const auto declared_language = explicit_language(language);
    if (accent && !declared_language.empty() && !english_language(declared_language))
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "OmniVoice accent requires English language conditioning"});
    if (dialect && !declared_language.empty() && !chinese_language(declared_language))
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "OmniVoice dialect requires Chinese language conditioning"});

    const auto use_chinese = dialect || (!accent && has_han(text));
    std::vector<std::string> values;
    for (const auto key : KEYS) {
        const auto found = selected.find(key);
        if (found == selected.end()) continue;
        auto value = found->second;
        if (use_chinese)
            if (const auto translated = CHINESE.find(value); translated != CHINESE.end()) value = translated->second;
        values.push_back(std::move(value));
    }
    const auto separator = use_chinese ? "，" : ", ";
    std::string result;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) result += separator;
        result += values[index];
    }
    return result;
}

auto OmniVoiceImpl::set_checkpoint(const checkpoint::Weights& weights) -> Result<void> {
    for (const auto& name : weights.names()) {
        auto value = weights.tensor(name);
        if (!value) return std::unexpected(std::move(value.error()));
        const auto scale = name.ends_with(".scale") || name.ends_with("_scale");
        if ((!scale && value->dtype() != tensor::DType::I8) || (scale && value->dtype() != tensor::DType::F32))
            return std::unexpected(
                Error{ErrorCode::INVALID_ARGUMENT, "OmniVoice checkpoint is not fully INT8: " + name});
    }
    return set_state(weights);
}

auto OmniVoiceImpl::forward(std::span<const std::int32_t> input_ids, std::span<const std::uint8_t> audio_mask,
                            std::size_t target_start, std::size_t target_length) -> Result<tensor::Tensor> {
    try {
        const auto length = audio_mask.size();
        if (!length || input_ids.size() != static_cast<std::size_t>(impl_->codebooks) * length ||
            target_start > length || !target_length || target_length > length - target_start)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid OmniVoice input dimensions"});
        std::vector<std::int32_t> text_tokens(length);
        std::ranges::copy(input_ids.first(length), text_tokens.begin());
        auto text = impl_->llm->embed(impl_->context, text_tokens);

        tensor::Tensor audio;
        std::vector<std::int32_t> shifted(length);
        for (std::int32_t codebook = 0; codebook < impl_->codebooks; ++codebook) {
            for (std::size_t position = 0; position < length; ++position) {
                const auto token =
                    audio_mask[position] ? input_ids[static_cast<std::size_t>(codebook) * length + position] : 0;
                if (token < 0 || token >= impl_->audio_vocabulary)
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "audio token is outside the vocabulary"});
                shifted[position] = token + codebook * impl_->audio_vocabulary;
            }
            auto embedded = impl_->audio_embeddings->forward(impl_->context, shifted);
            audio = audio.defined() ? impl_->context.add(audio, embedded) : std::move(embedded);
        }

        impl_->context.synchronize();
        const auto text_values = require(text.data<float>());
        const auto audio_values = require(audio.data<float>());
        auto combined = require(tensor::Tensor::empty({1, static_cast<std::int64_t>(length), impl_->llm->hidden_size()},
                                                      tensor::DType::F32));
        auto values = require(combined.data<float>());
        const auto width = static_cast<std::size_t>(impl_->llm->hidden_size());
        for (std::size_t position = 0; position < length; ++position) {
            const auto source = audio_mask[position] ? audio_values : text_values;
            std::ranges::copy(source.subspan(position * width, width), values.begin() + position * width);
        }
        auto hidden = impl_->llm->forward(impl_->context, combined);
        hidden = impl_->context.slice(hidden, 1, target_start, target_length);
        return impl_->audio_heads->forward(impl_->context, hidden);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto OmniVoiceImpl::decode(std::span<const std::int32_t> tokens, std::size_t length) -> Result<std::vector<float>> {
    try {
        auto decoded = impl_->audio_tokenizer->forward(impl_->context, tokens, length);
        impl_->context.synchronize();
        const auto values = require(decoded.data<float>());
        return std::vector<float>(values.begin(), values.end());
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto OmniVoiceImpl::target_length(std::string_view text, const inference::SynthesisOptions& options) const
    -> std::size_t {
    if (options.duration_seconds) {
        if (!std::isfinite(*options.duration_seconds) || *options.duration_seconds <= 0)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "TTS duration must be positive and finite"});
        return std::max<std::size_t>(1, static_cast<std::size_t>(*options.duration_seconds * frame_rate()));
    }
    const auto settings = impl_->package["synthesis"]["auto_duration"];
    const auto estimate = require(audio::RuleDurationEstimator().estimate(
        text, settings["reference_text"].as<std::string>("Nice to meet you."),
        settings["reference_tokens"].as<float>(25.F), settings["low_threshold"].as<float>(50.F),
        settings["boost_strength"].as<float>(3.F)));
    return std::max<std::size_t>(1, static_cast<std::size_t>(estimate));
}

auto OmniVoiceImpl::synthesize(std::string_view text, const inference::SynthesisOptions& options)
    -> Result<inference::Synthesis> {
    try {
        if (!impl_->tokenizer)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "OmniVoice was not loaded as a synthesis model"});
        const auto cleaned = clean_text(text);
        if (cleaned.empty()) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "TTS text must not be empty"});
        if (!options.steps || options.steps > 128 || !std::isfinite(options.guidance_scale) ||
            !std::isfinite(options.time_shift) || options.time_shift <= 0 || !std::isfinite(options.layer_penalty) ||
            !std::isfinite(options.position_temperature) || options.position_temperature < 0)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid OmniVoice generation options"});

        const auto target = target_length(cleaned, options);
        const auto language = options.language.empty() ? "None" : options.language;
        auto instruction = require(voice_instruction(options.voice, cleaned, language));
        if (instruction.empty()) instruction = "None";
        auto prompt = require(impl_->tokenizer->encode_template(
            "synthesis", {{"language", language}, {"instruction", instruction}, {"text", cleaned}}));
        if (prompt.empty())
            throw ops::Failure({ErrorCode::RUNTIME, "OmniVoice prompt tokenization returned no tokens"});

        std::vector<std::int32_t> tokens(static_cast<std::size_t>(impl_->codebooks) * target, impl_->mask_id);
        std::vector<std::uint8_t> conditional_mask(prompt.size() + target, 0);
        std::ranges::fill(conditional_mask.begin() + prompt.size(), conditional_mask.end(), 1);
        std::vector<std::uint8_t> unconditional_mask(target, 1);

        std::vector<float> timesteps(options.steps + 1);
        for (std::size_t step = 0; step <= options.steps; ++step) {
            const auto time = static_cast<float>(step) / static_cast<float>(options.steps);
            timesteps[step] = options.time_shift * time / (1.F + (options.time_shift - 1.F) * time);
        }
        std::vector<std::size_t> schedule(options.steps);
        auto remaining = tokens.size();
        for (std::size_t step = 0; step < options.steps; ++step) {
            const auto count =
                step + 1 == options.steps
                    ? remaining
                    : std::min<std::size_t>(
                          remaining,
                          static_cast<std::size_t>(std::ceil(tokens.size() * (timesteps[step + 1] - timesteps[step]))));
            schedule[step] = count;
            remaining -= count;
        }

        std::mt19937_64 random(options.seed);
        std::uniform_real_distribution<float> uniform(std::nextafter(0.F, 1.F), 1.F);
        const auto generation_start = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < options.steps; ++step) {
            const auto conditional = prepare_input(prompt, tokens, impl_->codebooks);
            auto conditional_logits = require(forward(conditional, conditional_mask, prompt.size(), target));
            auto conditional_values = require(conditional_logits.data<float>());
            std::vector<float> conditional_copy(conditional_values.begin(), conditional_values.end());

            const auto unconditional = prepare_input({}, tokens, impl_->codebooks);
            auto unconditional_logits = require(forward(unconditional, unconditional_mask, 0, target));
            const auto unconditional_values = require(unconditional_logits.data<float>());
            if (conditional_copy.size() != unconditional_values.size())
                throw ops::Failure({ErrorCode::RUNTIME, "conditional and unconditional logits disagree"});

            std::vector<Candidate> candidates;
            candidates.reserve(tokens.size());
            std::vector<float> guided(static_cast<std::size_t>(impl_->audio_vocabulary));
            for (std::size_t position = 0; position < target; ++position)
                for (std::int32_t codebook = 0; codebook < impl_->codebooks; ++codebook) {
                    const auto token_index = static_cast<std::size_t>(codebook) * target + position;
                    if (tokens[token_index] != impl_->mask_id) continue;
                    const auto logits_offset =
                        (position * static_cast<std::size_t>(impl_->codebooks) + codebook) * impl_->audio_vocabulary;
                    const auto conditional_distribution =
                        std::span<const float>(conditional_copy).subspan(logits_offset, impl_->audio_vocabulary);
                    const auto unconditional_distribution =
                        unconditional_values.subspan(logits_offset, impl_->audio_vocabulary);
                    const auto conditional_norm = log_sum_exp(conditional_distribution);
                    const auto unconditional_norm = log_sum_exp(unconditional_distribution);
                    for (std::int32_t token = 0; token < impl_->audio_vocabulary; ++token) {
                        const auto conditioned = conditional_distribution[token] - conditional_norm;
                        const auto unconditioned = unconditional_distribution[token] - unconditional_norm;
                        guided[token] = conditioned + options.guidance_scale * (conditioned - unconditioned);
                    }
                    const auto guided_norm = log_sum_exp(guided);
                    auto best = -std::numeric_limits<float>::infinity();
                    std::int32_t predicted = -1;
                    for (std::int32_t token = 0; token < impl_->audio_vocabulary; ++token)
                        if (token != impl_->mask_id && guided[token] > best) {
                            best = guided[token];
                            predicted = token;
                        }
                    auto score = best - guided_norm - codebook * options.layer_penalty;
                    if (options.position_temperature > 0) {
                        const auto sample = uniform(random);
                        const auto gumbel = -std::log(-std::log(sample + 1e-10F) + 1e-10F);
                        score = score / options.position_temperature + gumbel;
                    }
                    candidates.push_back({score, token_index, predicted});
                }
            const auto count = std::min(schedule[step], candidates.size());
            std::ranges::partial_sort(candidates, candidates.begin() + count, std::greater{}, &Candidate::score);
            for (const auto& candidate : std::span(candidates).first(count)) tokens[candidate.index] = candidate.token;
            if (options.on_progress) options.on_progress(step + 1, options.steps);
        }
        const auto generation_ns = inference::synthesis_elapsed_ns(generation_start);
        if (std::ranges::find(tokens, impl_->mask_id) != tokens.end())
            throw ops::Failure({ErrorCode::RUNTIME, "OmniVoice generation left masked audio tokens"});

        const auto decode_start = std::chrono::steady_clock::now();
        auto samples = require(decode(tokens, target));
        samples = postprocess(std::move(samples), sample_rate());
        return inference::Synthesis{
            .samples = std::move(samples),
            .sample_rate = sample_rate(),
            .tokens = std::move(tokens),
            .stats =
                {
                    .generation_ns = generation_ns,
                    .decode_ns = inference::synthesis_elapsed_ns(decode_start),
                    .preparation_ns = preparation_ns(),
                    .audio_tokens = target,
                },
        };
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto OmniVoiceImpl::text_vocabulary_size() const noexcept -> std::int32_t { return impl_->llm->vocabulary_size(); }
auto OmniVoiceImpl::audio_vocabulary_size() const noexcept -> std::int32_t { return impl_->audio_vocabulary; }
auto OmniVoiceImpl::audio_mask_id() const noexcept -> std::int32_t { return impl_->mask_id; }
auto OmniVoiceImpl::codebook_count() const noexcept -> std::int32_t { return impl_->codebooks; }
auto OmniVoiceImpl::sample_rate() const noexcept -> std::uint32_t { return impl_->sample_rate; }
auto OmniVoiceImpl::frame_rate() const noexcept -> std::uint32_t { return impl_->frame_rate; }
auto OmniVoiceImpl::preparation_ns() const noexcept -> std::uint64_t { return impl_->context.preparation_ns(); }

} // namespace kidi::model
