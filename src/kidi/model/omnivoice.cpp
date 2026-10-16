#include "kidi/model/omnivoice.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
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

constexpr std::size_t BATCH_GUIDANCE_MIN_TOKENS = 128;

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

auto cpu_decoder(const YAML::Node& config) -> layers::HiggsDecoder {
    const ModuleScope scope(tensor::Device::cpu());
    return layers::HiggsDecoder(decoder_shape(config));
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

auto generator_replay(tensor::Device device) -> std::optional<bool> {
#if defined(__ANDROID__)
    if ((device == tensor::Device::cpu() || device == tensor::Device::vulkan()) && !std::getenv("KIDI_REPLAY"))
        return false;
#endif
    return std::nullopt;
}

} // namespace

struct OmniVoiceImpl::State {
    ops::Context context, decoder_context;
    std::int32_t audio_vocabulary, mask_id, codebooks;
    std::uint32_t sample_rate, frame_rate;
    Qwen3 llm;
    layers::TokenEmbedding audio_embeddings;
    layers::Linear audio_heads;
    layers::HiggsDecoder audio_tokenizer;
    YAML::Node package;
    std::optional<text::Tokenizer> tokenizer;

    State(const YAML::Node& config, tensor::Device device)
        : context(device, false, generator_replay(device)),
          decoder_context(tensor::Device::cpu(), false, generator_replay(tensor::Device::cpu())),
          audio_vocabulary(config["audio_vocab_size"].as<std::int32_t>()),
          mask_id(config["audio_mask_id"].as<std::int32_t>()),
          codebooks(config["num_audio_codebook"].as<std::int32_t>()),
          sample_rate(config["codec_config"]["sample_rate"].as<std::uint32_t>()),
          frame_rate(config["codec_config"]["frame_rate"].as<std::uint32_t>()),
          llm(config["llm_config"]),
          audio_embeddings(codebooks * audio_vocabulary, llm->hidden_size(), 1.F),
          audio_heads(llm->hidden_size(), codebooks * audio_vocabulary, true, false),
          audio_tokenizer(cpu_decoder(config)) {}

    auto embed_audio(std::span<const std::int32_t> tokens, std::size_t length) -> tensor::Tensor {
        if (!length || tokens.size() != static_cast<std::size_t>(codebooks) * length)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid OmniVoice audio token dimensions"});
        tensor::Tensor audio;
        std::vector<std::int32_t> shifted(length);
        for (std::int32_t codebook = 0; codebook < codebooks; ++codebook) {
            for (std::size_t position = 0; position < length; ++position) {
                const auto token = tokens[static_cast<std::size_t>(codebook) * length + position];
                if (token < 0 || token >= audio_vocabulary)
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "audio token is outside the vocabulary"});
                shifted[position] = token + codebook * audio_vocabulary;
            }
            auto embedded = audio_embeddings->forward(context, shifted);
            if (audio.defined())
                context.add_(audio, embedded);
            else
                audio = std::move(embedded);
        }
        return audio;
    }

    auto project(const tensor::Tensor& embeddings, std::size_t target_start, std::size_t target_length)
        -> tensor::Tensor {
        auto hidden = llm->forward(context, embeddings);
        hidden = context.slice(hidden, 1, target_start, target_length);
        return audio_heads->forward(context, hidden);
    }

    auto host_floats(const tensor::Tensor& value, std::vector<float>& storage) -> std::span<const float> {
        if (value.is_host_accessible()) return require(value.data<float>());
        const auto bytes = require(value.copy_to_host());
        const auto values = std::span(reinterpret_cast<const float*>(bytes.data()), value.numel());
        storage.assign(values.begin(), values.end());
        return storage;
    }
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
    if (module_dtype != tensor::DType::I8 ||
        (module_device != tensor::Device::cpu() && module_device != tensor::Device::vulkan() &&
         module_device != tensor::Device::apple_gpu()))
        throw ops::Failure({ErrorCode::UNSUPPORTED, "OmniVoice requires INT8 parameters on CPU, Metal, or Vulkan"});
    impl_ = std::make_unique<State>(config, device());
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
        if (device != tensor::Device::cpu() && device != tensor::Device::vulkan() &&
            device != tensor::Device::apple_gpu())
            throw ops::Failure({ErrorCode::UNSUPPORTED, "OmniVoice supports CPU, Metal, and Vulkan generation"});
        auto tokenizer = require(text::Tokenizer::load(package["tokenizer_file"].as<std::string>(),
                                                       {{"synthesis", package["template_file"].as<std::string>()}}));
        auto weights = require(checkpoint::Weights::load(package["weights_file"].as<std::string>()));
        const ModuleScope construction(tensor::DType::I8, false, device);
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
        if ((!scale && value->dtype() != tensor::DType::I8 && value->dtype() != tensor::DType::U8) ||
            (scale && value->dtype() != tensor::DType::F32))
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
        std::vector<float> text_storage, audio_storage;
        const auto text_values = impl_->host_floats(text, text_storage);
        const auto audio_values = impl_->host_floats(audio, audio_storage);
        const auto width = static_cast<std::size_t>(impl_->llm->hidden_size());
        std::vector<float> values(length * width);
        for (std::size_t position = 0; position < length; ++position) {
            const auto source = audio_mask[position] ? audio_values : text_values;
            std::ranges::copy(source.subspan(position * width, width), values.begin() + position * width);
        }
        auto combined =
            require(tensor::Tensor::from_host({1, static_cast<std::int64_t>(length), impl_->llm->hidden_size()},
                                              std::span<const float>(values), impl_->context.device()));
        return impl_->project(combined, target_start, target_length);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto OmniVoiceImpl::decode(std::span<const std::int32_t> tokens, std::size_t length) -> Result<std::vector<float>> {
    try {
        auto decoded = impl_->audio_tokenizer->forward(impl_->decoder_context, tokens, length);
        impl_->decoder_context.synchronize();
        const auto values = require(decoded.data<float>());
        return std::vector<float>(values.begin(), values.end());
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto OmniVoiceImpl::target_length(std::string_view text, const inference::SynthesisOptions& options,
                                  const inference::SynthesisReference* reference) const -> std::size_t {
    if (options.duration_seconds) {
        if (!std::isfinite(*options.duration_seconds) || *options.duration_seconds <= 0)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "TTS duration must be positive and finite"});
        return std::max<std::size_t>(1, static_cast<std::size_t>(*options.duration_seconds * frame_rate()));
    }
    const auto settings = impl_->package["synthesis"]["auto_duration"];
    const auto reference_text =
        reference ? std::string(reference->text) : settings["reference_text"].as<std::string>("Nice to meet you.");
    const auto reference_tokens =
        reference ? static_cast<float>(reference->tokens.size() / static_cast<std::size_t>(impl_->codebooks))
                  : settings["reference_tokens"].as<float>(25.F);
    const auto estimate = require(audio::RuleDurationEstimator().estimate(text, reference_text, reference_tokens,
                                                                          settings["low_threshold"].as<float>(50.F),
                                                                          settings["boost_strength"].as<float>(3.F)));
    return std::max<std::size_t>(1, static_cast<std::size_t>(estimate));
}

auto OmniVoiceImpl::synthesize(std::string_view text, const inference::SynthesisOptions& options)
    -> Result<inference::Synthesis> {
    return synthesize_(text, options, nullptr);
}

auto OmniVoiceImpl::synthesize_with_reference(std::string_view text, const inference::SynthesisOptions& options,
                                              const inference::SynthesisReference& reference)
    -> Result<inference::Synthesis> {
    return synthesize_(text, options, &reference);
}

auto OmniVoiceImpl::voice_reference_chunk(std::span<const std::string> chunks,
                                          const inference::SynthesisOptions& options) const
    -> Result<std::optional<std::size_t>> {
    try {
        if (chunks.empty()) return std::optional<std::size_t>{};
        const auto minimum = static_cast<std::size_t>(3 * frame_rate());
        std::optional<std::size_t> selected, fallback;
        auto shortest_qualified = std::numeric_limits<std::size_t>::max();
        std::size_t longest = 0;
        for (std::size_t index = 0; index < chunks.size(); ++index) {
            const auto cleaned = clean_text(chunks[index]);
            if (cleaned.empty()) continue;
            const auto length = target_length(cleaned, options);
            if (!fallback || length > longest) {
                fallback = index;
                longest = length;
            }
            if (length >= minimum && length < shortest_qualified) {
                selected = index;
                shortest_qualified = length;
            }
        }
        return selected ? selected : fallback;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto OmniVoiceImpl::synthesize_(std::string_view text, const inference::SynthesisOptions& options,
                                const inference::SynthesisReference* reference) -> Result<inference::Synthesis> {
    try {
        if (!impl_->tokenizer)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "OmniVoice was not loaded as a synthesis model"});
        const auto preparation = preparation_ns();
        const auto cleaned = clean_text(text);
        if (cleaned.empty()) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "TTS text must not be empty"});
        if (!options.steps || options.steps > 128 || !std::isfinite(options.guidance_scale) ||
            !std::isfinite(options.time_shift) || options.time_shift <= 0 || !std::isfinite(options.layer_penalty) ||
            !std::isfinite(options.position_temperature) || options.position_temperature < 0)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid OmniVoice generation options"});

        std::string reference_text;
        std::size_t reference_length = 0;
        if (reference) {
            reference_text = clean_text(reference->text);
            if (reference_text.empty() || reference->tokens.empty() ||
                reference->tokens.size() % static_cast<std::size_t>(impl_->codebooks))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid OmniVoice voice reference"});
            reference_length = reference->tokens.size() / static_cast<std::size_t>(impl_->codebooks);
        }

        const auto target = target_length(cleaned, options, reference);
        const auto language = options.language.empty() ? "None" : options.language;
        auto instruction = require(voice_instruction(options.voice, cleaned, language));
        if (instruction.empty()) instruction = "None";
        auto prompt_text = reference ? reference_text + " " + cleaned : cleaned;
        auto prompt = require(impl_->tokenizer->encode_template(
            "synthesis", {{"language", language}, {"instruction", instruction}, {"text", prompt_text}}));
        if (reference) {
            const auto denoise = impl_->tokenizer->token_id("<|denoise|>");
            if (!denoise)
                throw ops::Failure({ErrorCode::INVALID_MANIFEST, "OmniVoice tokenizer is missing <|denoise|>"});
            prompt.insert(prompt.begin(), *denoise);
        }
        if (prompt.empty())
            throw ops::Failure({ErrorCode::RUNTIME, "OmniVoice prompt tokenization returned no tokens"});

        std::vector<std::int32_t> tokens(static_cast<std::size_t>(impl_->codebooks) * target, impl_->mask_id);
        const auto audio_length = reference_length + target;
        std::vector<std::int32_t> conditional_tokens;
        if (reference) {
            conditional_tokens.resize(static_cast<std::size_t>(impl_->codebooks) * audio_length, impl_->mask_id);
            for (std::int32_t codebook = 0; codebook < impl_->codebooks; ++codebook) {
                const auto source =
                    reference->tokens.subspan(static_cast<std::size_t>(codebook) * reference_length, reference_length);
                std::ranges::copy(source,
                                  conditional_tokens.begin() + static_cast<std::size_t>(codebook) * audio_length);
            }
        }
        const auto prompt_embedding = impl_->llm->embed(impl_->context, prompt);
        const auto conditional_target_start = prompt.size() + reference_length;
        const auto conditional_length = conditional_target_start + target;
        const auto packed_length = conditional_length + target;
        const auto* disable_packing = std::getenv("KIDI_OMNIVOICE_PACKED_GUIDANCE");
        const auto* guidance_layout = std::getenv("KIDI_OMNIVOICE_GUIDANCE_LAYOUT");
        const auto force_batch = guidance_layout && std::string_view(guidance_layout) == "batch";
        const auto force_packed = guidance_layout && std::string_view(guidance_layout) == "packed";
        const auto guidance = options.guidance_scale != 0.F;
        const auto automatic_batch =
            impl_->context.device() != tensor::Device::cpu() && target >= BATCH_GUIDANCE_MIN_TOKENS;
        const auto batch_guidance = guidance &&
                                    conditional_length <= static_cast<std::size_t>(impl_->llm->maximum_position()) &&
                                    (force_batch || (!force_packed && automatic_batch));
        const auto packed_guidance = guidance && !batch_guidance &&
                                     packed_length <= static_cast<std::size_t>(impl_->llm->maximum_position()) &&
                                     !(disable_packing && std::string_view(disable_packing) == "0");
        tensor::Tensor guidance_mask;
        tensor::Tensor unconditional_padding;
        if (batch_guidance) {
            std::vector<float> values(2 * conditional_length * conditional_length, 0.F);
            auto* unconditional = values.data() + conditional_length * conditional_length;
            std::fill(unconditional, unconditional + conditional_length * conditional_length, -1.0e9F);
            for (std::size_t row = 0; row < target; ++row)
                std::fill_n(unconditional + row * conditional_length, target, 0.F);
            for (std::size_t row = target; row < conditional_length; ++row)
                unconditional[row * conditional_length + row] = 0.F;
            guidance_mask = require(tensor::Tensor::from_host(
                {2, 1, static_cast<std::int64_t>(conditional_length), static_cast<std::int64_t>(conditional_length)},
                std::span<const float>(values), impl_->context.device()));
            unconditional_padding = require(tensor::Tensor::zeros(
                {1, static_cast<std::int64_t>(conditional_length - target), impl_->llm->hidden_size()},
                tensor::DType::F32, impl_->context.device()));
        }

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
        std::vector<float> conditional_copy;
        std::vector<float> logits_host;
        std::vector<float> guided(static_cast<std::size_t>(impl_->audio_vocabulary));
        std::vector<Candidate> candidates;
        candidates.reserve(tokens.size());
        const auto generation_start = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < options.steps; ++step) {
            const auto target_audio_embedding = impl_->embed_audio(tokens, target);
            const auto conditional_audio_embedding =
                reference ? impl_->embed_audio(conditional_tokens, audio_length) : target_audio_embedding;
            const std::array conditional_parts{prompt_embedding, conditional_audio_embedding};
            const auto conditional_embedding = impl_->context.concat(conditional_parts, 1);
            tensor::Tensor logits;
            std::span<const float> conditional_values, unconditional_values;
            if (!guidance) {
                auto hidden = impl_->llm->forward(impl_->context, conditional_embedding);
                hidden = impl_->context.slice(hidden, 1, conditional_target_start, target);
                logits = impl_->audio_heads->forward(impl_->context, hidden);
                impl_->context.synchronize();
                conditional_values = impl_->host_floats(logits, logits_host);
                unconditional_values = conditional_values;
            } else if (batch_guidance) {
                const std::array unconditional_parts{target_audio_embedding, unconditional_padding};
                const auto unconditional_embedding = impl_->context.concat(unconditional_parts, 1);
                const std::array guidance_parts{conditional_embedding, unconditional_embedding};
                const auto guidance_embedding = impl_->context.concat(guidance_parts, 0);
                auto hidden = impl_->llm->forward(impl_->context, guidance_embedding, guidance_mask);
                auto conditional_hidden = impl_->context.slice(hidden, 0, 0, 1);
                conditional_hidden = impl_->context.slice(conditional_hidden, 1, conditional_target_start, target);
                auto unconditional_hidden = impl_->context.slice(hidden, 0, 1, 1);
                unconditional_hidden = impl_->context.slice(unconditional_hidden, 1, 0, target);
                const std::array hidden_parts{conditional_hidden, unconditional_hidden};
                hidden = impl_->context.concat(hidden_parts, 1);
                logits = impl_->audio_heads->forward(impl_->context, hidden);
                impl_->context.synchronize();
                const auto values = impl_->host_floats(logits, logits_host);
                const auto distribution_size = values.size() / 2;
                conditional_values = values.first(distribution_size);
                unconditional_values = values.last(distribution_size);
            } else if (packed_guidance) {
                const std::array guidance_parts{conditional_embedding, target_audio_embedding};
                const auto guidance_embedding = impl_->context.concat(guidance_parts, 1);
                const std::array segments{conditional_length, target};
                auto hidden = impl_->llm->forward_segmented(impl_->context, guidance_embedding, segments);
                hidden = impl_->context.slice(hidden, 1, conditional_target_start, 2 * target);
                logits = impl_->audio_heads->forward(impl_->context, hidden);
                impl_->context.synchronize();
                const auto values = impl_->host_floats(logits, logits_host);
                const auto distribution_size = values.size() / 2;
                conditional_values = values.first(distribution_size);
                unconditional_values = values.last(distribution_size);
            } else {
                auto conditional_logits = impl_->project(conditional_embedding, conditional_target_start, target);
                impl_->context.synchronize();
                const auto values = impl_->host_floats(conditional_logits, logits_host);
                conditional_copy.assign(values.begin(), values.end());
                logits = impl_->project(target_audio_embedding, 0, target);
                impl_->context.synchronize();
                conditional_values = conditional_copy;
                unconditional_values = impl_->host_floats(logits, logits_host);
            }
            if (conditional_values.size() != unconditional_values.size())
                throw ops::Failure({ErrorCode::RUNTIME, "conditional and unconditional logits disagree"});

            candidates.clear();
            for (std::size_t position = 0; position < target; ++position)
                for (std::int32_t codebook = 0; codebook < impl_->codebooks; ++codebook) {
                    const auto token_index = static_cast<std::size_t>(codebook) * target + position;
                    if (tokens[token_index] != impl_->mask_id) continue;
                    const auto logits_offset =
                        (position * static_cast<std::size_t>(impl_->codebooks) + codebook) * impl_->audio_vocabulary;
                    const auto conditional_distribution =
                        conditional_values.subspan(logits_offset, impl_->audio_vocabulary);
                    const auto unconditional_distribution =
                        unconditional_values.subspan(logits_offset, impl_->audio_vocabulary);
                    for (std::int32_t token = 0; token < impl_->audio_vocabulary; ++token) {
                        const auto conditioned = conditional_distribution[token];
                        guided[token] =
                            conditioned + options.guidance_scale * (conditioned - unconditional_distribution[token]);
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
            for (const auto& candidate : std::span(candidates).first(count)) {
                tokens[candidate.index] = candidate.token;
                if (reference) {
                    const auto codebook = candidate.index / target;
                    const auto position = candidate.index % target;
                    conditional_tokens[codebook * audio_length + reference_length + position] = candidate.token;
                }
            }
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
                    .preparation_ns = preparation_ns() - preparation,
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
auto OmniVoiceImpl::preparation_ns() const noexcept -> std::uint64_t {
    return impl_->context.preparation_ns() + impl_->decoder_context.preparation_ns();
}
auto OmniVoiceImpl::execution() const -> std::string {
    if (const auto accelerator = impl_->context.accelerator(); !accelerator.empty())
        return "cpu+" + std::string(accelerator);
    if (impl_->context.device() == tensor::Device::apple_gpu()) return "mps+ynnpack";
    return impl_->context.device() == tensor::Device::vulkan() ? "vulkan+ynnpack" : "ynnpack";
}
auto OmniVoiceImpl::release_workspaces() -> void {
    impl_->context.release_workspaces();
    impl_->decoder_context.release_workspaces();
    impl_->llm->clear_attention();
}

} // namespace kidi::model
