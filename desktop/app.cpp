#include "kidi/core/precision.h"

#include "kidi/checkpoint/hub.h"
#include "ebook.h"
#include "kidi/inference/generator.h"
#include "kidi/inference/synthesizer.h"
#include "kidi/inference/transcriber.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/text/tokenizer.h"
#include "liteapp/app/desktop.h"
#include "liteapp/core/activity_gate.h"
#include "liteapp/core/random.h"
#include "liteapp/http/server.h"
#include "liteapp/service/router.h"
#include "liteapp/vfs/filesystem.h"

#include <kidi/desktop_assets.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using Json = nlohmann::json;
using kidi::inference::GenerationOptions;
using kidi::inference::Generator;
using kidi::text::ChatMessage;
using Clock = std::chrono::steady_clock;

auto json_response(Json value, int status = 200) -> liteapp::Response {
    return liteapp::Response::json(value.dump(), status);
}

auto valid_language_setting(std::string_view language) -> bool {
    if (language == "auto") return true;
    std::size_t code = 0;
    for (const auto value : language) {
        if (value == ',') {
            if (code < 2 || code > 3) return false;
            code = 0;
        } else if (value >= 'a' && value <= 'z')
            ++code;
        else
            return false;
    }
    return code >= 2 && code <= 3;
}

auto milliseconds(Clock::time_point started) -> double {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

auto messages_from(const Json& source, const std::map<std::uint32_t, std::filesystem::path>& images)
    -> kidi::Result<std::vector<ChatMessage>> {
    const auto invalid = [](std::string message) {
        return std::unexpected(kidi::Error{kidi::ErrorCode::INVALID_ARGUMENT, std::move(message)});
    };
    if (!source.is_array() || source.empty() || source.size() > 128)
        return invalid("messages must be a nonempty array with at most 128 entries");
    std::vector<ChatMessage> result;
    result.reserve(source.size());
    for (const auto& message : source) {
        if (!message.is_object() || message.size() < 2 || message.size() > 3 || !message.contains("role") ||
            !message["role"].is_string() || !message.contains("content") || !message["content"].is_string())
            return invalid("each message must contain role, content, and optional images fields only");
        const auto role = message["role"].get<std::string>();
        if (role != "user" && role != "assistant") return invalid("message role must be user or assistant");
        auto content = message["content"].get<std::string>();
        std::vector<std::filesystem::path> paths;
        if (message.contains("images")) {
            if (!message["images"].is_array() || message["images"].size() > 8 ||
                (role != "user" && !message["images"].empty()))
                return invalid("images require a user message with at most eight entries");
            for (const auto& value : message["images"]) {
                if (!value.is_number_unsigned()) return invalid("image IDs must be unsigned integers");
                const auto found = images.find(value.get<std::uint32_t>());
                if (found == images.end()) return invalid("message references an unavailable image");
                paths.push_back(found->second);
            }
        }
        if (content.empty() && paths.empty()) return invalid("message content or images must not be empty");
        result.push_back({.role = role, .content = std::move(content), .images = std::move(paths)});
    }
    return result;
}

class StreamingAudioJoiner {
public:
    using Emit = std::function<void(std::span<const float>, std::uint32_t)>;

    explicit StreamingAudioJoiner(Emit emit) : emit_(std::move(emit)) {}

    void push(std::span<const float> samples, std::uint32_t sample_rate) {
        if (!sample_rate || samples.empty()) return;
        if (sample_rate_ && sample_rate_ != sample_rate)
            throw kidi::ops::Failure({kidi::ErrorCode::RUNTIME, "TTS segments use different sample rates"});
        sample_rate_ = sample_rate;
        const auto padding = std::min<std::size_t>(sample_rate / 10, samples.size() / 2);
        auto body = samples.subspan(padding, samples.size() - 2 * padding);
        constexpr float SILENCE = 0.001F;
        const auto search = std::min<std::size_t>(sample_rate / 4, body.size() / 2);
        const auto margin = static_cast<std::size_t>(sample_rate / 100);
        std::size_t active_begin = 0;
        while (active_begin < search && std::abs(body[active_begin]) < SILENCE) ++active_begin;
        std::size_t active_end = body.size();
        while (active_end > body.size() - search && std::abs(body[active_end - 1]) < SILENCE) --active_end;
        if (active_begin < active_end) {
            const auto begin = active_begin > margin ? active_begin - margin : 0;
            const auto end = std::min(body.size(), active_end + margin);
            body = body.subspan(begin, end - begin);
        }
        if (!tail_.empty()) {
            const auto overlap = std::min(tail_.size(), body.size());
            std::vector<float> mixed(overlap);
            for (std::size_t index = 0; index < overlap; ++index)
                mixed[index] = std::clamp(tail_[index] + body[index], -1.F, 1.F);
            emit_(mixed, sample_rate);
            tail_.clear();
            body = body.subspan(overlap);
        }
        const auto overlap = std::min<std::size_t>(sample_rate / 10, body.size());
        if (body.size() > overlap) emit_(body.first(body.size() - overlap), sample_rate);
        const auto trailing = body.last(overlap);
        tail_.assign(trailing.begin(), trailing.end());
    }

    void finish() {
        if (!tail_.empty()) emit_(tail_, sample_rate_);
        tail_.clear();
    }

private:
    Emit emit_;
    std::uint32_t sample_rate_ = 0;
    std::vector<float> tail_;
};

struct ModelConfig {
    bool enabled = false;
    std::string model_id;
    std::string revision{"main"};
    std::filesystem::path directory;
    std::string accelerator{"auto"};
    std::size_t metal_prefill_cache_bytes = 0;
};

struct TranslationConfig {
    std::size_t target_chars = 800;
    std::size_t max_chars = 1200;
    std::size_t max_sentence_chars = 600;
    std::size_t max_input_tokens = 1536;
};

struct AppConfig {
    std::filesystem::path source;
    std::string model_hub_remote{"https://huggingface.co"};
    std::filesystem::path model_hub_local;
    std::size_t threads = 4;
    bool preload_speech = true;
    std::size_t idle_model_timeout_seconds = 300;
    ModelConfig chat;
    ModelConfig asr;
    ModelConfig tts;
    kidi::core::InferencePrecision precision = kidi::core::InferencePrecision::CHECKPOINT;
    kidi::core::KVCachePrecision kv_cache_precision = kidi::core::KVCachePrecision::AUTO;
    std::size_t maximum_tokens = 512;
    std::size_t context_tokens = 8192;
    std::size_t prefill_chunk_tokens = 128;
    std::size_t prefix_cache_bytes = 512U * 1024U * 1024U;
    std::size_t asr_partial_interval_ms = 800;
    std::size_t max_audio_queue_ms = 3000;
    std::size_t max_text_frame_bytes = 1024U * 1024U;
    std::string system_prompt;
    std::string translation_prompt;
    TranslationConfig translation;
    Json asr_settings;
    Json tts_settings;

    auto components() const -> Json {
        return {
            {"chat",
             {
                 {"enabled", chat.enabled},
                 {"model_id", chat.model_id},
                 {"revision", chat.revision},
                 {"directory", chat.directory.string()},
                 {"accelerator", chat.accelerator},
                 {"metal_prefill_cache_mb", chat.metal_prefill_cache_bytes / (1024U * 1024U)},
                 {"precision", kidi::core::to_string(precision)},
                 {"kv_cache_precision", kidi::core::to_string(kv_cache_precision)},
                 {"system_prompt", system_prompt},
                 {"translation_prompt", translation_prompt},
                 {"translation",
                  {
                      {"target_chars", translation.target_chars},
                      {"max_chars", translation.max_chars},
                      {"max_sentence_chars", translation.max_sentence_chars},
                      {"max_input_tokens", translation.max_input_tokens},
                  }},
                 {"maximum_tokens", maximum_tokens},
                 {"context_tokens", context_tokens},
                 {"prefill_chunk_tokens", prefill_chunk_tokens},
                 {"prefix_cache_mb", prefix_cache_bytes / (1024U * 1024U)},
             }},
            {"asr",
             {
                 {"enabled", asr.enabled},
                 {"model_id", asr.model_id},
                 {"revision", asr.revision},
                 {"directory", asr.directory.string()},
                 {"accelerator", asr.accelerator},
                 {"settings", asr_settings},
             }},
            {"tts",
             {
                 {"enabled", tts.enabled},
                 {"model_id", tts.model_id},
                 {"revision", tts.revision},
                 {"directory", tts.directory.string()},
                 {"accelerator", tts.accelerator},
                 {"metal_prefill_cache_mb", tts.metal_prefill_cache_bytes / (1024U * 1024U)},
                 {"settings", tts_settings},
             }},
        };
    }
};

auto invalid_config(std::string message) -> kidi::Result<AppConfig> {
    return std::unexpected(kidi::Error{kidi::ErrorCode::INVALID_ARGUMENT, std::move(message)});
}

auto yaml_string(const YAML::Node& node, std::string_view key, std::string fallback = {}) -> std::string {
    if (!node || !node.IsMap()) return fallback;
    const auto value = node[std::string(key)];
    return value ? value.as<std::string>() : std::move(fallback);
}

auto resolve_model_path(const std::filesystem::path& config, const YAML::Node& node) -> std::filesystem::path {
    auto path = std::filesystem::path(yaml_string(node, "directory"));
    if (path.empty()) return {};
    if (path.is_relative()) path = config.parent_path() / path;
    return path.lexically_normal();
}

auto positive_size(const YAML::Node& node, std::string_view key, std::size_t fallback, std::size_t maximum)
    -> std::size_t;

auto model_config(const std::filesystem::path& source, const YAML::Node& node, bool default_enabled) -> ModelConfig {
    if (!node || !node.IsMap()) return {.enabled = default_enabled};
    return {
        .enabled = node["enabled"] ? node["enabled"].as<bool>() : default_enabled,
        .model_id = yaml_string(node, "model_id"),
        .revision = yaml_string(node, "revision", "main"),
        .directory = resolve_model_path(source, node),
        .accelerator = yaml_string(node, "accelerator", "auto"),
        .metal_prefill_cache_bytes = positive_size(node, "metal_prefill_cache_mb", 512, 4096) * 1024U * 1024U,
    };
}

auto positive_size(const YAML::Node& node, std::string_view key, std::size_t fallback, std::size_t maximum)
    -> std::size_t {
    if (!node || !node.IsMap()) return fallback;
    const auto value = node[std::string(key)] ? node[std::string(key)].as<std::size_t>() : fallback;
    if (!value || value > maximum) throw YAML::BadConversion(node.Mark());
    return value;
}

void configure_metal_prefill_cache(std::size_t bytes) {
    const auto value = std::to_string(bytes);
#if defined(_WIN32)
    _putenv_s("KIDI_PREFILL_CACHE_BYTES", value.c_str());
#else
    setenv("KIDI_PREFILL_CACHE_BYTES", value.c_str(), 1);
#endif
}

auto application_config(const std::filesystem::path& requested) -> kidi::Result<AppConfig> {
    try {
        std::error_code error;
        auto source = std::filesystem::absolute(requested, error);
        if (error || !std::filesystem::is_regular_file(source, error))
            return invalid_config("Configuration file does not exist");
        source = source.lexically_normal();
        const auto root = YAML::LoadFile(source.string());
        if (!root.IsMap() || !root["version"] || root["version"].as<int>() != 1)
            return invalid_config("Configuration version must be 1");
        const auto models = root["models"];
        if (!models.IsMap() || !models["chat"].IsMap()) return invalid_config("models.chat configuration is required");
        AppConfig result;
        result.source = source;
        const auto hub = root["model_hub"];
        const kidi::checkpoint::Hub model_hub{yaml_string(hub, "remote", "https://huggingface.co"),
                                              yaml_string(hub, "local", "~/.cache/kidi/model-hub")};
        result.model_hub_remote = model_hub.remote();
        result.model_hub_local = model_hub.local();
        result.threads = positive_size(root["runtime"], "threads", 4, 64);
        result.preload_speech =
            root["runtime"] && root["runtime"]["preload_speech"] ? root["runtime"]["preload_speech"].as<bool>() : true;
        result.idle_model_timeout_seconds = positive_size(root["runtime"], "idle_model_timeout_seconds", 300, 86400);
        result.chat = model_config(source, models["chat"], true);
        result.asr = model_config(source, models["asr"], false);
        result.tts = model_config(source, models["tts"], false);
        for (auto* model : {&result.chat, &result.asr, &result.tts}) {
            if (!model->enabled || !model->directory.empty()) continue;
            auto resolved = model_hub.resolve(model->model_id, model->revision);
            if (!resolved) return invalid_config(resolved.error().message);
            model->directory = std::move(*resolved);
        }
        if (!result.chat.enabled) return invalid_config("models.chat must be enabled");
        if (result.chat.directory.empty()) return invalid_config("Chat model is not present in model_hub.local");
        for (const auto* model : {&result.chat, &result.asr, &result.tts}) {
            if (model->enabled && model->directory.empty())
                return invalid_config("Enabled model is not present in model_hub.local: " + model->model_id);
            if (model->accelerator != "auto" && model->accelerator != "cpu" && model->accelerator != "gpu" &&
                model->accelerator != "npu")
                return invalid_config("Model accelerator must be auto, cpu, gpu, or npu");
            if (model->enabled && !std::filesystem::is_directory(model->directory, error))
                return invalid_config("Enabled model directory does not exist: " + model->directory.string());
        }
        if (result.tts.enabled) {
            for (const auto* file :
                 {"model.yaml", "model.safetensors", "sentence-prefixes.tsv", "synthesis_template.jinja", "NOTICE"}) {
                if (!std::filesystem::is_regular_file(result.tts.directory / file, error))
                    return invalid_config("TTS package is missing required file: " + std::string(file));
            }
            if (!std::filesystem::is_regular_file(result.tts.directory / "tokenizer.json", error) &&
                !std::filesystem::is_regular_file(result.tts.directory / "tokenizer.json.gz", error))
                return invalid_config("TTS package is missing tokenizer.json or tokenizer.json.gz");
        }
        const auto precision_name = yaml_string(models["chat"], "precision", "checkpoint");
        const auto precision = kidi::core::parse_precision(precision_name);
        if (!precision) return invalid_config("Unknown chat precision: " + precision_name);
        result.precision = *precision;
        const auto cache_name = yaml_string(models["chat"], "kv_cache_precision", "auto");
        const auto cache_precision = kidi::core::parse_kv_cache_precision(cache_name);
        if (!cache_precision) return invalid_config("Unknown chat KV-cache precision: " + cache_name);
        result.kv_cache_precision = *cache_precision;
        result.system_prompt = yaml_string(models["chat"], "system_prompt");
        result.translation_prompt = yaml_string(
            models["chat"], "translation_prompt",
            "You are a literary translator. Translate <text_to_translate> into {target_language}. Use the "
            "surrounding context only to resolve meaning and return only the translated text.");
        const auto translation = models["chat"]["translation"];
        result.translation = {
            .target_chars = positive_size(translation, "target_chars", 800, 16384),
            .max_chars = positive_size(translation, "max_chars", 1200, 16384),
            .max_sentence_chars = positive_size(translation, "max_sentence_chars", 600, 16384),
            .max_input_tokens = positive_size(translation, "max_input_tokens", 1536, 8192),
        };
        if (result.translation.target_chars > result.translation.max_chars ||
            result.translation.max_sentence_chars > result.translation.max_chars)
            return invalid_config("Reader translation limits are inconsistent");
        const auto generation = models["chat"]["generation"];
        result.maximum_tokens = positive_size(generation, "max_tokens", 512, 2048);
        result.context_tokens = positive_size(generation, "context_tokens", 8192, 32768);
        result.prefill_chunk_tokens = positive_size(generation, "prefill_chunk_tokens", 128, 2048);
        const auto prefix_mb = positive_size(generation, "prefix_cache_mb", 512, 4096);
        result.prefix_cache_bytes = prefix_mb * 1024U * 1024U;
        const auto translation_context =
            (result.translation.max_input_tokens + result.maximum_tokens + 127) / 128 * 128;
        if (translation_context > result.context_tokens)
            return invalid_config("Reader translation token limit exceeds the shared chat context-token budget");
        const auto asr = models["asr"];
        result.asr_partial_interval_ms = positive_size(asr, "partial_interval_ms", 800, 5000);
        const auto vad = asr["vad"];
        const auto vad_threshold = vad && vad["threshold"] ? vad["threshold"].as<float>() : 0.012F;
        if (!std::isfinite(vad_threshold) || vad_threshold <= 0.F || vad_threshold > 1.F)
            return invalid_config("ASR VAD threshold must be between 0 and 1");
        result.asr_settings = {
            {"language", yaml_string(asr, "language", "auto")},
            {"max_tokens", positive_size(asr, "max_tokens", 128, 2048)},
            {"vad",
             {{"threshold", vad_threshold},
              {"minimum_speech_ms", positive_size(vad, "minimum_speech_ms", 250, 10000)},
              {"silence_ms", positive_size(vad, "silence_ms", 800, 10000)}}},
        };
        result.max_audio_queue_ms = positive_size(root["transport"], "max_audio_queue_ms", 3000, 30000);
        result.max_text_frame_bytes =
            positive_size(root["transport"], "max_text_frame_bytes", 1024U * 1024U, 8U * 1024U * 1024U);
        const auto tts = models["tts"];
        const auto pace = tts["pace"] ? tts["pace"].as<float>() : 1.F;
        if (!std::isfinite(pace) || pace < 0.5F || pace > 2.F)
            return invalid_config("TTS pace must be between 0.5 and 2");
        const auto voice = tts["voice"];
        result.tts_settings = {
            {"language", yaml_string(tts, "language", "English")},
            {"pace", pace},
            {"reference_language", yaml_string(tts, "reference_language", "English")},
            {"reference_text", yaml_string(tts, "reference_text", "Nice to meet you.")},
            {"steps", positive_size(tts, "steps", 10, 128)},
            {"realtime_steps", std::max<std::size_t>(10, positive_size(tts, "realtime_steps", 10, 128))},
            {"voice",
             {
                 {"gender", yaml_string(voice, "gender", "female")},
                 {"age", yaml_string(voice, "age", "young adult")},
                 {"pitch", yaml_string(voice, "pitch", "moderate")},
                 {"style", yaml_string(voice, "style")},
                 {"accent", yaml_string(voice, "accent")},
             }},
        };
        return result;
    } catch (const YAML::Exception& error) {
        return invalid_config(std::string{"Invalid YAML configuration: "} + error.what());
    }
}

struct ChatRunResult {
    std::string text;
    Json stats;
};

struct TtsRunResult {
    kidi::inference::Synthesis synthesis;
    bool voice_prompt_reused = false;
};

class AppBackend {
public:
    auto load_config(const std::filesystem::path& source) -> liteapp::Result<void> {
        auto parsed = application_config(source);
        if (!parsed) return std::unexpected(liteapp::Error{"invalid_config", parsed.error().message});
        const auto selected = kidi::inference::select_device(parsed->chat.accelerator);
        if (!selected) return std::unexpected(liteapp::Error{"unavailable", selected.error().message});

        std::scoped_lock lock(chat_mutex_);
        configure_metal_prefill_cache(parsed->chat.metal_prefill_cache_bytes);
        kidi::runtime::ynn::set_thread_count(parsed->threads);
        if (auto reserved = kidi::runtime::ynn::reserve_thread_pool(parsed->threads); !reserved)
            return std::unexpected(liteapp::Error{"load_failed", reserved.error().message});
        auto loaded = Generator::load(parsed->chat.directory, {
                                                                  .device = *selected,
                                                                  .precision = parsed->precision,
                                                                  .kv_cache_precision = parsed->kv_cache_precision,
                                                                  .packed_prefill = true,
                                                              });
        if (!loaded) return std::unexpected(liteapp::Error{"load_failed", loaded.error().message});
        auto configured = loaded->configure_serving({
            .maximum_active = 1,
            .maximum_requests = 1,
            .cache_token_budget = parsed->context_tokens,
            .prefill_tokens_per_step = parsed->prefill_chunk_tokens,
            .compact_cache = true,
        });
        if (!configured) return std::unexpected(liteapp::Error{"load_failed", configured.error().message});
        generator_.emplace(std::move(*loaded));
        if (auto warmed = generator_->warm_up(); !warmed)
            return std::unexpected(liteapp::Error{"load_failed", warmed.error().message});
        std::string speech_execution;
        if (parsed->tts.enabled) {
            const auto selected_speech = kidi::inference::select_device(parsed->tts.accelerator);
            if (!selected_speech)
                return std::unexpected(liteapp::Error{"unavailable", selected_speech.error().message});
            speech_execution = std::string(kidi::tensor::to_string(selected_speech->kind));
        }
        config_.emplace(std::move(*parsed));
        speech_execution_ = std::move(speech_execution);
        {
            std::scoped_lock prompt_lock(prompt_mutex_);
            system_prompt_ = config_->system_prompt;
        }
        configure_metal_prefill_cache(config_->tts.metal_prefill_cache_bytes);
        model_path_ = config_->chat.directory.string();
        execution_ = generator_->execution();
        vision_supported_ = generator_->vision_supported();
        return {};
    }

    void start_background_warmup() {
        if (!config_) return;
        if (config_->preload_speech && !warmup_.joinable()) {
            warmup_ = std::jthread([this] {
                if (config_->asr.enabled)
                    if (auto warmed = warm_asr(); !warmed)
                        std::cerr << "ASR background warm-up failed: " << warmed.error().message << '\n';
                if (config_->tts.enabled)
                    if (auto warmed = prepare_tts_profile(config_->tts_settings, [] { return false; }); !warmed)
                        std::cerr << "TTS background warm-up failed: " << warmed.error().message << '\n';
            });
        }
        if (!reaper_.joinable()) {
            reaper_ = std::jthread([this](std::stop_token stop) {
                while (!stop.stop_requested()) {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    if (stop.stop_requested() || request_gate_.any()) continue;
                    const auto now = speech_clock();
                    const auto timeout = config_->idle_model_timeout_seconds * 1000;
                    if (last_asr_use_.load() && now - last_asr_use_.load() >= timeout) {
                        std::unique_lock lock(asr_mutex_, std::try_to_lock);
                        if (lock.owns_lock() && now - last_asr_use_.load() >= timeout) {
                            transcriber_.reset();
                            last_asr_use_.store(0);
                        }
                    }
                    if (last_tts_use_.load() && now - last_tts_use_.load() >= timeout) {
                        std::unique_lock lock(tts_mutex_, std::try_to_lock);
                        if (lock.owns_lock() && now - last_tts_use_.load() >= timeout) {
                            synthesizer_.reset();
                            voice_prompts_.clear();
                            last_tts_use_.store(0);
                        }
                    }
                }
            });
        }
    }

    auto status() -> liteapp::Response {
        return json_response({
            {"ready", generator_.has_value()},
            {"config", config_ ? config_->source.string() : ""},
            {"runtime", runtime()},
            {"components", components()},
            {"model", model_path_},
            {"backend", execution_},
            {"vision", vision_supported_},
        });
    }

    auto runtime() const -> Json {
        if (!config_) return Json::object();
        return {
            {"host", "native"},
            {"threads", config_->threads},
            {"hardware_concurrency", std::max(1U, std::thread::hardware_concurrency())},
            {"concurrency", "single"},
            {"model_hub_remote", config_->model_hub_remote},
            {"model_hub_local", config_->model_hub_local.string()},
            {"speech_backend", speech_execution_},
            {"preload_speech", config_->preload_speech},
            {"idle_model_timeout_seconds", config_->idle_model_timeout_seconds},
        };
    }

    auto components() const -> Json {
        if (!config_) return Json::object();
        auto result = config_->components();
        result["chat"]["configured_system_prompt"] = config_->system_prompt;
        result["chat"]["system_prompt"] = system_prompt();
        return result;
    }

    auto system_prompt() const -> std::string {
        std::scoped_lock lock(prompt_mutex_);
        return system_prompt_;
    }

    void set_system_prompt(std::string value) {
        std::scoped_lock lock(prompt_mutex_);
        system_prompt_ = std::move(value);
    }

    auto chat(std::span<const ChatMessage> messages, std::string_view languages, std::string_view language_names,
              const std::function<bool()>& cancelled, const std::function<void(std::string_view)>& on_delta)
        -> kidi::Result<ChatRunResult> {
        auto prompt = system_prompt();
        if (!languages.empty())
            prompt += languages.find(',') == std::string_view::npos
                          ? "\nThe selected output language is " + std::string(language_names) +
                                ". You must respond entirely in " + std::string(language_names) +
                                ", regardless of the language of the user's message. This requirement overrides any "
                                "earlier instruction to match the user's language."
                          : "\nThe user may code-switch among " + std::string(language_names) +
                                ". Respond in the same language or language mix as the latest user message.";
        return generate(messages, std::move(prompt), cancelled, on_delta, config_->context_tokens, 0, false);
    }

    auto translate(std::string text, std::string previous, std::string next, std::string target_language,
                   const std::function<bool()>& cancelled, const std::function<void(std::string_view)>& on_delta)
        -> kidi::Result<ChatRunResult> {
        if (!config_)
            return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME, "chat model is not ready"});
        auto prompt = config_->translation_prompt;
        constexpr std::string_view PLACEHOLDER{"{target_language}"};
        for (auto position = prompt.find(PLACEHOLDER); position != std::string::npos;
             position = prompt.find(PLACEHOLDER, position + target_language.size())) {
            prompt.replace(position, PLACEHOLDER.size(), target_language);
        }
        const auto request = "<preceding_context>\n" + std::move(previous) +
                             "\n</preceding_context>\n<text_to_translate>\n" + std::move(text) +
                             "\n</text_to_translate>\n<following_context>\n" + std::move(next) +
                             "\n</following_context>";
        const std::array messages{ChatMessage{.role = "user", .content = request, .images = {}}};
        const auto context =
            (config_->translation.max_input_tokens + config_->maximum_tokens + 127) / 128 * 128;
        return generate(messages, std::move(prompt), cancelled, on_delta, context,
                        config_->translation.max_input_tokens, true);
    }

private:
    auto generate(std::span<const ChatMessage> messages, std::string prompt,
                  const std::function<bool()>& cancelled, const std::function<void(std::string_view)>& on_delta,
                  std::size_t context_tokens, std::size_t maximum_prompt_tokens, bool release_workspaces)
        -> kidi::Result<ChatRunResult> {
        std::scoped_lock lock(chat_mutex_);
        if (!generator_ || !config_)
            return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME, "chat model is not ready"});
        const auto started = Clock::now();
        auto run = [&]() -> kidi::Result<ChatRunResult> {
            std::vector<ChatMessage> conversation;
            conversation.reserve(messages.size() + !prompt.empty());
            if (!prompt.empty())
                conversation.push_back({.role = "system", .content = std::move(prompt), .images = {}});
            conversation.insert(conversation.end(), messages.begin(), messages.end());
            GenerationOptions options{
                .maximum_new_tokens = config_->maximum_tokens,
                .context_size = context_tokens,
                .prefill_chunk_size = config_->prefill_chunk_tokens,
                .prefix_cache_bytes = config_->prefix_cache_bytes,
                .maximum_prompt_tokens = maximum_prompt_tokens,
                .stream_text = true,
            };
            auto queued = generator_->enqueue_chat(conversation, options);
            if (!queued) return std::unexpected(queued.error());
            const auto request_id = *queued;
            std::optional<kidi::inference::TextGeneration> completed;
            while (generator_->pending_requests()) {
                if (cancelled()) {
                    auto cancellation = generator_->cancel(request_id);
                    if (!cancellation) return std::unexpected(cancellation.error());
                    return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME, "generation cancelled"});
                }
                auto step = generator_->step();
                if (!step) return std::unexpected(step.error());
                for (auto& event : step->events) {
                    if (event.request_id != request_id) continue;
                    if (!event.text.empty()) on_delta(event.text);
                    if (event.completed) completed.emplace(std::move(*event.completed));
                }
            }
            if (!completed)
                return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME,
                                                   "generation ended without a response"});
            return ChatRunResult{
                .text = std::move(completed->text),
                .stats =
                    {
                        {"elapsed_ms", milliseconds(started)},
                        {"prompt_tokens", completed->stats.prompt_tokens},
                        {"decode_tokens", completed->stats.decode_tokens},
                        {"prefill_ms", completed->stats.prefill_ns / 1e6},
                        {"decode_ms", completed->stats.decode_ns / 1e6},
                        {"preparation_ms", completed->stats.preparation_ns / 1e6},
                    },
            };
        }();
        if (release_workspaces && !generator_->pending_requests()) {
            auto released = generator_->release_workspaces();
            if (!released && run) return std::unexpected(released.error());
        }
        return run;
    }

public:

    auto transcribe(std::span<const float> samples, const std::function<bool()>& cancelled,
                    const std::function<void(std::string_view, std::string_view)>& on_partial,
                    std::string_view language = {}) -> kidi::Result<kidi::inference::Transcription> {
        std::scoped_lock lock(asr_mutex_);
        if (auto prepared = prepare_asr_locked(); !prepared) return std::unexpected(prepared.error());
        return transcriber_->transcribe(
            samples, 16000,
            {
                .language = language.empty() ? config_->asr_settings.value("language", "auto") : std::string(language),
                .maximum_tokens = config_->asr_settings.value("max_tokens", std::size_t{128}),
                .fit_audio = true,
                .on_partial = on_partial,
                .cancelled = cancelled,
            });
    }

    auto prepare_asr() -> kidi::Result<void> {
        std::scoped_lock lock(asr_mutex_);
        return prepare_asr_locked();
    }

    auto warm_asr() -> kidi::Result<void> {
        std::scoped_lock lock(asr_mutex_);
        if (auto prepared = prepare_asr_locked(); !prepared) return prepared;
        return transcriber_->warm_up();
    }

    auto synthesize(std::string_view text, const Json& settings, const std::function<bool()>& cancelled,
                    const std::function<void(std::span<const float>, std::uint32_t)>& on_audio,
                    const std::function<void(std::size_t, std::size_t)>& on_progress = {})
        -> kidi::Result<TtsRunResult> {
        std::scoped_lock lock(tts_mutex_);
        if (auto prepared = prepare_tts_locked(); !prepared) return std::unexpected(prepared.error());
        auto options = synthesis_options(settings, cancelled, on_progress);
        auto prompt = voice_prompt_locked(settings, options);
        if (!prompt) return std::unexpected(prompt.error());
        options.voice_prompt = prompt->first;
        options.on_audio_chunk = on_audio;
        options.retain_audio = false;
        auto result = synthesizer_->synthesize(text, options);
        if (!result) return std::unexpected(result.error());
        return TtsRunResult{.synthesis = std::move(*result), .voice_prompt_reused = prompt->second};
    }

    auto prepare_tts() -> kidi::Result<void> {
        std::scoped_lock lock(tts_mutex_);
        return prepare_tts_locked();
    }

    auto prepare_tts_profile(const Json& settings, const std::function<bool()>& cancelled) -> kidi::Result<void> {
        std::scoped_lock lock(tts_mutex_);
        if (auto prepared = prepare_tts_locked(); !prepared) return prepared;
        auto options = synthesis_options(settings, cancelled);
        auto prompt = voice_prompt_locked(settings, options);
        if (!prompt) return std::unexpected(prompt.error());
        return {};
    }

    auto config() const -> const AppConfig& { return *config_; }
    auto begin_request(std::string_view kind) -> bool { return request_gate_.acquire(kind); }
    void end_request(std::string_view kind) { request_gate_.release(kind); }

private:
    auto prepare_asr_locked() -> kidi::Result<void> {
        if (!config_ || !config_->asr.enabled)
            return std::unexpected(kidi::Error{kidi::ErrorCode::UNSUPPORTED, "ASR is disabled"});
        last_asr_use_.store(speech_clock());
        if (transcriber_) return {};
        const auto selected = kidi::inference::select_device(config_->asr.accelerator, true);
        if (!selected) return std::unexpected(selected.error());
        auto loaded = kidi::inference::Transcriber::load(config_->asr.directory, *selected);
        if (!loaded) return std::unexpected(loaded.error());
        transcriber_.emplace(std::move(*loaded));
        return {};
    }

    auto prepare_tts_locked() -> kidi::Result<void> {
        if (!config_ || !config_->tts.enabled)
            return std::unexpected(kidi::Error{kidi::ErrorCode::UNSUPPORTED, "TTS is disabled"});
        last_tts_use_.store(speech_clock());
        if (synthesizer_) return {};
        configure_metal_prefill_cache(config_->tts.metal_prefill_cache_bytes);
        const auto selected = kidi::inference::select_device(config_->tts.accelerator);
        if (!selected) return std::unexpected(selected.error());
        auto loaded = kidi::inference::Synthesizer::load(config_->tts.directory, *selected);
        if (!loaded) return std::unexpected(loaded.error());
        synthesizer_.emplace(std::move(*loaded));
        return {};
    }

    static auto speech_clock() -> std::uint64_t {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count());
    }

    auto synthesis_options(const Json& settings, const std::function<bool()>& cancelled,
                           const std::function<void(std::size_t, std::size_t)>& on_progress = {}) const
        -> kidi::inference::SynthesisOptions {
        kidi::inference::SynthesisOptions options;
        options.language = settings.value("language", "English");
        options.steps = settings.value("steps", std::size_t{10});
        options.duration_scale = settings.value("pace", 1.F);
        for (const auto& [name, value] : settings["voice"].items())
            if (value.is_string() && !value.get_ref<const std::string&>().empty())
                options.voice[name] = value.get<std::string>();
        options.on_progress = [cancelled, on_progress](std::size_t step, std::size_t total) {
            if (cancelled()) throw kidi::ops::Failure({kidi::ErrorCode::RUNTIME, "synthesis cancelled"});
            if (on_progress) on_progress(step, total);
        };
        return options;
    }

    auto voice_prompt_locked(const Json& settings, kidi::inference::SynthesisOptions& options)
        -> kidi::Result<std::pair<std::shared_ptr<const kidi::inference::VoicePrompt>, bool>> {
        if (!synthesizer_->supports_voice_prompt()) return std::pair{options.voice_prompt, true};
        const auto key = settings["voice"].dump();
        auto found = voice_prompts_.find(key);
        if (found != voice_prompts_.end()) return std::pair{found->second, true};
        auto reference_options = options;
        reference_options.language = config_->tts_settings.value("reference_language", "English");
        reference_options.steps = config_->tts_settings.value("steps", std::size_t{10});
        reference_options.duration_scale = 1.F;
        reference_options.on_audio_chunk = {};
        reference_options.retain_audio = false;
        const auto reference_text = config_->tts_settings.value("reference_text", "Nice to meet you.");
        auto reference = synthesizer_->synthesize(reference_text, reference_options);
        if (!reference) return std::unexpected(reference.error());
        if (!reference->voice_prompt)
            return std::unexpected(
                kidi::Error{kidi::ErrorCode::RUNTIME, "TTS model did not produce a reusable voice prompt"});
        found = voice_prompts_.emplace(key, std::move(reference->voice_prompt)).first;
        return std::pair{found->second, false};
    }

    std::mutex chat_mutex_;
    std::mutex asr_mutex_;
    std::mutex tts_mutex_;
    liteapp::ActivityGate request_gate_;
    std::optional<Generator> generator_;
    std::optional<kidi::inference::Transcriber> transcriber_;
    std::optional<kidi::inference::Synthesizer> synthesizer_;
    std::map<std::string, std::shared_ptr<const kidi::inference::VoicePrompt>, std::less<>> voice_prompts_;
    std::optional<AppConfig> config_;
    mutable std::mutex prompt_mutex_;
    std::string system_prompt_;
    std::string model_path_;
    std::string execution_;
    std::string speech_execution_;
    bool vision_supported_ = false;
    std::jthread warmup_;
    std::jthread reaper_;
    std::atomic_uint64_t last_asr_use_{0};
    std::atomic_uint64_t last_tts_use_{0};
};

constexpr std::size_t AUDIO_HEADER_BYTES = 24;
constexpr std::uint8_t AUDIO_VERSION = 1;
constexpr std::uint8_t AUDIO_ASR = 1;
constexpr std::uint8_t AUDIO_TTS = 2;
constexpr std::uint8_t IMAGE_INPUT = 3;
constexpr std::uint8_t EBOOK_INPUT = 4;

void write_u32(std::span<std::byte> output, std::size_t offset, std::uint32_t value) {
    for (std::size_t byte = 0; byte < 4; ++byte)
        output[offset + byte] = static_cast<std::byte>((value >> (byte * 8)) & 0xffU);
}

auto read_u32(std::string_view input, std::size_t offset) -> std::uint32_t {
    std::uint32_t value = 0;
    for (std::size_t byte = 0; byte < 4; ++byte)
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(input[offset + byte])) << (byte * 8);
    return value;
}

class AppSession {
public:
    explicit AppSession(std::shared_ptr<AppBackend> backend)
        : backend_(std::move(backend)), router_({.max_active_requests = 8}) {
        router_.register_method("settings.system_prompt.set",
                                [this](liteapp::ServiceContext& context) { return set_system_prompt(context); });
        router_.register_method("image.start",
                                [this](liteapp::ServiceContext& context) { return start_image(context); });
        router_.register_method("image.finish",
                                [this](liteapp::ServiceContext& context) { return finish_image(context); });
        router_.register_method("ebook.start",
                                [this](liteapp::ServiceContext& context) { return start_ebook(context); });
        router_.register_method("ebook.finish",
                                [this](liteapp::ServiceContext& context) { return finish_ebook(context); });
        router_.register_method("ebook.chapter",
                                [this](liteapp::ServiceContext& context) { return read_ebook_chapter(context); });
        router_.register_method("reader.translate", [this](liteapp::ServiceContext& context) {
            return run_inference([&] { return run_reader_translation(context); });
        });
        router_.register_method("chat.start", [this](liteapp::ServiceContext& context) {
            return run_inference([&] { return run_chat(context); });
        });
        router_.register_method("asr.start", [this](liteapp::ServiceContext& context) {
            return run_inference([&] { return run_asr(context); });
        });
        router_.register_method("asr.stop", [this](liteapp::ServiceContext& context) { return stop_asr(context); });
        router_.register_method("tts.start", [this](liteapp::ServiceContext& context) {
            return run_inference([&] { return run_tts(context); });
        });
        router_.set_fallback([](liteapp::ServiceContext& context) -> liteapp::Result<Json> {
            return failure("unknown_type", "Unknown request type: " + std::string(context.method()));
        });
        router_.set_binary_handler([this](std::span<const std::byte> frame) { handle_binary(frame); });
    }

    ~AppSession() { clear_uploaded_images(); }

    void run(liteapp::WebSocket& socket) { router_.run(socket); }

private:
    class RequestLease {
    public:
        RequestLease(AppBackend& backend, liteapp::ActivityGate& session, std::string kind)
            : backend_{backend}, session_{session}, kind_{kind} {}
        ~RequestLease() {
            backend_.end_request(kind_);
            session_.release("inference");
        }
        RequestLease(const RequestLease&) = delete;
        RequestLease& operator=(const RequestLease&) = delete;

    private:
        AppBackend& backend_;
        liteapp::ActivityGate& session_;
        std::string kind_;
    };

    static auto failure(std::string code, std::string message) -> liteapp::Result<Json> {
        return std::unexpected(liteapp::Error{std::move(code), std::move(message)});
    }

    static auto inference_failure(const kidi::Error& error) -> liteapp::Result<Json> {
        return failure("inference_failed", error.message);
    }

    template <class Work>
    static auto run_inference(Work&& work) -> liteapp::Result<Json> {
        try {
            return std::forward<Work>(work)();
        } catch (const kidi::ops::Failure& error) {
            return inference_failure(error.error());
        } catch (const std::exception& error) {
            return failure("internal_error", error.what());
        }
    }

    auto tts_settings(const Json& requested) const -> liteapp::Result<Json> {
        auto settings = backend_->config().tts_settings;
        if (requested.is_null()) return settings;
        if (!requested.is_object()) return failure("invalid_tts", "TTS options must be an object");
        for (const auto& [name, value] : requested.items()) {
            if (name == "language") {
                if (!value.is_string() || value.get_ref<const std::string&>().empty() ||
                    value.get_ref<const std::string&>().size() > 64)
                    return failure("invalid_tts", "TTS language must contain 1-64 characters");
                settings["language"] = value;
            } else if (name == "steps") {
                if (!value.is_number_unsigned() || value.get<std::size_t>() < 1 || value.get<std::size_t>() > 128)
                    return failure("invalid_tts", "TTS steps must be between 1 and 128");
                settings["steps"] = value;
            } else if (name == "pace") {
                if (!value.is_number() || !std::isfinite(value.get<float>()) || value.get<float>() < 0.5F ||
                    value.get<float>() > 2.F)
                    return failure("invalid_tts", "TTS pace must be between 0.5 and 2");
                settings["pace"] = value;
            } else if (name == "voice") {
                if (!value.is_object()) return failure("invalid_tts", "TTS voice must be an object");
                for (const auto& [attribute, selected] : value.items()) {
                    if (attribute != "gender" && attribute != "age" && attribute != "pitch" && attribute != "style" &&
                        attribute != "accent")
                        return failure("invalid_tts", "Unsupported TTS voice attribute: " + attribute);
                    if (!selected.is_string() || selected.get_ref<const std::string&>().size() > 100)
                        return failure("invalid_tts", "TTS voice attributes must be short strings");
                    settings["voice"][attribute] = selected;
                }
            } else {
                return failure("invalid_tts", "Unsupported TTS option: " + name);
            }
        }
        return settings;
    }

    auto set_system_prompt(liteapp::ServiceContext& context) -> liteapp::Result<Json> {
        const auto& payload = context.payload();
        if (!payload.is_object() || !payload.contains("prompt") || !payload["prompt"].is_string())
            return failure("invalid_system_prompt", "System prompt must be a string");
        auto prompt = payload["prompt"].get<std::string>();
        if (prompt.size() > 32768)
            return failure("invalid_system_prompt", "System prompt must not exceed 32768 UTF-8 bytes");
        backend_->set_system_prompt(prompt);
        return Json{{"prompt", std::move(prompt)}};
    }

    auto start_image(liteapp::ServiceContext& context) -> liteapp::Result<Json> {
        const auto& payload = context.payload();
        if (session_gate_.any() || !payload.is_object() || !payload.contains("stream_id") ||
            !payload["stream_id"].is_number_unsigned() || !payload.contains("size") ||
            !payload["size"].is_number_unsigned() || !payload.contains("content_type") ||
            !payload["content_type"].is_string())
            return failure("invalid_image", "Image upload requires stream_id, size, and content_type");
        const auto stream_id = payload["stream_id"].get<std::uint32_t>();
        const auto size = payload["size"].get<std::size_t>();
        const auto content_type = payload["content_type"].get<std::string>();
        std::scoped_lock lock{image_mutex_};
        if (!size || size > 32U * 1024U * 1024U || (content_type != "image/jpeg" && content_type != "image/png") ||
            image_uploads_.contains(stream_id) || uploaded_images_.contains(stream_id))
            return failure("invalid_image", "Image must be a unique JPEG or PNG no larger than 32 MiB");
        image_uploads_.emplace(stream_id, ImageUpload{
                                              .content_type = content_type,
                                              .expected_size = size,
                                          });
        return Json{{"stream_id", stream_id}};
    }

    auto finish_image(liteapp::ServiceContext& context) -> liteapp::Result<Json> {
        const auto& payload = context.payload();
        if (!payload.is_object() || !payload.contains("stream_id") || !payload["stream_id"].is_number_unsigned())
            return failure("invalid_image", "Image stream_id is required");
        const auto stream_id = payload["stream_id"].get<std::uint32_t>();
        std::scoped_lock lock{image_mutex_};
        const auto found = image_uploads_.find(stream_id);
        if (found == image_uploads_.end()) return failure("invalid_image", "Image upload is incomplete");
        if (found->second.error) {
            const auto error = *found->second.error;
            image_uploads_.erase(found);
            return std::unexpected(error);
        }
        if (found->second.bytes.size() != found->second.expected_size)
            return failure("invalid_image", "Image upload is incomplete");
        const auto token = liteapp::secure_random_token(18);
        if (!token) return failure("image_storage", token.error().message);
        const auto extension = found->second.content_type == "image/png" ? ".png" : ".jpg";
        const auto path = std::filesystem::temp_directory_path() / ("kidi-image-" + *token + extension);
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(found->second.bytes.data()),
                     static_cast<std::streamsize>(found->second.bytes.size()));
        output.close();
        if (!output) return failure("image_storage", "Unable to store the uploaded image");
        uploaded_images_[stream_id] = path;
        image_uploads_.erase(found);
        return Json{{"stream_id", stream_id}};
    }

    auto start_ebook(liteapp::ServiceContext& context) -> liteapp::Result<Json> {
        const auto& payload = context.payload();
        if (!payload.is_object() || !payload.contains("stream_id") || !payload["stream_id"].is_number_unsigned() ||
            !payload.contains("size") || !payload["size"].is_number_unsigned())
            return failure("invalid_ebook", "EPUB upload requires stream_id and size");
        const auto stream_id = payload["stream_id"].get<std::uint32_t>();
        const auto size = payload["size"].get<std::size_t>();
        std::scoped_lock lock{ebook_mutex_};
        if (!size || size > 64U * 1024U * 1024U || ebook_uploads_.contains(stream_id))
            return failure("invalid_ebook", "EPUB must use a unique stream and be no larger than 64 MiB");
        ebook_uploads_.emplace(stream_id, EbookUpload{.expected_size = size});
        return Json{{"stream_id", stream_id}};
    }

    auto finish_ebook(liteapp::ServiceContext& context) -> liteapp::Result<Json> {
        const auto& payload = context.payload();
        if (!payload.is_object() || !payload.contains("stream_id") || !payload["stream_id"].is_number_unsigned())
            return failure("invalid_ebook", "EPUB stream_id is required");
        const auto stream_id = payload["stream_id"].get<std::uint32_t>();
        std::vector<std::byte> archive;
        {
            std::unique_lock lock{ebook_mutex_};
            const auto found = ebook_uploads_.find(stream_id);
            if (found == ebook_uploads_.end()) return failure("invalid_ebook", "EPUB upload is incomplete");
            if (found->second.finishing)
                return failure("invalid_ebook", "EPUB upload is already being finalized");
            found->second.finishing = true;
            ebook_ready_.wait_for(lock, std::chrono::seconds{5}, [&] {
                return found->second.error || found->second.bytes.size() == found->second.expected_size;
            });
            if (found->second.error) {
                const auto error = *found->second.error;
                ebook_uploads_.erase(found);
                return std::unexpected(error);
            }
            if (found->second.bytes.size() != found->second.expected_size) {
                const auto message = "EPUB upload is incomplete: received " +
                                     std::to_string(found->second.bytes.size()) + " of " +
                                     std::to_string(found->second.expected_size) + " bytes";
                ebook_uploads_.erase(found);
                return failure("invalid_ebook", message);
            }
            archive = std::move(found->second.bytes);
            ebook_uploads_.erase(found);
        }
        auto parsed = kidi::desktop::parse_epub(archive);
        if (!parsed) return std::unexpected(parsed.error());
        auto book = std::make_shared<kidi::desktop::Ebook>(std::move(*parsed));
        Json chapters = Json::array();
        for (const auto& chapter : book->chapters)
            chapters.push_back({{"id", chapter.id}, {"title", chapter.title}});
        const auto result = Json{
            {"book_id", book->id},
            {"title", book->title},
            {"author", book->author},
            {"language", book->language},
            {"chapters", std::move(chapters)},
        };
        {
            std::scoped_lock lock{ebook_mutex_};
            if (!ebooks_.contains(book->id) && ebooks_.size() >= 8)
                return failure("ebook_limit", "Close and reopen Kidi before loading more books");
            ebooks_.insert_or_assign(book->id, std::move(book));
        }
        return result;
    }

    auto read_ebook_chapter(liteapp::ServiceContext& context) -> liteapp::Result<Json> {
        const auto& payload = context.payload();
        if (!payload.is_object() || !payload.contains("book_id") || !payload["book_id"].is_string() ||
            !payload.contains("chapter") || !payload["chapter"].is_number_unsigned())
            return failure("invalid_ebook", "EPUB chapter request requires book_id and chapter");
        const auto book_id = payload["book_id"].get<std::string>();
        const auto chapter = payload["chapter"].get<std::size_t>();
        std::shared_ptr<const kidi::desktop::Ebook> book;
        {
            std::scoped_lock lock{ebook_mutex_};
            const auto found = ebooks_.find(book_id);
            if (found == ebooks_.end()) return failure("ebook_unavailable", "EPUB is not loaded in this session");
            book = found->second;
        }
        auto html = kidi::desktop::read_epub_chapter(*book, chapter);
        if (!html) return std::unexpected(html.error());
        return Json{
            {"book_id", book_id},
            {"chapter", chapter},
            {"title", book->chapters[chapter].title},
            {"html", std::move(*html)},
        };
    }

    auto run_chat(liteapp::ServiceContext& context) -> liteapp::Result<Json> {
        const auto& payload = context.payload();
        if (!payload.is_object() || !payload.contains("messages"))
            return failure("invalid_messages", "messages is required");
        std::vector<ChatMessage> messages;
        {
            std::scoped_lock lock{image_mutex_};
            auto parsed = messages_from(payload["messages"], uploaded_images_);
            if (!parsed) return failure("invalid_messages", parsed.error().message);
            messages = std::move(*parsed);
        }
        if (payload.contains("languages") && !payload["languages"].is_string())
            return failure("invalid_languages", "languages must be a string");
        const auto languages = payload.value("languages", std::string{});
        if (!languages.empty() && !valid_language_setting(languages))
            return failure("invalid_languages", "languages must contain comma-separated language codes");
        if (payload.contains("language_names") && !payload["language_names"].is_string())
            return failure("invalid_languages", "language_names must be a string");
        const auto language_names = payload.value("language_names", languages);
        if (language_names.size() > 256 ||
            std::ranges::any_of(language_names, [](unsigned char value) { return value < 0x20; }))
            return failure("invalid_languages", "language_names contains invalid characters");
        if (!session_gate_.acquire("inference")) return failure("busy", "Another inference request is active");
        if (!backend_->begin_request("chat")) {
            session_gate_.release("inference");
            return failure("busy", "Another inference request is active");
        }
        RequestLease lease{*backend_, session_gate_, "chat"};
        static_cast<void>(context.emit("chat.started"));
        auto result = backend_->chat(
            messages, languages, language_names, [&context] { return context.cancelled(); },
            [&context](std::string_view delta) { static_cast<void>(context.emit("chat.delta", {{"text", delta}})); });
        clear_uploaded_images();
        if (!result) return inference_failure(result.error());
        return Json{{"text", std::move(result->text)}, {"stats", std::move(result->stats)}};
    }

    auto run_reader_translation(liteapp::ServiceContext& context) -> liteapp::Result<Json> {
        const auto& payload = context.payload();
        for (const auto* field : {"text", "previous", "next", "target_language", "target_language_name"})
            if (!payload.contains(field) || !payload[field].is_string())
                return failure("invalid_translation", std::string{"Reader translation requires string field: "} + field);
        const auto text = payload["text"].get<std::string>();
        const auto previous = payload["previous"].get<std::string>();
        const auto next = payload["next"].get<std::string>();
        const auto target_language = payload["target_language"].get<std::string>();
        const auto target_name = payload["target_language_name"].get<std::string>();
        const auto contains_control = [](std::string_view value) {
            return std::ranges::any_of(value, [](unsigned char character) {
                return character < 0x20 && character != '\n' && character != '\t';
            });
        };
        const auto characters = [](std::string_view value) {
            return static_cast<std::size_t>(
                std::ranges::count_if(value, [](unsigned char character) { return (character & 0xc0U) != 0x80U; }));
        };
        const auto maximum = backend_->config().translation.max_chars;
        if (text.empty() || characters(text) > maximum || characters(previous) > maximum ||
            characters(next) > maximum ||
            target_name.empty() || target_name.size() > 100 || contains_control(text) || contains_control(previous) ||
            contains_control(next) || contains_control(target_name) || !valid_language_setting(target_language) ||
            target_language.contains(','))
            return failure("invalid_translation", "Reader translation payload is invalid or too large");
        if (!session_gate_.acquire("inference")) return failure("busy", "Another inference request is active");
        if (!backend_->begin_request("chat")) {
            session_gate_.release("inference");
            return failure("busy", "Another inference request is active");
        }
        RequestLease lease{*backend_, session_gate_, "chat"};
        static_cast<void>(context.emit("reader.translation.started"));
        auto result = backend_->translate(
            text, previous, next, target_name, [&context] { return context.cancelled(); },
            [&context](std::string_view delta) {
                static_cast<void>(context.emit("reader.translation.delta", {{"text", delta}}));
            });
        if (!result) return inference_failure(result.error());
        return Json{{"text", std::move(result->text)}, {"stats", std::move(result->stats)}};
    }

    auto run_asr(liteapp::ServiceContext& context) -> liteapp::Result<Json> {
        const auto& payload = context.payload();
        if (!backend_->config().asr.enabled)
            return failure("asr_disabled", "ASR is disabled by the application configuration");
        if (!payload.is_object() || !payload.contains("stream_id") || !payload["stream_id"].is_number_unsigned())
            return failure("invalid_stream", "ASR stream_id must be an unsigned integer");
        const auto stream_id = payload["stream_id"].get<std::uint32_t>();
        if (payload.contains("language") && !payload["language"].is_string())
            return failure("invalid_asr", "ASR language must be a string");
        const auto language = payload.contains("language")
                                  ? payload["language"].get<std::string>()
                                  : backend_->config().asr_settings.value("language", std::string{"auto"});
        if (language.size() > 32 || !valid_language_setting(language))
            return failure("invalid_asr", "ASR language must be auto or comma-separated language codes");
        if (payload.contains("voice_mode") && !payload["voice_mode"].is_boolean())
            return failure("invalid_message", "voice_mode must be a boolean");
        const auto voice_mode = payload.value("voice_mode", false);
        auto settings = voice_mode ? tts_settings(payload.contains("tts_options") ? payload["tts_options"] : Json{})
                                   : liteapp::Result<Json>{Json::object()};
        if (!settings) return std::unexpected(settings.error());
        if (!session_gate_.acquire("inference")) return failure("busy", "Another inference request is active");
        if (!backend_->begin_request("asr")) {
            session_gate_.release("inference");
            return failure("busy", "Another inference request is active");
        }
        RequestLease lease{*backend_, session_gate_, "asr"};

        std::optional<kidi::Error> warm_error;
        std::jthread tts_warm;
        if (voice_mode) {
            tts_warm = std::jthread([&, this] {
                if (auto warmed = backend_->prepare_tts_profile(*settings, [&context] { return context.cancelled(); });
                    !warmed)
                    warm_error = warmed.error();
            });
        }
        if (auto prepared = backend_->prepare_asr(); !prepared) return inference_failure(prepared.error());
        {
            std::scoped_lock lock{audio_mutex_};
            audio_.emplace(AudioInput{
                .request_id = std::string{context.id()},
                .stream_id = stream_id,
            });
        }
        const auto clear_audio = [this, id = std::string{context.id()}] {
            std::scoped_lock lock{audio_mutex_};
            if (audio_ && audio_->request_id == id) audio_.reset();
        };
        struct AudioCleanup {
            std::function<void()> work;
            ~AudioCleanup() { work(); }
        } cleanup{clear_audio};

        static_cast<void>(context.emit("asr.started", {{"stream_id", stream_id}, {"sample_rate", 16000}}));
        std::size_t processed = 0;
        std::string previous;
        const auto interval = std::max<std::size_t>(8000, backend_->config().asr_partial_interval_ms * 16000 / 1000);
        while (!context.cancelled()) {
            std::vector<float> snapshot;
            bool final = false;
            std::optional<liteapp::Error> input_error;
            {
                std::unique_lock lock{audio_mutex_};
                audio_ready_.wait_for(lock, std::chrono::milliseconds{50}, [&] {
                    return context.cancelled() || !audio_ || audio_->error || audio_->final ||
                           audio_->samples.size() >= processed + interval;
                });
                if (context.cancelled()) break;
                if (!audio_) break;
                if (audio_->error) input_error = audio_->error;
                final = audio_->final;
                if (!input_error && !final && audio_->samples.size() < processed + interval) continue;
                snapshot = audio_->samples;
            }
            if (input_error) return std::unexpected(*input_error);
            if (snapshot.size() < 1600 && !final) continue;
            auto result = backend_->transcribe(snapshot, [&context] { return context.cancelled(); }, {}, language);
            if (!result) return inference_failure(result.error());
            processed = snapshot.size();
            {
                std::scoped_lock lock{audio_mutex_};
                if (audio_) audio_->processed = processed;
            }
            if (final) {
                if (tts_warm.joinable()) tts_warm.join();
                if (warm_error)
                    static_cast<void>(context.emit("voice.prepare.error",
                                                   {{"code", "synthesis_failed"}, {"message", warm_error->message}}));
                return Json{
                    {"text", result->text},
                    {"language", result->language},
                    {"stats",
                     {{"feature_ms", result->stats.feature_ns / 1e6},
                      {"encode_ms", result->stats.encode_ns / 1e6},
                      {"decode_ms", result->stats.decode_ns / 1e6}}},
                };
            }
            std::size_t common = 0;
            while (common < previous.size() && common < result->text.size() && previous[common] == result->text[common])
                ++common;
            const auto boundary = result->text.rfind(' ', common);
            previous = boundary == std::string::npos ? std::string{} : result->text.substr(0, boundary + 1);
            static_cast<void>(context.emit("asr.partial", {{"stable", previous},
                                                           {"provisional", result->text.substr(previous.size())},
                                                           {"language", result->language}}));
        }
        return Json::object();
    }

    auto stop_asr(liteapp::ServiceContext& context) -> liteapp::Result<Json> {
        const auto& payload = context.payload();
        if (!payload.is_object() || !payload.contains("request_id") || !payload["request_id"].is_string())
            return failure("invalid_message", "ASR request_id must be a string");
        const auto request_id = payload["request_id"].get<std::string>();
        bool stopped = false;
        {
            std::scoped_lock lock{audio_mutex_};
            if (audio_ && audio_->request_id == request_id) {
                audio_->final = true;
                stopped = true;
            }
        }
        audio_ready_.notify_all();
        return Json{{"stopped", stopped}};
    }

    auto run_tts(liteapp::ServiceContext& context) -> liteapp::Result<Json> {
        const auto& payload = context.payload();
        if (!backend_->config().tts.enabled)
            return failure("tts_disabled", "TTS is disabled by the application configuration");
        if (!payload.is_object() || !payload.contains("stream_id") || !payload["stream_id"].is_number_unsigned() ||
            !payload.contains("text") || !payload["text"].is_string())
            return failure("invalid_tts", "TTS requires text and an unsigned stream_id");
        const auto stream_id = payload["stream_id"].get<std::uint32_t>();
        const auto source = payload["text"].get<std::string>();
        if (source.empty() || source.size() > 65536)
            return failure("invalid_tts", "TTS text must contain 1-65536 UTF-8 bytes");
        auto settings = tts_settings(payload.contains("options") ? payload["options"] : Json{});
        if (!settings) return std::unexpected(settings.error());
        if (!session_gate_.acquire("inference")) return failure("busy", "Another inference request is active");
        if (!backend_->begin_request("tts")) {
            session_gate_.release("inference");
            return failure("busy", "Another inference request is active");
        }
        RequestLease lease{*backend_, session_gate_, "tts"};
        if (auto prepared = backend_->prepare_tts(); !prepared) return inference_failure(prepared.error());
        static_cast<void>(context.emit("tts.started", {{"stream_id", stream_id}, {"format", "pcm_s16le"}}));
        std::uint32_t sequence = 0;
        std::size_t total_samples = 0;
        StreamingAudioJoiner audio_joiner([&, this](std::span<const float> samples, std::uint32_t rate) {
            total_samples += samples.size();
            send_audio(context, stream_id, sequence, samples, rate);
        });
        auto result = backend_->synthesize(
            source, *settings, [&context] { return context.cancelled(); },
            [&audio_joiner](std::span<const float> samples, std::uint32_t rate) { audio_joiner.push(samples, rate); },
            [&context](std::size_t step, std::size_t steps) {
                static_cast<void>(
                    context.emit("tts.progress", {{"phase", "synthesizing"}, {"step", step}, {"steps", steps}}));
            });
        if (!result) return inference_failure(result.error());
        audio_joiner.finish();
        return Json{
            {"stream_id", stream_id},
            {"sample_rate", result->synthesis.sample_rate},
            {"samples", total_samples},
            {"chunks", result->synthesis.stats.chunks},
            {"stats",
             {{"generation_ms", result->synthesis.stats.generation_ns / 1e6},
              {"decode_ms", result->synthesis.stats.decode_ns / 1e6},
              {"preparation_ms", result->synthesis.stats.preparation_ns / 1e6},
              {"audio_tokens", result->synthesis.stats.audio_tokens}}},
            {"voice_prompt_reused", result->voice_prompt_reused},
        };
    }

    void send_audio(liteapp::ServiceContext& context, std::uint32_t stream_id, std::uint32_t& sequence,
                    std::span<const float> samples, std::uint32_t sample_rate) {
        constexpr std::size_t MAX_FRAME_SAMPLES = 4096;
        for (std::size_t start = 0; start < samples.size() && !context.cancelled(); start += MAX_FRAME_SAMPLES) {
            const auto count = std::min(MAX_FRAME_SAMPLES, samples.size() - start);
            std::vector<std::byte> frame(AUDIO_HEADER_BYTES + count * sizeof(std::int16_t));
            frame[0] = std::byte{'K'};
            frame[1] = std::byte{'D'};
            frame[2] = std::byte{'A'};
            frame[3] = std::byte{'1'};
            frame[4] = static_cast<std::byte>(AUDIO_VERSION);
            frame[5] = static_cast<std::byte>(AUDIO_TTS);
            write_u32(frame, 8, stream_id);
            write_u32(frame, 12, sequence++);
            write_u32(frame, 16, sample_rate);
            write_u32(frame, 20, static_cast<std::uint32_t>(count * sizeof(std::int16_t)));
            for (std::size_t index = 0; index < count; ++index) {
                const auto scaled = std::nearbyint(samples[start + index] * 32768.F);
                const auto value = static_cast<std::int16_t>(std::clamp(scaled, -32768.F, 32767.F));
                frame[AUDIO_HEADER_BYTES + index * 2] = static_cast<std::byte>(value & 0xff);
                frame[AUDIO_HEADER_BYTES + index * 2 + 1] =
                    static_cast<std::byte>((static_cast<std::uint16_t>(value) >> 8) & 0xff);
            }
            if (!context.send_binary(frame)) return;
        }
    }

    void handle_binary(std::span<const std::byte> bytes) {
        const std::string_view frame{reinterpret_cast<const char*>(bytes.data()), bytes.size()};
        if (frame.size() < AUDIO_HEADER_BYTES || frame.substr(0, 4) != "KDA1" ||
            static_cast<std::uint8_t>(frame[4]) != AUDIO_VERSION) {
            fail_audio("invalid_binary", "Invalid binary frame header");
            return;
        }
        const auto kind = static_cast<std::uint8_t>(frame[5]);
        const auto stream_id = read_u32(frame, 8);
        const auto sequence = read_u32(frame, 12);
        const auto sample_rate = read_u32(frame, 16);
        const auto payload_bytes = read_u32(frame, 20);
        if (kind == IMAGE_INPUT) {
            std::scoped_lock lock{image_mutex_};
            const auto found = image_uploads_.find(stream_id);
            if (found == image_uploads_.end()) return;
            if (sequence != found->second.sequence++ || sample_rate ||
                payload_bytes != frame.size() - AUDIO_HEADER_BYTES ||
                found->second.bytes.size() + payload_bytes > found->second.expected_size) {
                found->second.error = liteapp::Error{"invalid_image", "Image upload frame is invalid or out of order"};
                return;
            }
            found->second.bytes.insert(found->second.bytes.end(), frame.begin() + AUDIO_HEADER_BYTES, frame.end());
            return;
        }
        if (kind == EBOOK_INPUT) {
            std::scoped_lock ebook_lock{ebook_mutex_};
            const auto upload = ebook_uploads_.find(stream_id);
            if (upload == ebook_uploads_.end()) return;
            if (sequence != upload->second.sequence++ || sample_rate ||
                payload_bytes != frame.size() - AUDIO_HEADER_BYTES ||
                upload->second.bytes.size() + payload_bytes > upload->second.expected_size) {
                upload->second.error =
                    liteapp::Error{"invalid_ebook", "EPUB upload frame is invalid or out of order"};
                ebook_ready_.notify_all();
                return;
            }
            upload->second.bytes.insert(
                upload->second.bytes.end(), bytes.begin() + AUDIO_HEADER_BYTES, bytes.end());
            if (upload->second.bytes.size() == upload->second.expected_size) ebook_ready_.notify_all();
            return;
        }
        if (kind != AUDIO_ASR) {
            fail_audio("invalid_audio", "Invalid ASR audio frame header");
            return;
        }
        if (sample_rate != 16000 || payload_bytes != frame.size() - AUDIO_HEADER_BYTES ||
            payload_bytes % sizeof(std::int16_t)) {
            fail_audio("invalid_audio", "ASR frames must contain 16 kHz mono PCM16");
            return;
        }
        {
            std::scoped_lock lock{audio_mutex_};
            if (!audio_) return;
            if (stream_id != audio_->stream_id || sequence != audio_->expected_sequence++) {
                audio_->error = liteapp::Error{"audio_sequence", "ASR audio frame sequence is invalid"};
                audio_ready_.notify_all();
                return;
            }
            const auto samples = payload_bytes / sizeof(std::int16_t);
            constexpr std::size_t MAX_ASR_SAMPLES = 30U * 16000U;
            const auto max_queued = backend_->config().max_audio_queue_ms * 16000 / 1000;
            if (audio_->samples.size() + samples > audio_->processed + max_queued) {
                audio_->error = liteapp::Error{"audio_overflow", "ASR audio queue exceeded its configured limit"};
                audio_ready_.notify_all();
                return;
            }
            if (audio_->samples.size() + samples > MAX_ASR_SAMPLES) {
                audio_->error = liteapp::Error{"audio_too_long", "ASR input exceeds 30 seconds"};
                audio_ready_.notify_all();
                return;
            }
            audio_->samples.reserve(audio_->samples.size() + samples);
            for (std::size_t offset = AUDIO_HEADER_BYTES; offset < frame.size(); offset += 2) {
                const auto low = static_cast<std::uint8_t>(frame[offset]);
                const auto high = static_cast<std::uint8_t>(frame[offset + 1]);
                const auto value = static_cast<std::int16_t>(static_cast<std::uint16_t>(low) |
                                                             (static_cast<std::uint16_t>(high) << 8));
                audio_->samples.push_back(static_cast<float>(value) / 32768.F);
            }
        }
        audio_ready_.notify_all();
    }

    void fail_audio(std::string code, std::string message) {
        {
            std::scoped_lock lock{audio_mutex_};
            if (!audio_) return;
            audio_->error = liteapp::Error{std::move(code), std::move(message)};
        }
        audio_ready_.notify_all();
    }

    void clear_uploaded_images() {
        std::scoped_lock lock{image_mutex_};
        for (const auto& [stream, path] : uploaded_images_) {
            static_cast<void>(stream);
            std::error_code error;
            std::filesystem::remove(path, error);
        }
        uploaded_images_.clear();
        image_uploads_.clear();
    }

    struct ImageUpload {
        std::string content_type;
        std::size_t expected_size = 0;
        std::uint32_t sequence = 0;
        std::vector<std::uint8_t> bytes;
        std::optional<liteapp::Error> error;
    };

    struct EbookUpload {
        std::size_t expected_size = 0;
        std::uint32_t sequence = 0;
        bool finishing = false;
        std::vector<std::byte> bytes;
        std::optional<liteapp::Error> error;
    };

    struct AudioInput {
        std::string request_id;
        std::uint32_t stream_id = 0;
        std::uint32_t expected_sequence = 0;
        std::size_t processed = 0;
        std::vector<float> samples;
        bool final = false;
        std::optional<liteapp::Error> error;
    };

    std::shared_ptr<AppBackend> backend_;
    liteapp::ServiceRouter router_;
    liteapp::ActivityGate session_gate_;
    std::mutex image_mutex_;
    std::map<std::uint32_t, ImageUpload> image_uploads_;
    std::map<std::uint32_t, std::filesystem::path> uploaded_images_;
    std::mutex ebook_mutex_;
    std::condition_variable ebook_ready_;
    std::map<std::uint32_t, EbookUpload> ebook_uploads_;
    std::map<std::string, std::shared_ptr<kidi::desktop::Ebook>, std::less<>> ebooks_;
    std::mutex audio_mutex_;
    std::condition_variable audio_ready_;
    std::optional<AudioInput> audio_;
};
struct Arguments {
    std::filesystem::path config;
    int port = 0;
    bool dev = false;
};

auto parse_arguments(int count, char** values) -> std::optional<Arguments> {
    Arguments result;
    for (int index = 1; index < count; ++index) {
        const std::string_view argument{values[index]};
        if (argument == "--dev") {
            result.dev = true;
        } else if (argument == "--port" && index + 1 < count) {
            const std::string_view value{values[++index]};
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result.port);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result.port < 0 ||
                result.port > std::numeric_limits<std::uint16_t>::max())
                return std::nullopt;
        } else if (argument == "--help") {
            std::cout << "usage: kidi-app CONFIG [--port PORT] [--dev]\n";
            std::exit(0);
        } else if (!argument.starts_with('-') && result.config.empty()) {
            result.config = argument;
        } else {
            return std::nullopt;
        }
    }
    if (result.config.empty()) return std::nullopt;
    return result;
}

} // namespace

auto main(int argc, char** argv) -> int {
    const auto arguments = parse_arguments(argc, argv);
    if (!arguments) {
        std::cerr << "usage: kidi-app CONFIG [--port PORT] [--dev]\n";
        return 2;
    }
    auto backend = std::make_shared<AppBackend>();
    const auto loaded = backend->load_config(arguments->config);
    if (!loaded) {
        std::cerr << "Unable to load " << arguments->config << ": " << loaded.error().message << '\n';
        return 1;
    }
    std::optional<liteapp::LocalAccess> local_access;
    if (arguments->dev) {
        std::cerr << "WARNING: --dev disables local access authentication and serves source files directly.\n";
    } else {
        const auto token = liteapp::secure_random_token();
        if (!token) {
            std::cerr << "Unable to create local access token: " << token.error().message << '\n';
            return 1;
        }
        local_access = liteapp::LocalAccess{.token = *token, .cookie_name = "kidi_app_access"};
    }
    auto created = liteapp::HttpServer::create({
        .host = "127.0.0.1",
        .port = arguments->port,
        .max_body_bytes = 2U * 1024U * 1024U,
        .local_access = std::move(local_access),
        .content_security_policy =
            "default-src 'self'; base-uri 'none'; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; "
            "img-src 'self' data: blob:; script-src 'self'; style-src 'self'",
    });
    if (!created) {
        std::cerr << "Unable to create server: " << created.error().message << '\n';
        return 1;
    }
    auto server = std::move(*created);
    server.get("/api/status", [backend](const liteapp::Request&) { return backend->status(); });
    server.websocket("/ws",
                     [backend](const liteapp::Request&, liteapp::WebSocket& socket) {
                         AppSession session(backend);
                         session.run(socket);
                     },
                     {.max_text_bytes = backend->config().max_text_frame_bytes, .max_binary_bytes = 128U * 1024U});

    if (arguments->dev) {
        const auto root = std::filesystem::current_path();
        if (auto served = server.serve_manifest(root / "web/ui", root / "web/ui/assets.txt"); !served) {
            std::cerr << "Unable to serve shared development UI: " << served.error().message << '\n';
            return 1;
        }
        if (auto served = server.serve_manifest(root, root / "desktop/assets.txt"); !served) {
            std::cerr << "Unable to serve desktop development UI: " << served.error().message << '\n';
            return 1;
        }
    } else {
        auto assets = liteapp::vfs::FileSystem::from_zip(kidi::desktop_assets::archive());
        if (!assets) {
            std::cerr << "Unable to read embedded UI: " << assets.error().message << '\n';
            return 1;
        }
        if (auto served = server.serve_static(*assets); !served) {
            std::cerr << "Unable to serve embedded UI: " << served.error().message << '\n';
            return 1;
        }
    }
    const auto ran = liteapp::run_desktop_application(
        server, {
                    .app_id = "ai.gowda.kidi",
                    .title = "Kidi",
                    .profile_directory = std::filesystem::temp_directory_path() / "kidi-webview",
                    .development = arguments->dev,
                    .on_ready = [](std::string_view url) { std::cout << url << '\n'
                                                                     << std::flush; },
                    .on_started = [backend] { backend->start_background_warmup(); },
                });
    if (!ran) {
        std::cerr << "Local application failed: " << ran.error().message << '\n';
        return 1;
    }
    return 0;
}
