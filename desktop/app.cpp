#include "kidi/core/precision.h"
#include "kidi/inference/generator.h"
#include "kidi/inference/synthesizer.h"
#include "kidi/inference/transcriber.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/text/tokenizer.h"
#include "kidi/desktop/request_gate.h"
#include "liteapp/core/random.h"
#include "liteapp/http/server.h"
#include "liteapp/platform/openers.h"
#include "liteapp/platform/webview_window.h"
#include "liteapp/vfs/filesystem.h"

#include <kidi/desktop_assets.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
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
#include <unordered_map>
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
    std::filesystem::path directory;
    std::string accelerator{"auto"};
    std::size_t metal_prefill_cache_bytes = 0;
};

struct AppConfig {
    std::filesystem::path source;
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
    Json asr_settings;
    Json tts_settings;

    auto components() const -> Json {
        return {
            {"chat",
             {
                 {"enabled", chat.enabled},
                 {"directory", chat.directory.string()},
                 {"accelerator", chat.accelerator},
                 {"metal_prefill_cache_mb", chat.metal_prefill_cache_bytes / (1024U * 1024U)},
                 {"precision", kidi::core::to_string(precision)},
                 {"kv_cache_precision", kidi::core::to_string(kv_cache_precision)},
                 {"system_prompt", system_prompt},
                 {"maximum_tokens", maximum_tokens},
                 {"context_tokens", context_tokens},
                 {"prefill_chunk_tokens", prefill_chunk_tokens},
                 {"prefix_cache_mb", prefix_cache_bytes / (1024U * 1024U)},
             }},
            {"asr",
             {
                 {"enabled", asr.enabled},
                 {"directory", asr.directory.string()},
                 {"accelerator", asr.accelerator},
                 {"settings", asr_settings},
             }},
            {"tts",
             {
                 {"enabled", tts.enabled},
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
        result.threads = positive_size(root["runtime"], "threads", 4, 64);
        result.preload_speech =
            root["runtime"] && root["runtime"]["preload_speech"] ? root["runtime"]["preload_speech"].as<bool>() : true;
        result.idle_model_timeout_seconds = positive_size(root["runtime"], "idle_model_timeout_seconds", 300, 86400);
        result.chat = model_config(source, models["chat"], true);
        result.asr = model_config(source, models["asr"], false);
        result.tts = model_config(source, models["tts"], false);
        if (!result.chat.enabled) return invalid_config("models.chat must be enabled");
        if (result.chat.directory.empty()) return invalid_config("models.chat.directory is required");
        for (const auto* model : {&result.chat, &result.asr, &result.tts}) {
            if (model->enabled && model->directory.empty())
                return invalid_config("Every enabled model requires a directory");
            if (model->accelerator != "auto" && model->accelerator != "cpu" && model->accelerator != "gpu" &&
                model->accelerator != "npu")
                return invalid_config("Model accelerator must be auto, cpu, gpu, or npu");
            if (model->enabled && !std::filesystem::is_directory(model->directory, error))
                return invalid_config("Enabled model directory does not exist: " + model->directory.string());
        }
        if (result.tts.enabled) {
            for (const auto* file : {"model.yaml", "model.safetensors", "tokenizer.json", "sentence-prefixes.tsv",
                                     "synthesis_template.jinja", "NOTICE"}) {
                if (!std::filesystem::is_regular_file(result.tts.directory / file, error))
                    return invalid_config("TTS package is missing required file: " + std::string(file));
            }
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
        const auto generation = models["chat"]["generation"];
        result.maximum_tokens = positive_size(generation, "max_tokens", 512, 2048);
        result.context_tokens = positive_size(generation, "context_tokens", 8192, 32768);
        result.prefill_chunk_tokens = positive_size(generation, "prefill_chunk_tokens", 128, 2048);
        const auto prefix_mb = positive_size(generation, "prefix_cache_mb", 512, 4096);
        result.prefix_cache_bytes = prefix_mb * 1024U * 1024U;
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
              {"silence_ms", positive_size(vad, "silence_ms", 1500, 10000)}}},
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
            {"reference_text",
             yaml_string(tts, "reference_text",
                         "Hello, I am Kidi, your helpful local assistant. I am here to help clearly and concisely.")},
            {"steps", positive_size(tts, "steps", 10, 128)},
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
            .maximum_requests = 4,
            .cache_token_budget = parsed->context_tokens,
            .prefill_tokens_per_step = parsed->prefill_chunk_tokens,
        });
        if (!configured) return std::unexpected(liteapp::Error{"load_failed", configured.error().message});
        generator_.emplace(std::move(*loaded));
        if (auto warmed = generator_->warm_up(); !warmed)
            return std::unexpected(liteapp::Error{"load_failed", warmed.error().message});
        config_.emplace(std::move(*parsed));
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
            {"runtime", config_ ? Json{{"threads", config_->threads},
                                       {"concurrency", "single"},
                                       {"preload_speech", config_->preload_speech},
                                       {"idle_model_timeout_seconds", config_->idle_model_timeout_seconds}}
                                : Json::object()},
            {"components", components()},
            {"model", model_path_},
            {"backend", execution_},
            {"vision", vision_supported_},
        });
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

    auto chat(std::span<const ChatMessage> messages, const std::function<bool()>& cancelled,
              const std::function<void(std::string_view)>& on_delta) -> kidi::Result<ChatRunResult> {
        std::scoped_lock lock(chat_mutex_);
        if (!generator_ || !config_)
            return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME, "chat model is not ready"});
        GenerationOptions options{
            .maximum_new_tokens = config_->maximum_tokens,
            .context_size = config_->context_tokens,
            .prefill_chunk_size = config_->prefill_chunk_tokens,
            .prefix_cache_bytes = config_->prefix_cache_bytes,
            .stream_text = true,
        };
        const auto started = Clock::now();
        const auto prompt = system_prompt();
        std::vector<ChatMessage> conversation;
        conversation.reserve(messages.size() + !prompt.empty());
        if (!prompt.empty()) conversation.push_back({.role = "system", .content = prompt, .images = {}});
        conversation.insert(conversation.end(), messages.begin(), messages.end());
        auto queued = generator_->enqueue_chat(conversation, options);
        if (!queued) return std::unexpected(queued.error());
        const auto request_id = *queued;
        std::optional<kidi::inference::TextGeneration> completed;
        std::uint64_t prefill_ns = 0, decode_ns = 0, preparation_ns = 0;
        while (generator_->pending_requests()) {
            if (cancelled()) {
                const auto cancellation = generator_->cancel(request_id);
                if (!cancellation) return std::unexpected(cancellation.error());
                return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME, "generation cancelled"});
            }
            auto step = generator_->step();
            if (!step) return std::unexpected(step.error());
            prefill_ns += step->prefill_ns;
            decode_ns += step->decode_ns;
            preparation_ns += step->preparation_ns;
            for (auto& event : step->events) {
                if (event.request_id != request_id) continue;
                if (!event.text.empty()) on_delta(event.text);
                if (event.completed) completed.emplace(std::move(*event.completed));
            }
        }
        if (!completed)
            return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME, "generation ended without a response"});
        return ChatRunResult{
            .text = std::move(completed->text),
            .stats =
                {
                    {"elapsed_ms", milliseconds(started)},
                    {"prompt_tokens", completed->stats.prompt_tokens},
                    {"decode_tokens", completed->stats.decode_tokens},
                    {"prefill_ms", prefill_ns / 1e6},
                    {"decode_ms", decode_ns / 1e6},
                    {"preparation_ms", preparation_ns / 1e6},
                },
        };
    }

    auto transcribe(std::span<const float> samples, const std::function<bool()>& cancelled,
                    const std::function<void(std::string_view, std::string_view)>& on_partial)
        -> kidi::Result<kidi::inference::Transcription> {
        std::scoped_lock lock(asr_mutex_);
        if (auto prepared = prepare_asr_locked(); !prepared) return std::unexpected(prepared.error());
        return transcriber_->transcribe(
            samples, 16000,
            {
                .language = config_->asr_settings.value("language", "auto"),
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
                    const std::function<void(std::span<const float>, std::uint32_t)>& on_audio)
        -> kidi::Result<TtsRunResult> {
        std::scoped_lock lock(tts_mutex_);
        if (auto prepared = prepare_tts_locked(); !prepared) return std::unexpected(prepared.error());
        auto options = synthesis_options(settings, cancelled);
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

    auto synthesis_options(const Json& settings, const std::function<bool()>& cancelled) const
        -> kidi::inference::SynthesisOptions {
        kidi::inference::SynthesisOptions options;
        options.language = settings.value("language", "English");
        options.steps = settings.value("steps", std::size_t{10});
        options.duration_scale = settings.value("pace", 1.F);
        for (const auto& [name, value] : settings["voice"].items())
            if (value.is_string() && !value.get_ref<const std::string&>().empty())
                options.voice[name] = value.get<std::string>();
        options.on_progress = [cancelled](std::size_t, std::size_t) {
            if (cancelled()) throw kidi::ops::Failure({kidi::ErrorCode::RUNTIME, "synthesis cancelled"});
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
        reference_options.duration_scale = 1.F;
        reference_options.on_audio_chunk = {};
        reference_options.retain_audio = false;
        const auto reference_text = config_->tts_settings.value(
            "reference_text",
            "Hello, I am Kidi, your helpful local assistant. I am here to help clearly and concisely.");
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
    kidi::desktop::RequestGate request_gate_;
    std::optional<Generator> generator_;
    std::optional<kidi::inference::Transcriber> transcriber_;
    std::optional<kidi::inference::Synthesizer> synthesizer_;
    std::map<std::string, std::shared_ptr<const kidi::inference::VoicePrompt>, std::less<>> voice_prompts_;
    std::optional<AppConfig> config_;
    mutable std::mutex prompt_mutex_;
    std::string system_prompt_;
    std::string model_path_;
    std::string execution_;
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
    AppSession(std::shared_ptr<AppBackend> backend, liteapp::WebSocket& socket)
        : backend_(std::move(backend)), socket_(socket) {}

    ~AppSession() {
        disconnected_.store(true);
        cancel_requested_.store(true);
        {
            std::scoped_lock lock(audio_mutex_);
            audio_final_ = true;
        }
        audio_ready_.notify_all();
        if (task_.joinable()) task_.join();
        clear_uploaded_images();
    }

    void run() {
        send_event("", "session.ready",
                   {{"components", backend_->components()},
                    {"config", backend_->config().source.string()},
                    {"runtime",
                     {{"threads", backend_->config().threads},
                      {"concurrency", "single"},
                      {"preload_speech", backend_->config().preload_speech},
                      {"idle_model_timeout_seconds", backend_->config().idle_model_timeout_seconds}}},
                    {"protocol", 1},
                    {"wire_audio", "pcm_s16le"}});
        while (socket_.is_open()) {
            auto received = socket_.read();
            if (!received || !*received) break;
            if ((**received).kind == liteapp::WebSocketMessageKind::text)
                handle_text((**received).data);
            else
                handle_binary((**received).data);
        }
    }

private:
    void send_event(std::string_view id, std::string_view type, Json payload = Json::object()) {
        std::scoped_lock lock(send_mutex_);
        if (disconnected_.load() || !socket_.is_open()) return;
        auto& sequence = sequences_[std::string(id)];
        const auto message = Json{
            {"v", 1}, {"id", id}, {"type", type}, {"seq", sequence++}, {"payload", std::move(payload)},
        };
        if (!socket_.send_text(message.dump())) disconnected_.store(true);
    }

    void send_error(std::string_view id, std::string_view code, std::string message) {
        send_event(id, "error", {{"code", code}, {"message", std::move(message)}});
    }

    auto tts_settings(std::string_view id, const Json& requested) -> std::optional<Json> {
        auto settings = backend_->config().tts_settings;
        if (requested.is_null()) return settings;
        if (!requested.is_object()) {
            send_error(id, "invalid_tts", "TTS options must be an object");
            return std::nullopt;
        }
        for (const auto& [name, value] : requested.items()) {
            if (name == "language") {
                if (!value.is_string() || value.get_ref<const std::string&>().empty() ||
                    value.get_ref<const std::string&>().size() > 64) {
                    send_error(id, "invalid_tts", "TTS language must contain 1-64 characters");
                    return std::nullopt;
                }
                settings["language"] = value;
            } else if (name == "steps") {
                if (!value.is_number_unsigned() || value.get<std::size_t>() < 1 || value.get<std::size_t>() > 128) {
                    send_error(id, "invalid_tts", "TTS steps must be between 1 and 128");
                    return std::nullopt;
                }
                settings["steps"] = value;
            } else if (name == "pace") {
                if (!value.is_number() || !std::isfinite(value.get<float>()) || value.get<float>() < 0.5F ||
                    value.get<float>() > 2.F) {
                    send_error(id, "invalid_tts", "TTS pace must be between 0.5 and 2");
                    return std::nullopt;
                }
                settings["pace"] = value;
            } else if (name == "voice") {
                if (!value.is_object()) {
                    send_error(id, "invalid_tts", "TTS voice must be an object");
                    return std::nullopt;
                }
                for (const auto& [attribute, selected] : value.items()) {
                    if (attribute != "gender" && attribute != "age" && attribute != "pitch" && attribute != "style" &&
                        attribute != "accent") {
                        send_error(id, "invalid_tts", "Unsupported TTS voice attribute: " + attribute);
                        return std::nullopt;
                    }
                    if (!selected.is_string() || selected.get_ref<const std::string&>().size() > 100) {
                        send_error(id, "invalid_tts", "TTS voice attributes must be short strings");
                        return std::nullopt;
                    }
                    settings["voice"][attribute] = selected;
                }
            } else {
                send_error(id, "invalid_tts", "Unsupported TTS option: " + name);
                return std::nullopt;
            }
        }
        return settings;
    }

    auto start_task(std::string id, std::string kind, std::function<void()> work) -> bool {
        if (active_.exchange(true)) {
            send_error(id, "busy", "Another inference request is active");
            return false;
        }
        if (!backend_->begin_request(kind)) {
            active_.store(false);
            send_error(id, "busy", "Another inference request is active");
            return false;
        }
        if (task_.joinable()) task_.join();
        cancel_requested_.store(false);
        {
            std::scoped_lock lock(active_mutex_);
            active_id_ = std::move(id);
            active_kind_ = kind;
        }
        task_ = std::jthread([this, kind, work = std::move(work)] {
            work();
            {
                std::scoped_lock lock(active_mutex_);
                active_id_.clear();
                active_kind_.clear();
            }
            backend_->end_request(kind);
            active_.store(false);
        });
        return true;
    }

    auto cancelled() const -> bool { return disconnected_.load() || cancel_requested_.load(); }

    void finish_error(std::string_view id, const kidi::Error& error) {
        if (cancelled() || error.message.find("cancel") != std::string::npos)
            send_event(id, "request.cancelled");
        else
            send_error(id, "inference_failed", error.message);
    }

    void handle_text(std::string_view text) {
        const auto message = Json::parse(text, nullptr, false);
        if (message.is_discarded() || !message.is_object() || message.value("v", 0) != 1 || !message.contains("id") ||
            !message["id"].is_string() || !message.contains("type") || !message["type"].is_string() ||
            !message.contains("payload") || !message["payload"].is_object()) {
            send_error("", "invalid_message", "Expected a versioned request object");
            return;
        }
        const auto id = message["id"].get<std::string>();
        const auto type = message["type"].get<std::string>();
        const auto& payload = message["payload"];
        if (id.empty() || id.size() > 128) {
            send_error("", "invalid_id", "Request ID must contain 1-128 characters");
            return;
        }
        if (type == "settings.system_prompt.set") {
            if (!payload.contains("prompt") || !payload["prompt"].is_string()) {
                send_error(id, "invalid_system_prompt", "System prompt must be a string");
                return;
            }
            auto prompt = payload["prompt"].get<std::string>();
            if (prompt.size() > 32768) {
                send_error(id, "invalid_system_prompt", "System prompt must not exceed 32768 UTF-8 bytes");
                return;
            }
            backend_->set_system_prompt(prompt);
            send_event(id, "settings.system_prompt.updated", {{"prompt", std::move(prompt)}});
            return;
        }
        if (type == "cancel") {
            bool active = false;
            {
                std::scoped_lock lock(active_mutex_);
                active = id == active_id_;
                if (active) {
                    cancel_requested_.store(true);
                    audio_ready_.notify_all();
                }
            }
            if (!active) send_event(id, "request.cancelled");
            return;
        }
        if (type == "image.start") {
            if (active_.load() || !payload.contains("stream_id") || !payload["stream_id"].is_number_unsigned() ||
                !payload.contains("size") || !payload["size"].is_number_unsigned() ||
                !payload.contains("content_type") || !payload["content_type"].is_string()) {
                send_error(id, "invalid_image", "Image upload requires stream_id, size, and content_type");
                return;
            }
            const auto stream_id = payload["stream_id"].get<std::uint32_t>();
            const auto size = payload["size"].get<std::size_t>();
            const auto content_type = payload["content_type"].get<std::string>();
            if (!size || size > 32U * 1024U * 1024U || (content_type != "image/jpeg" && content_type != "image/png") ||
                image_uploads_.contains(stream_id) || uploaded_images_.contains(stream_id)) {
                send_error(id, "invalid_image", "Image must be a unique JPEG or PNG no larger than 32 MiB");
                return;
            }
            image_uploads_.emplace(stream_id, ImageUpload{
                                                  .request_id = id,
                                                  .content_type = content_type,
                                                  .expected_size = size,
                                              });
            return;
        }
        if (type == "image.finish") {
            if (!payload.contains("stream_id") || !payload["stream_id"].is_number_unsigned()) {
                send_error(id, "invalid_image", "Image stream_id is required");
                return;
            }
            const auto stream_id = payload["stream_id"].get<std::uint32_t>();
            const auto found = image_uploads_.find(stream_id);
            if (found == image_uploads_.end() || found->second.request_id != id ||
                found->second.bytes.size() != found->second.expected_size) {
                send_error(id, "invalid_image", "Image upload is incomplete");
                return;
            }
            const auto token = liteapp::secure_random_token(18);
            if (!token) {
                send_error(id, "image_storage", token.error().message);
                return;
            }
            const auto extension = found->second.content_type == "image/png" ? ".png" : ".jpg";
            const auto path = std::filesystem::temp_directory_path() / ("kidi-image-" + *token + extension);
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char*>(found->second.bytes.data()),
                         static_cast<std::streamsize>(found->second.bytes.size()));
            output.close();
            if (!output) {
                send_error(id, "image_storage", "Unable to store the uploaded image");
                return;
            }
            uploaded_images_[stream_id] = path;
            image_uploads_.erase(found);
            send_event(id, "image.completed", {{"stream_id", stream_id}});
            return;
        }
        if (type == "chat.start") {
            if (!payload.contains("messages")) {
                send_error(id, "invalid_messages", "messages is required");
                return;
            }
            auto messages = messages_from(payload["messages"], uploaded_images_);
            if (!messages) {
                send_error(id, "invalid_messages", messages.error().message);
                return;
            }
            auto owned = std::move(*messages);
            start_task(id, "chat", [this, id, messages = std::move(owned)] { run_chat(id, messages); });
            return;
        }
        if (type == "asr.start") {
            if (!backend_->config().asr.enabled) {
                send_error(id, "asr_disabled", "ASR is disabled by the application configuration");
                return;
            }
            if (!payload.contains("stream_id") || !payload["stream_id"].is_number_unsigned()) {
                send_error(id, "invalid_stream", "ASR stream_id must be an unsigned integer");
                return;
            }
            const auto stream_id = payload["stream_id"].get<std::uint32_t>();
            if (payload.contains("voice_mode") && !payload["voice_mode"].is_boolean()) {
                send_error(id, "invalid_message", "voice_mode must be a boolean");
                return;
            }
            const auto voice_mode = payload.value("voice_mode", false);
            auto settings = voice_mode
                                ? tts_settings(id, payload.contains("tts_options") ? payload["tts_options"] : Json{})
                                : std::optional<Json>{Json::object()};
            if (!settings) return;
            start_task(id, "asr", [this, id, stream_id, voice_mode, settings = std::move(*settings)] {
                run_asr(id, stream_id, voice_mode, settings);
            });
            return;
        }
        if (type == "asr.stop") {
            std::scoped_lock active_lock(active_mutex_);
            if (active_kind_ != "asr" || active_id_ != id) return;
            {
                std::scoped_lock audio_lock(audio_mutex_);
                audio_final_ = true;
            }
            audio_ready_.notify_all();
            return;
        }
        if (type == "tts.start") {
            if (!backend_->config().tts.enabled) {
                send_error(id, "tts_disabled", "TTS is disabled by the application configuration");
                return;
            }
            if (!payload.contains("stream_id") || !payload["stream_id"].is_number_unsigned() ||
                !payload.contains("text") || !payload["text"].is_string()) {
                send_error(id, "invalid_tts", "TTS requires text and an unsigned stream_id");
                return;
            }
            const auto stream_id = payload["stream_id"].get<std::uint32_t>();
            const auto source = payload["text"].get<std::string>();
            if (source.empty() || source.size() > 65536) {
                send_error(id, "invalid_tts", "TTS text must contain 1-65536 UTF-8 bytes");
                return;
            }
            auto settings = tts_settings(id, payload.contains("options") ? payload["options"] : Json{});
            if (!settings) return;
            start_task(id, "tts", [this, id, stream_id, source, settings = std::move(*settings)] {
                run_tts(id, stream_id, source, settings);
            });
            return;
        }
        send_error(id, "unknown_type", "Unknown request type: " + type);
    }

    void handle_binary(std::string_view frame) {
        if (frame.size() < AUDIO_HEADER_BYTES || frame.substr(0, 4) != "KDA1" ||
            static_cast<std::uint8_t>(frame[4]) != AUDIO_VERSION) {
            send_error("", "invalid_binary", "Invalid binary frame header");
            return;
        }
        const auto kind = static_cast<std::uint8_t>(frame[5]);
        const auto stream_id = read_u32(frame, 8);
        const auto sequence = read_u32(frame, 12);
        const auto sample_rate = read_u32(frame, 16);
        const auto bytes = read_u32(frame, 20);
        if (kind == IMAGE_INPUT) {
            const auto found = image_uploads_.find(stream_id);
            if (found == image_uploads_.end() || sequence != found->second.sequence++ || sample_rate ||
                bytes != frame.size() - AUDIO_HEADER_BYTES ||
                found->second.bytes.size() + bytes > found->second.expected_size) {
                send_error(found == image_uploads_.end() ? "" : found->second.request_id, "invalid_image",
                           "Image upload frame is invalid or out of order");
                if (found != image_uploads_.end()) image_uploads_.erase(found);
                return;
            }
            found->second.bytes.insert(found->second.bytes.end(), frame.begin() + AUDIO_HEADER_BYTES, frame.end());
            return;
        }
        if (kind != AUDIO_ASR) {
            send_error("", "invalid_audio", "Invalid ASR audio frame header");
            return;
        }
        if (sample_rate != 16000 || bytes != frame.size() - AUDIO_HEADER_BYTES || bytes % sizeof(std::int16_t)) {
            send_error("", "invalid_audio", "ASR frames must contain 16 kHz mono PCM16");
            return;
        }
        std::string active_id;
        {
            std::scoped_lock lock(active_mutex_);
            if (active_kind_ != "asr") return;
            active_id = active_id_;
        }
        {
            std::scoped_lock lock(audio_mutex_);
            if (stream_id != audio_stream_id_ || sequence != audio_expected_sequence_++) {
                cancel_requested_.store(true);
                send_error(active_id, "audio_sequence", "ASR audio frame sequence is invalid");
                audio_ready_.notify_all();
                return;
            }
            const auto samples = bytes / sizeof(std::int16_t);
            constexpr std::size_t MAX_ASR_SAMPLES = 30U * 16000U;
            const auto max_queued = backend_->config().max_audio_queue_ms * 16000 / 1000;
            if (audio_samples_.size() + samples > audio_processed_ + max_queued) {
                cancel_requested_.store(true);
                send_error(active_id, "audio_overflow", "ASR audio queue exceeded its configured limit");
                audio_ready_.notify_all();
                return;
            }
            if (audio_samples_.size() + samples > MAX_ASR_SAMPLES) {
                cancel_requested_.store(true);
                send_error(active_id, "audio_too_long", "ASR input exceeds 30 seconds");
                audio_ready_.notify_all();
                return;
            }
            audio_samples_.reserve(audio_samples_.size() + samples);
            for (std::size_t offset = AUDIO_HEADER_BYTES; offset < frame.size(); offset += 2) {
                const auto low = static_cast<std::uint8_t>(frame[offset]);
                const auto high = static_cast<std::uint8_t>(frame[offset + 1]);
                const auto value = static_cast<std::int16_t>(static_cast<std::uint16_t>(low) |
                                                             (static_cast<std::uint16_t>(high) << 8));
                audio_samples_.push_back(static_cast<float>(value) / 32768.F);
            }
        }
        audio_ready_.notify_all();
    }

    void clear_uploaded_images() {
        for (const auto& [stream, path] : uploaded_images_) {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
        uploaded_images_.clear();
        image_uploads_.clear();
    }

    void run_chat(const std::string& id, std::span<const ChatMessage> messages) {
        send_event(id, "chat.started");
        auto result = backend_->chat(
            messages, [this] { return cancelled(); },
            [this, &id](std::string_view delta) {
                send_event(id, "chat.delta", {{"text", delta}});
            });
        clear_uploaded_images();
        if (result) send_event(id, "chat.completed", {{"text", result->text}, {"stats", result->stats}});
        if (!result) return finish_error(id, result.error());
    }

    void run_asr(const std::string& id, std::uint32_t stream_id, bool voice_mode, const Json& speech_settings) {
        std::optional<kidi::Error> warm_error;
        std::jthread tts_warm;
        if (voice_mode) {
            tts_warm = std::jthread([&, this] {
                if (auto warmed = backend_->prepare_tts_profile(speech_settings, [this] { return cancelled(); });
                    !warmed)
                    warm_error = warmed.error();
            });
        }
        if (auto prepared = backend_->prepare_asr(); !prepared) return finish_error(id, prepared.error());
        {
            std::scoped_lock lock(audio_mutex_);
            audio_samples_.clear();
            audio_final_ = false;
            audio_stream_id_ = stream_id;
            audio_expected_sequence_ = 0;
            audio_processed_ = 0;
        }
        send_event(id, "asr.started", {{"stream_id", stream_id}, {"sample_rate", 16000}});
        std::size_t processed = 0;
        std::string previous;
        const auto interval = std::max<std::size_t>(8000, backend_->config().asr_partial_interval_ms * 16000 / 1000);
        while (!cancelled()) {
            std::vector<float> snapshot;
            bool final = false;
            {
                std::unique_lock lock(audio_mutex_);
                audio_ready_.wait(lock, [this, processed, interval] {
                    return cancelled() || audio_final_ || audio_samples_.size() >= processed + interval;
                });
                if (cancelled()) break;
                final = audio_final_;
                if (!final && audio_samples_.size() < processed + interval) continue;
                snapshot = audio_samples_;
            }
            if (snapshot.size() < 1600 && !final) continue;
            auto result = backend_->transcribe(snapshot, [this] { return cancelled(); }, {});
            if (!result) return finish_error(id, result.error());
            processed = snapshot.size();
            {
                std::scoped_lock lock(audio_mutex_);
                audio_processed_ = processed;
            }
            if (final) {
                if (tts_warm.joinable()) tts_warm.join();
                if (warm_error) {
                    send_event(id, "voice.prepare.error",
                               {{"code", "synthesis_failed"}, {"message", warm_error->message}});
                }
                send_event(id, "asr.completed",
                           {{"text", result->text},
                            {"language", result->language},
                            {"stats",
                             {{"feature_ms", result->stats.feature_ns / 1e6},
                              {"encode_ms", result->stats.encode_ns / 1e6},
                              {"decode_ms", result->stats.decode_ns / 1e6}}}});
                return;
            }
            std::size_t common = 0;
            while (common < previous.size() && common < result->text.size() && previous[common] == result->text[common])
                ++common;
            const auto boundary = result->text.rfind(' ', common);
            previous = boundary == std::string::npos ? std::string{} : result->text.substr(0, boundary + 1);
            send_event(id, "asr.partial",
                       {{"stable", previous},
                        {"provisional", result->text.substr(previous.size())},
                        {"language", result->language}});
        }
        send_event(id, "request.cancelled");
    }

    void send_audio(std::uint32_t stream_id, std::uint32_t& sequence, std::span<const float> samples,
                    std::uint32_t sample_rate) {
        constexpr std::size_t MAX_FRAME_SAMPLES = 4096;
        for (std::size_t start = 0; start < samples.size() && !cancelled(); start += MAX_FRAME_SAMPLES) {
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
            std::scoped_lock lock(send_mutex_);
            if (!socket_.send_binary(frame)) {
                disconnected_.store(true);
                return;
            }
        }
    }

    void run_tts(const std::string& id, std::uint32_t stream_id, const std::string& text, const Json& settings) {
        if (auto prepared = backend_->prepare_tts(); !prepared) return finish_error(id, prepared.error());
        send_event(id, "tts.started", {{"stream_id", stream_id}, {"format", "pcm_s16le"}});
        std::uint32_t sequence = 0;
        std::size_t total_samples = 0;
        StreamingAudioJoiner audio_joiner([&, this](std::span<const float> samples, std::uint32_t rate) {
            total_samples += samples.size();
            send_audio(stream_id, sequence, samples, rate);
        });
        auto result = backend_->synthesize(
            text, settings, [this] { return cancelled(); },
            [&audio_joiner](std::span<const float> samples, std::uint32_t rate) { audio_joiner.push(samples, rate); });
        if (!result) return finish_error(id, result.error());
        audio_joiner.finish();
        send_event(id, "tts.completed",
                   {{"stream_id", stream_id},
                    {"sample_rate", result->synthesis.sample_rate},
                    {"samples", total_samples},
                    {"chunks", result->synthesis.stats.chunks},
                    {"voice_prompt_reused", result->voice_prompt_reused}});
    }

    struct ImageUpload {
        std::string request_id;
        std::string content_type;
        std::size_t expected_size = 0;
        std::uint32_t sequence = 0;
        std::vector<std::uint8_t> bytes;
    };

    std::shared_ptr<AppBackend> backend_;
    liteapp::WebSocket& socket_;
    std::mutex send_mutex_;
    std::unordered_map<std::string, std::uint64_t> sequences_;
    std::jthread task_;
    std::atomic_bool active_{false};
    std::atomic_bool cancel_requested_{false};
    std::atomic_bool disconnected_{false};
    std::mutex active_mutex_;
    std::string active_id_;
    std::string active_kind_;
    std::mutex audio_mutex_;
    std::condition_variable audio_ready_;
    std::vector<float> audio_samples_;
    bool audio_final_ = false;
    std::uint32_t audio_stream_id_ = 0;
    std::uint32_t audio_expected_sequence_ = 0;
    std::size_t audio_processed_ = 0;
    std::map<std::uint32_t, ImageUpload> image_uploads_;
    std::map<std::uint32_t, std::filesystem::path> uploaded_images_;
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

auto development_files(const std::filesystem::path& root) -> std::vector<liteapp::StaticFile> {
    std::vector<liteapp::StaticFile> files;
    const auto add = [&](std::string route, std::filesystem::path path, std::string content_type) {
        files.push_back({
            .route = std::move(route),
            .path = root / std::move(path),
            .content_type = std::move(content_type),
        });
    };
    const auto text = "text/javascript; charset=utf-8";
    std::ifstream manifest(root / "web/ui/assets.txt");
    for (std::string file; std::getline(manifest, file);) {
        if (file.empty()) continue;
        const auto extension = std::filesystem::path(file).extension();
        const auto content_type = extension == ".html" ? "text/html; charset=utf-8" :
                                  extension == ".css"  ? "text/css; charset=utf-8" :
                                  extension == ".svg"  ? "image/svg+xml"
                                                       : text;
        add(file == "index.html" ? "/" : '/' + file, std::filesystem::path{"web/ui"} / file, content_type);
    }
    for (const auto* file : {"backend.js", "capture-worklet.js", "playback-worklet.js"})
        add('/' + std::string(file), std::filesystem::path{"web/hosts/desktop"} / file, text);
    for (const auto* file : {"mic.svg", "message-square.svg", "settings.svg", "square.svg"})
        add("/icons/" + std::string(file), std::filesystem::path{"web/libs/lucide-static/icons"} / file,
            "image/svg+xml");
    add("/libs/marked/marked.esm.js", "web/libs/marked/marked.esm.js", text);
    add("/libs/dompurify/purify.es.mjs", "web/libs/dompurify/purify.es.mjs", text);
    add("/libs/highlightjs/highlight.min.js", "web/libs/highlightjs/highlight.min.js", text);
    add("/libs/mermaid/mermaid.min.js", "web/libs/mermaid/mermaid.min.js", text);
    return files;
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
                         AppSession session(backend, socket);
                         session.run();
                     },
                     {.max_text_bytes = backend->config().max_text_frame_bytes, .max_binary_bytes = 128U * 1024U});

    if (arguments->dev) {
        const auto files = development_files(std::filesystem::current_path());
        if (auto served = server.serve_files(files); !served) {
            std::cerr << "Unable to serve development UI: " << served.error().message << '\n';
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
    const auto port = server.bind();
    if (!port) {
        std::cerr << "Unable to bind local server: " << port.error().message << '\n';
        return 1;
    }
    auto running = server.start();
    if (!running) {
        std::cerr << "Unable to start local server: " << running.error().message << '\n';
        return 1;
    }
    backend->start_background_warmup();
    const auto launch_url = server.local_launch_url(*port);
    std::cout << launch_url << '\n' << std::flush;
    if (!arguments->dev && liteapp::platform::webview_window_supported()) {
        const auto opened = liteapp::platform::run_webview_window({
            .app_id = "ai.gowda.kidi",
            .title = "Kidi",
            .launch_url = launch_url,
            .allowed_origin = "http://127.0.0.1:" + std::to_string(*port),
            .profile_directory = std::filesystem::temp_directory_path() / "kidi-webview",
        });
        server.stop();
        if (!opened) {
            std::cerr << "Unable to open app window: " << opened.error().message << '\n';
            return 1;
        }
    } else if (!arguments->dev) {
        if (auto opened = liteapp::platform::open_url(launch_url); !opened) {
            std::cerr << "Unable to open the app automatically: " << opened.error().message << '\n';
        }
    }
    if (auto waited = running->wait(); !waited) {
        std::cerr << "Local server failed: " << waited.error().message << '\n';
        return 1;
    }
    return 0;
}
