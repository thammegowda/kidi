#include "kidi/checkpoint/prepare.h"
#include "kidi/inference/generator.h"
#include "kidi/inference/transcriber.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/tensor/backend.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <jni.h>
#include <nlohmann/json.hpp>
#include <sys/system_properties.h>
#include <uni_algo/conv.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#include <string>

namespace {
constexpr int MAXIMUM_OUTPUT_TOKENS = 8192;
constexpr std::size_t CONTEXT_TOKENS = 9216;
constexpr std::size_t PREFIX_CACHE_BYTES = 512 * 1024 * 1024;

// Chat and speech have separate locks so dictation never waits behind a model load or reply, as the web app gives
// speech its own worker. The CPU thread pool accepts concurrent callers.
std::mutex runtime_mutex, speech_mutex;
std::optional<kidi::inference::Generator> generator;
std::optional<kidi::inference::Transcriber> transcriber;
// Incremented to abandon the transcription in progress, for example a draft made stale by the final request.
std::atomic<std::uint64_t> speech_generation = 0;

class JavaString {
public:
    JavaString(JNIEnv* environment, jstring value)
        : environment_(environment), value_(value), characters_(environment->GetStringChars(value, nullptr)) {
        if (!characters_) throw std::runtime_error("Unable to read Java string");
    }

    ~JavaString() {
        if (characters_) environment_->ReleaseStringChars(value_, characters_);
    }

    auto get() const -> std::string {
        const std::u16string text(characters_, characters_ + environment_->GetStringLength(value_));
        return una::utf16to8(std::u16string_view(text));
    }

private:
    JNIEnv* environment_;
    jstring value_;
    const jchar* characters_;
};

template <typename Function>
auto answer(JNIEnv* environment, Function&& function) -> jstring {
    nlohmann::json response;
    try {
        response = function();
    } catch (const kidi::ops::Failure& error) {
        response = {{"error", error.error().message}};
    } catch (const std::exception& error) {
        response = {{"error", error.what()}};
    } catch (...) {
        response = {{"error", "Unknown C++ runtime failure"}};
    }
    return environment->NewStringUTF(response.dump(-1, ' ', true).c_str());
}

auto loaded() -> kidi::inference::Generator& {
    if (!generator) throw std::runtime_error("Load a model first");
    return *generator;
}

auto loaded_transcriber() -> kidi::inference::Transcriber& {
    if (!transcriber) throw std::runtime_error("Load a speech model first");
    return *transcriber;
}

auto messages(std::string_view source) -> std::vector<kidi::text::ChatMessage> {
    std::vector<kidi::text::ChatMessage> result;
    for (const auto& message : nlohmann::json::parse(source)) {
        kidi::text::ChatMessage value{message.at("role").get<std::string>(), message.at("content").get<std::string>()};
        if (message.contains("images"))
            for (const auto& image : message.at("images")) value.images.emplace_back(image.get<std::string>());
        result.push_back(std::move(value));
    }
    return result;
}

/// Devices to try for a preference: "auto" falls back from the NPU to the GPU to the CPU when loading fails;
/// an explicit accelerator is tried alone so failures stay visible.
auto candidates(std::string_view accelerator, bool speech) -> std::vector<kidi::tensor::Device> {
    if (accelerator != "auto") return {kidi::ops::require(kidi::inference::select_device(accelerator, speech))};
    if (speech) return {kidi::tensor::Device::cpu()};
    std::vector<kidi::tensor::Device> result;
    for (const auto choice : {"npu", "gpu"}) {
        auto device = kidi::inference::select_device(choice);
        if (device) result.push_back(*device);
    }
    result.push_back(kidi::tensor::Device::cpu());
    return result;
}

auto options(int maximum_tokens) -> kidi::inference::GenerationOptions {
    if (maximum_tokens < 1 || maximum_tokens > MAXIMUM_OUTPUT_TOKENS)
        throw std::runtime_error("Output tokens must be between 1 and 8192");
    kidi::inference::GenerationOptions result;
    result.maximum_new_tokens = maximum_tokens;
    result.context_size = CONTEXT_TOKENS;
    result.prefill_chunk_size = 32;
    result.prefix_cache_bytes = PREFIX_CACHE_BYTES;
    result.stream_text = true;
    return result;
}

auto android_property(const char* name) -> std::string {
    std::array<char, PROP_VALUE_MAX> value{};
    const auto length = __system_property_get(name, value.data());
    return length > 0 ? std::string(value.data(), static_cast<std::size_t>(length)) : std::string{};
}

auto lowercase(std::string value) -> std::string {
    std::ranges::transform(value, value.begin(),
                           [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return value;
}

auto library_available(const char* name) -> bool {
    if (auto* library = dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
        dlclose(library);
        return true;
    }
    return false;
}

auto hardware_info() -> nlohmann::json {
    const auto manufacturer = android_property("ro.soc.manufacturer");
    const auto model = android_property("ro.soc.model");
    const auto hardware = android_property("ro.hardware");
    const auto identity = lowercase(manufacturer + ' ' + model + ' ' + hardware);
    const bool qualcomm = identity.contains("qualcomm") || identity.contains("qti") || identity.contains("qcom");
    const bool sm8750 = model == "SM8750";

    auto cpu_name = manufacturer.empty() ? model : manufacturer + (model.empty() ? "" : " " + model);
    if (sm8750) cpu_name = "Qualcomm Snapdragon 8 Elite (SM8750)";
    if (cpu_name.empty()) cpu_name = hardware.empty() ? "Unknown CPU" : hardware;

    nlohmann::json gpu{{"name", sm8750 ? "Qualcomm Adreno 830" : "Unknown GPU"},
                       {"backend", "vulkan"},
                       {"recognized", sm8750},
                       {"available", false},
                       {"reason", "No Vulkan backend was registered"}};
    for (const auto& backend : kidi::tensor::BackendRegistry::instance().backends()) {
        if (backend.device_kind != kidi::tensor::DeviceKind::VULKAN) continue;
        const auto probed = backend.device_name;
        const bool adreno = lowercase(probed).contains("adreno");
        gpu = {{"name", adreno ? probed : (sm8750 ? "Qualcomm Adreno 830" : probed)},
               {"backend", backend.name},
               {"recognized", adreno || sm8750},
               {"available", backend.execution_available},
               {"reason", backend.execution_available ? "" : backend.unavailable_reason}};
        break;
    }

    const auto skel_directory =
        std::filesystem::path(std::getenv("KIDI_QNN_SKEL_DIR") ? std::getenv("KIDI_QNN_SKEL_DIR") : "");
    const bool htp_runtime = library_available("libQnnHtp.so");
    const bool fastrpc = library_available("libcdsprpc.so");
    const bool skel = !skel_directory.empty() && std::filesystem::exists(skel_directory / "libQnnHtpV79Skel.so");
    const bool npu_available = qualcomm && htp_runtime && fastrpc && skel;
    std::string npu_reason;
    if (!qualcomm)
        npu_reason = "Qualcomm SoC not recognized";
    else if (!htp_runtime)
        npu_reason = "QNN HTP runtime is unavailable";
    else if (!fastrpc)
        npu_reason = "FastRPC library is unavailable";
    else if (!skel)
        npu_reason = "Hexagon V79 skel is unavailable";

    return {
        {"soc", {{"manufacturer", manufacturer}, {"model", model}, {"hardware", hardware}}},
        {"cpu",
         {{"name", cpu_name},
          {"backend", "ynnpack"},
          {"recognized", true},
          {"available", true},
          {"detail", kidi::runtime::ynn::supported_arch_names()},
          {"reason", ""}}},
        {"gpu", std::move(gpu)},
        {"npu",
         {{"name", sm8750 ? "Qualcomm Hexagon HTP V79" : "Qualcomm Hexagon HTP"},
          {"backend", "qnn-htp"},
          {"recognized", qualcomm},
          {"available", npu_available},
          {"reason", npu_reason}}},
    };
}
} // namespace

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_configure(JNIEnv* environment, jobject,
                                                                             jint threads) -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        if (threads < 1 || threads > 8) throw std::runtime_error("Threads must be between 1 and 8");
        // Chat and speech both configure before loading on their own threads. Serializing configuration lets the
        // second caller see the count already applied and skip the model locks, rather than wait behind a load.
        static std::mutex configure_mutex;
        std::scoped_lock configure_lock(configure_mutex);
        if (kidi::runtime::ynn::thread_count() == static_cast<std::size_t>(threads))
            return {{"configured", true}, {"threads", threads}, {"backend", "android-cpu"}};
        std::scoped_lock lock(runtime_mutex, speech_mutex);
        kidi::runtime::ynn::set_thread_count(threads);
        kidi::ops::require(kidi::runtime::ynn::reserve_thread_pool(threads));
        return {{"configured", true}, {"threads", threads}, {"backend", "android-cpu"}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_deviceInfo(JNIEnv* environment, jobject) -> jstring {
    return answer(environment, [] { return hardware_info(); });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_setDiagnosticLogging(JNIEnv* environment, jobject,
                                                                                        jboolean enabled) -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        setenv("KIDI_QNN_DUMP", enabled ? "1" : "", 1);
        return {{"diagnostic_logging", static_cast<bool>(enabled)}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_setDataDirectory(JNIEnv* environment, jobject,
                                                                                    jstring directory,
                                                                                    jstring dsp_directory) -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        const std::filesystem::path root(JavaString(environment, directory).get());
        const auto dsp_directory_value = JavaString(environment, dsp_directory).get();
        // Packaged QNN libraries are preferred; files/qnn is a fallback for custom builds.
        const auto libraries = root / "qnn", cache = root / "qnn-cache";
        std::filesystem::create_directories(cache);
        setenv("KIDI_QNN_LIBRARY_DIR", libraries.c_str(), 0);
        setenv("KIDI_QNN_CACHE_DIR", cache.c_str(), 0);
        if (!dsp_directory_value.empty()) setenv("KIDI_QNN_SKEL_DIR", dsp_directory_value.c_str(), 1);
        return {{"qnn_library_dir", libraries.string()},
                {"qnn_cache_dir", cache.string()},
                {"qnn_skel_dir", dsp_directory_value}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_load(JNIEnv* environment, jobject, jstring directory,
                                                                        jstring accelerator, jstring precision_name,
                                                                        jstring kv_cache_precision_name) -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        JavaString path(environment, directory);
        const auto preference = JavaString(environment, accelerator).get();
        const auto precision_text = JavaString(environment, precision_name).get();
        const auto precision = kidi::core::parse_precision(precision_text);
        if (!precision)
            throw kidi::ops::Failure(
                {kidi::ErrorCode::INVALID_ARGUMENT, "unknown inference precision: " + precision_text});
        const auto kv_cache_precision_text = JavaString(environment, kv_cache_precision_name).get();
        const auto kv_cache_precision = kidi::core::parse_kv_cache_precision(kv_cache_precision_text);
        if (!kv_cache_precision)
            throw kidi::ops::Failure(
                {kidi::ErrorCode::INVALID_ARGUMENT, "unknown KV-cache precision: " + kv_cache_precision_text});
        std::scoped_lock lock(runtime_mutex);
        const auto started = std::chrono::steady_clock::now();
        const auto milliseconds = [](auto from, auto to) {
            return std::chrono::duration<double, std::milli>(to - from).count();
        };
        generator.reset();
        nlohmann::json attempts = nlohmann::json::array(), stages;
        for (const auto device : candidates(preference, false)) {
            const auto attempt_started = std::chrono::steady_clock::now();
            auto ready = [&]() -> kidi::Result<kidi::inference::Generator> {
                auto loaded = kidi::inference::Generator::load(
                    path.get(),
                    {.device = device,
                     .precision = *precision,
                     .kv_cache_precision = *kv_cache_precision,
                     .packed_prefill = true});
                if (!loaded) return loaded;
                const auto loaded_at = std::chrono::steady_clock::now();
                if (auto configured = loaded->configure_serving({1, 1, CONTEXT_TOKENS, 32}); !configured)
                    return std::unexpected(configured.error());
                const auto configured_at = std::chrono::steady_clock::now();
                // A short request with the serving shapes packs CPU weights, and on the NPU captures the decode step
                // so its context loads (or starts compiling) now rather than during the first reply. Nothing is
                // retained, and a device that cannot finish it is treated as a failed load.
                auto warm = options(3);
                warm.prefix_cache_bytes = 0;
                warm.stream_text = false;
                if (auto request = loaded->enqueue_chat(std::vector{kidi::text::ChatMessage{"user", "Hi"}}, warm);
                    !request)
                    return std::unexpected(request.error());
                while (loaded->pending_requests())
                    if (auto step = loaded->step(); !step) return std::unexpected(step.error());
                const auto ready_at = std::chrono::steady_clock::now();
                stages = {{"checkpoint_tokenizer_model", milliseconds(attempt_started, loaded_at)},
                          {"serving_setup", milliseconds(loaded_at, configured_at)},
                          {"warm_up", milliseconds(configured_at, ready_at)}};
                return loaded;
            }();
            if (ready) {
                generator.emplace(std::move(*ready));
                break;
            }
            attempts.push_back({{"device", kidi::tensor::to_string(device)}, {"error", ready.error().message}});
            if (preference != "auto") throw kidi::ops::Failure(ready.error());
        }
        if (!generator) throw std::runtime_error(attempts.back()["error"].get<std::string>());
        return {{"ready", true},
                {"native_qat", generator->native_qat()},
                {"vision", generator->vision_supported()},
                {"backend", generator->execution()},
                {"accelerator", preference},
                {"precision", kidi::core::to_string(generator->precision())},
                {"kv_cache_precision", kidi::core::to_string(generator->kv_cache_precision())},
                {"fallbacks", attempts},
                {"load_ms", milliseconds(started, std::chrono::steady_clock::now())},
                {"stages_ms", stages}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_enqueue(JNIEnv* environment, jobject,
                                                                           jstring messages_json,
                                                                           jint maximum_tokens) -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        JavaString source(environment, messages_json);
        std::scoped_lock lock(runtime_mutex);
        const auto request_id =
            kidi::ops::require(loaded().enqueue_chat(messages(source.get()), options(maximum_tokens)));
        return {{"request_id", request_id}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_step(JNIEnv* environment, jobject) -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        std::scoped_lock lock(runtime_mutex);
        auto step = kidi::ops::require(loaded().step());
        auto events = nlohmann::json::array();
        for (const auto& event : step.events) {
            nlohmann::json item{{"request_id", event.request_id}, {"text", event.text}};
            if (event.token) item["token"] = *event.token;
            if (event.completed) {
                const auto& result = *event.completed;
                item["completed"] = {{"text", result.text},
                                     {"prompt_tokens", result.stats.prompt_tokens},
                                     {"reused_prompt_tokens", result.stats.reused_prompt_tokens},
                                     {"prefix_cache_bytes", result.stats.prefix_cache_bytes},
                                     {"prefix_reserved_bytes", result.stats.prefix_reserved_bytes},
                                     {"prefill_ms", result.stats.prefill_ns / 1e6},
                                     {"first_token_ms", result.stats.time_to_first_token_ns / 1e6},
                                     {"output_tokens", result.generation.token_ids.size()},
                                     {"generation_ms", result.stats.generation_ns / 1e6},
                                     {"decode_ms", result.stats.decode_ns / 1e6},
                                     {"decode_tokens", result.stats.decode_tokens}};
            }
            events.push_back(std::move(item));
        }
        return {{"events", events}, {"pending", loaded().pending_requests()}, {"decode_ms", step.decode_ns / 1e6}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_cancel(JNIEnv* environment, jobject,
                                                                          jlong request_id) -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        std::scoped_lock lock(runtime_mutex);
        kidi::ops::require(loaded().cancel(static_cast<std::uint64_t>(request_id)));
        return {{"cancelled", true}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_loadAsr(JNIEnv* environment, jobject,
                                                                           jstring directory, jboolean int8,
                                                                           jstring accelerator) -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        JavaString path(environment, directory);
        const auto preference = JavaString(environment, accelerator).get();
        const auto devices = candidates(preference, true);
        std::scoped_lock lock(speech_mutex);
        transcriber.reset();
        const auto load_start = std::chrono::steady_clock::now();
        auto stage_start = load_start;
        nlohmann::json stages = nlohmann::json::object();
        const auto mark = [&](const char* stage) {
            const auto now = std::chrono::steady_clock::now();
            stages[stage] = std::chrono::duration<double, std::milli>(now - stage_start).count();
            stage_start = now;
        };
        const auto model_directory = int8 ? kidi::ops::require(kidi::checkpoint::prepare(
                                                path.get(), kidi::model::WhisperImpl::checkpoint_config(),
                                                kidi::model::WhisperImpl::int8_preparation))
                                          : std::filesystem::path(path.get());
        mark("int8_cache_validation");
        std::string failure;
        for (const auto device : devices) {
            auto loaded = kidi::inference::Transcriber::load(model_directory, device);
            if (loaded) {
                transcriber.emplace(std::move(*loaded));
                break;
            }
            if (preference != "auto") throw kidi::ops::Failure(loaded.error());
            failure = loaded.error().message;
        }
        if (!transcriber) throw std::runtime_error(failure);
        mark("checkpoint_tokenizer_model");
        // Weight packing and operator preparation happen here instead of during the first dictation.
        kidi::ops::require(transcriber->warm_up());
        mark("warm_up");
        return {{"ready", true},
                {"stages_ms", stages},
                {"startup_warmup", true},
                {"load_ms",
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count()},
                {"backend", transcriber->execution()},
                {"accelerator", preference},
                {"precision", int8 ? "int8" : "fp32"},
                {"model_bytes", std::filesystem::file_size(model_directory / "model.safetensors")}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_transcribe(JNIEnv* environment, jobject,
                                                                              jfloatArray samples_array,
                                                                              jstring language, jint maximum_tokens,
                                                                              jlong requested_generation,
                                                                              jobject listener) -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        const auto sample_count = environment->GetArrayLength(samples_array);
        if (sample_count < 1 || sample_count > 480000)
            throw std::runtime_error("Speech must contain 1 to 480000 samples");
        if (maximum_tokens < 1 || maximum_tokens > 444)
            throw std::runtime_error("Speech output tokens must be between 1 and 444");
        std::vector<float> samples(static_cast<std::size_t>(sample_count));
        environment->GetFloatArrayRegion(samples_array, 0, sample_count, samples.data());
        if (environment->ExceptionCheck()) throw std::runtime_error("Unable to read speech samples");
        JavaString language_name(environment, language);
        // A request belongs to the generation current when it was made, so cancelling also covers queued requests.
        const auto generation =
            requested_generation < 0 ? speech_generation.load() : static_cast<std::uint64_t>(requested_generation);
        kidi::inference::TranscriptionOptions options{
            .language = language_name.get(),
            .task = "transcribe",
            .maximum_tokens = static_cast<std::size_t>(maximum_tokens),
            // Like the web app: encode the speech plus a second of silence instead of a padded 30-second window.
            .fit_audio = true,
            .cancelled = [generation] { return speech_generation.load() != generation; }};
        if (listener) {
            const auto listener_class = environment->GetObjectClass(listener);
            if (!listener_class) {
                environment->ExceptionClear();
                throw std::runtime_error("Unable to read transcription listener");
            }
            const auto method = environment->GetMethodID(listener_class, "onPartial", "(Ljava/lang/String;)V");
            environment->DeleteLocalRef(listener_class);
            if (!method) {
                environment->ExceptionClear();
                throw std::runtime_error("Invalid transcription listener");
            }
            options.on_partial = [environment, listener, method](std::string_view text,
                                                                 std::string_view detected_language) {
                const auto payload =
                    nlohmann::json{{"text", text}, {"language", detected_language}}.dump(-1, ' ', true);
                const auto value = environment->NewStringUTF(payload.c_str());
                if (!value) {
                    environment->ExceptionClear();
                    throw kidi::ops::Failure({kidi::ErrorCode::RUNTIME, "Unable to deliver partial transcript"});
                }
                environment->CallVoidMethod(listener, method, value);
                environment->DeleteLocalRef(value);
                if (environment->ExceptionCheck()) {
                    environment->ExceptionClear();
                    throw kidi::ops::Failure({kidi::ErrorCode::RUNTIME, "Partial transcript listener failed"});
                }
            };
        }
        std::scoped_lock lock(speech_mutex);
        if (options.cancelled()) return {{"cancelled", true}};
        auto transcribed = loaded_transcriber().transcribe(samples, 16000, options);
        if (!transcribed && options.cancelled()) return {{"cancelled", true}};
        auto result = kidi::ops::require(std::move(transcribed));
        return {{"text", result.text},
                {"preparation_ms", result.stats.preparation_ns / 1e6},
                {"language", result.language},
                {"token_ids", result.token_ids},
                {"feature_ms", result.stats.feature_ns / 1e6},
                {"encode_ms", result.stats.encode_ns / 1e6},
                {"decode_ms", result.stats.decode_ns / 1e6}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_cancelTranscription(JNIEnv*, jobject) -> jlong {
    return static_cast<jlong>(++speech_generation);
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_transcriptionGeneration(JNIEnv*, jobject) -> jlong {
    return static_cast<jlong>(speech_generation.load());
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_unload(JNIEnv*, jobject) -> void {
    std::scoped_lock lock(runtime_mutex);
    generator.reset();
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_unloadAsr(JNIEnv*, jobject) -> void {
    std::scoped_lock lock(speech_mutex);
    transcriber.reset();
}