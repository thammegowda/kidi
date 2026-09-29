#include "kidi/checkpoint/prepare.h"
#include "kidi/inference/generator.h"
#include "kidi/inference/transcriber.h"
#include "kidi/runtime/ynn/graph.h"

#include <jni.h>
#include <nlohmann/json.hpp>
#include <uni_algo/conv.h>

#include <mutex>
#include <chrono>

namespace {
constexpr int MAXIMUM_OUTPUT_TOKENS = 8192;
constexpr std::size_t CONTEXT_TOKENS = 9216;
constexpr std::size_t PREFIX_CACHE_BYTES = 512 * 1024 * 1024;

std::mutex runtime_mutex;
std::optional<kidi::inference::Generator> generator;
std::optional<kidi::inference::Transcriber> transcriber;

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
} // namespace

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_configure(JNIEnv* environment, jobject, jint threads)
    -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        std::scoped_lock lock(runtime_mutex);
        if (threads < 1 || threads > 8) throw std::runtime_error("Threads must be between 1 and 8");
        kidi::runtime::ynn::set_thread_count(threads);
        kidi::ops::require(kidi::runtime::ynn::reserve_thread_pool(threads));
        return {{"configured", true}, {"threads", threads}, {"backend", "android-cpu"}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_load(JNIEnv* environment, jobject, jstring directory)
    -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        JavaString path(environment, directory);
        std::scoped_lock lock(runtime_mutex);
        const auto started = std::chrono::steady_clock::now();
        generator.emplace(kidi::ops::require(
            kidi::inference::Generator::load(path.get(), kidi::tensor::Device::cpu(), 0, 128, true)));
        const auto loaded_at = std::chrono::steady_clock::now();
        kidi::ops::require(generator->configure_serving({1, 1, CONTEXT_TOKENS, 32}));
        const auto ready_at = std::chrono::steady_clock::now();
        return {
            {"ready", true},
            {"native_qat", generator->native_qat()},
            {"vision", generator->vision_supported()},
            {"backend", "android-cpu"},
            {"load_ms", std::chrono::duration<double, std::milli>(ready_at - started).count()},
            {"stages_ms",
             {{"checkpoint_tokenizer_model", std::chrono::duration<double, std::milli>(loaded_at - started).count()},
              {"serving_setup", std::chrono::duration<double, std::milli>(ready_at - loaded_at).count()}}}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_enqueue(JNIEnv* environment, jobject,
                                                                           jstring messages_json, jint maximum_tokens)
    -> jstring {
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
                                                                           jstring directory, jboolean int8)
    -> jstring {
    return answer(environment, [&]() -> nlohmann::json {
        JavaString path(environment, directory);
        std::scoped_lock lock(runtime_mutex);
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
        auto loaded =
            kidi::ops::require(kidi::inference::Transcriber::load(model_directory, kidi::tensor::Device::cpu()));
        mark("checkpoint_tokenizer_model");
        transcriber.emplace(std::move(loaded));
        return {{"ready", true},
                {"stages_ms", stages},
                {"startup_warmup", false},
                {"load_ms",
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count()},
                {"backend", "android-cpu"},
                {"precision", int8 ? "int8" : "fp32"},
                {"model_bytes", std::filesystem::file_size(model_directory / "model.safetensors")}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_transcribe(JNIEnv* environment, jobject,
                                                                              jfloatArray samples_array,
                                                                              jstring language, jint maximum_tokens,
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
        kidi::inference::TranscriptionOptions options{.language = language_name.get(),
                                                      .task = "transcribe",
                                                      .maximum_tokens = static_cast<std::size_t>(maximum_tokens)};
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
        std::scoped_lock lock(runtime_mutex);
        auto result = kidi::ops::require(loaded_transcriber().transcribe(samples, 16000, options));
        return {{"text", result.text},
                {"preparation_ms", result.stats.preparation_ns / 1e6},
                {"language", result.language},
                {"token_ids", result.token_ids},
                {"feature_ms", result.stats.feature_ns / 1e6},
                {"encode_ms", result.stats.encode_ns / 1e6},
                {"decode_ms", result.stats.decode_ns / 1e6}};
    });
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_unload(JNIEnv*, jobject) -> void {
    std::scoped_lock lock(runtime_mutex);
    generator.reset();
}

extern "C" JNIEXPORT auto JNICALL Java_ai_gowda_kidi_NativeRuntime_unloadAsr(JNIEnv*, jobject) -> void {
    std::scoped_lock lock(runtime_mutex);
    transcriber.reset();
}