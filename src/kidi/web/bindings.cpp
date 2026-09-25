#include "kidi/inference/generator.h"
#include "kidi/runtime/ynn/graph.h"

#include <emscripten.h>
#include <nlohmann/json.hpp>

namespace {
constexpr int MAXIMUM_OUTPUT_TOKENS = 8192;
constexpr std::size_t CONTEXT_TOKENS = 9216;
std::optional<kidi::inference::Generator> generator;
std::string response;
int configured_threads;
#ifdef KIDI_HAS_WEBGPU
constexpr auto DEVICE = kidi::tensor::Device::web_gpu();
constexpr auto BACKEND = "webgpu";
#else
constexpr auto DEVICE = kidi::tensor::Device::cpu();
constexpr auto BACKEND = "wasm-cpu";
#endif

template <typename Function>
auto answer(Function&& function) -> const char* {
    try {
        response = function().dump();
    } catch (const kidi::ops::Failure& error) {
        response = nlohmann::json{{"error", error.error().message}}.dump();
    } catch (const std::exception& error) {
        response = nlohmann::json{{"error", error.what()}}.dump();
    } catch (...) {
        response = nlohmann::json{{"error", "unknown C++ runtime failure"}}.dump();
    }
    return response.c_str();
}

auto loaded() -> kidi::inference::Generator& {
    if (!generator) throw std::runtime_error("Load a model first");
    return *generator;
}
} // namespace

extern "C" {
EMSCRIPTEN_KEEPALIVE auto kidi_configure(int threads) -> const char* {
    return answer([&]() -> nlohmann::json {
#ifdef __EMSCRIPTEN_PTHREADS__
        if (threads < 1 || threads > 8) throw std::runtime_error("Threads must be between 1 and 8");
#else
        if (threads != 1) throw std::runtime_error("This build supports one CPU thread only");
#endif
        kidi::runtime::ynn::set_thread_count(threads);
        kidi::ops::require(kidi::runtime::ynn::reserve_thread_pool(threads));
        configured_threads = threads;
        return {{"configured", true}, {"threads", threads}, {"backend", BACKEND}};
    });
}

EMSCRIPTEN_KEEPALIVE auto kidi_load(const char* directory) -> const char* {
    return answer([&]() -> nlohmann::json {
        if (!configured_threads) throw std::runtime_error("Configure the runtime before loading the model");
        generator.emplace(kidi::ops::require(kidi::inference::Generator::load(directory, DEVICE, 0, 128, true)));
        kidi::ops::require(generator->configure_serving({1, 1, CONTEXT_TOKENS, 32}));
        return {{"ready", true},
                {"native_qat", generator->native_qat()},
                {"threads", configured_threads},
                {"backend", BACKEND}};
    });
}

EMSCRIPTEN_KEEPALIVE auto kidi_enqueue(const char* messages_json, int maximum_tokens) -> const char* {
    return answer([&]() -> nlohmann::json {
        if (maximum_tokens < 1 || maximum_tokens > MAXIMUM_OUTPUT_TOKENS)
            throw std::runtime_error("Output tokens must be between 1 and 8192");
        std::vector<kidi::text::ChatMessage> messages;
        for (const auto& message : nlohmann::json::parse(messages_json))
            messages.push_back({message.at("role").get<std::string>(), message.at("content").get<std::string>()});
        kidi::inference::GenerationOptions options;
        options.maximum_new_tokens = maximum_tokens;
        options.context_size = CONTEXT_TOKENS;
        options.prefill_chunk_size = 32;
        options.stream_text = true;
        auto id = kidi::ops::require(loaded().enqueue_chat(messages, options));
        return {{"request_id", id}};
    });
}

EMSCRIPTEN_KEEPALIVE auto kidi_step() -> const char* {
    return answer([]() -> nlohmann::json {
        auto step = kidi::ops::require(loaded().step());
        auto events = nlohmann::json::array();
        for (const auto& event : step.events) {
            nlohmann::json item{{"request_id", event.request_id}, {"text", event.text}};
            if (event.token) item["token"] = *event.token;
            if (event.completed) {
                const auto& result = *event.completed;
                item["completed"] = {{"text", result.text},
                                     {"token_ids", result.generation.token_ids},
                                     {"prompt_tokens", result.stats.prompt_tokens},
                                     {"generation_ms", result.stats.generation_ns / 1e6},
                                     {"decode_ms", result.stats.decode_ns / 1e6},
                                     {"decode_tokens", result.stats.decode_tokens}};
            }
            events.push_back(std::move(item));
        }
        return {{"events", events}, {"pending", loaded().pending_requests()}};
    });
}

EMSCRIPTEN_KEEPALIVE auto kidi_cancel(int request_id) -> const char* {
    return answer([&]() -> nlohmann::json {
        kidi::ops::require(loaded().cancel(request_id));
        return {{"cancelled", true}};
    });
}
}