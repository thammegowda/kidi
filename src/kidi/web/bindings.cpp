#include "kidi/core/memory.h"
#include "kidi/inference/generator.h"
#include "kidi/inference/transcriber.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/tensor/external.h"
#ifdef KIDI_HAS_WEBGPU
#include "kidi/tensor/web_gpu.h"
#endif

#include <emscripten.h>
#include <emscripten/heap.h>
#include <nlohmann/json.hpp>

namespace {
constexpr int MAXIMUM_OUTPUT_TOKENS = 8192;
constexpr std::size_t CONTEXT_TOKENS = 9216;
std::optional<kidi::inference::Generator> generator;
std::optional<kidi::inference::Transcriber> transcriber;
std::string response;
int configured_threads;
// clang-format off
EM_JS(void, image_progress, (const char* stage), {
    self.postMessage({type: 'encoding-images', stage: UTF8ToString(Number(stage)), heapBytes: HEAPU8.byteLength});
});
EM_JS(int, gather_external_rows,
      (const char* name, const std::int32_t* rows, std::size_t count, void* destination, std::size_t row_bytes), {
    try {
        Module.kidiExternal.gather(UTF8ToString(Number(name)), Number(rows), Number(count), Number(destination),
                                   Number(row_bytes));
        return 0;
    } catch (error) {
        err(`External tensor rows failed: ${error}`);
        return -1;
    }
});
#ifdef KIDI_HAS_WEBGPU
EM_JS(int, external_gpu_layout, (const char* name), {
    return Module.kidiExternal?.gpu?.get(UTF8ToString(Number(name)))?.layout ?? 0;
});
EM_JS(int, take_external_gpu_buffer, (const char* name), {
    const key = UTF8ToString(Number(name));
    const entry = Module.kidiExternal?.gpu?.get(key);
    Module.kidiExternal?.gpu?.delete(key);
    return entry?.handle ?? 0;
});
#endif
// clang-format on

/// Rows of a checkpoint table the browser loader keeps in JavaScript buffers instead of the Wasm heap.
class JavaScriptRows final : public kidi::tensor::RowSource {
public:
    explicit JavaScriptRows(std::string name) : name_(std::move(name)) {}
    auto gather(std::span<const std::int32_t> rows, std::span<std::byte> destination) const
        -> kidi::Result<void> override {
        if (rows.empty()) return {};
        if (gather_external_rows(name_.c_str(), rows.data(), rows.size(), destination.data(),
                                 destination.size() / rows.size()))
            return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME, "cannot read external rows of " + name_});
        return {};
    }

private:
    std::string name_;
};
/// Tensors the loader kept out of the heap: GPU buffers it already filled, or JavaScript row tables.
auto external_tensor(std::string_view name, kidi::tensor::DType dtype, std::span<const std::int64_t> shape)
    -> kidi::Result<kidi::tensor::Tensor> {
    const std::string key(name);
#ifdef KIDI_HAS_WEBGPU
    const auto layout = external_gpu_layout(key.c_str());
    if (const auto handle = take_external_gpu_buffer(key.c_str()))
        return kidi::tensor::adopt_web_gpu_buffer(static_cast<std::uint32_t>(handle), {shape.begin(), shape.end()},
                                                  dtype, layout);
#endif
    return kidi::tensor::external_tensor({shape.begin(), shape.end()}, dtype, std::make_shared<JavaScriptRows>(key));
}
#ifdef KIDI_HAS_WEBGPU
constexpr auto DEVICE = kidi::tensor::Device::web_gpu();
constexpr auto BACKEND = "webgpu";
// Each prefill chunk costs a JavaScript round trip, while the GPU has ample parallelism for wider chunks.
constexpr std::size_t PREFILL_TOKENS = 256;
#else
constexpr auto DEVICE = kidi::tensor::Device::cpu();
constexpr auto BACKEND = "wasm-cpu";
constexpr std::size_t PREFILL_TOKENS = 32;
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
auto loaded_transcriber() -> kidi::inference::Transcriber& {
    if (!transcriber) throw std::runtime_error("Load a speech model first");
    return *transcriber;
}
auto messages(const char* messages_json) -> std::vector<kidi::text::ChatMessage> {
    std::vector<kidi::text::ChatMessage> result;
    for (const auto& message : nlohmann::json::parse(messages_json)) {
        kidi::text::ChatMessage item{message.at("role").get<std::string>(), message.at("content").get<std::string>()};
        if (message.contains("images"))
            for (const auto& image : message.at("images")) item.images.emplace_back(image.get<std::string>());
        result.push_back(std::move(item));
    }
    return result;
}
auto options(int maximum_tokens) -> kidi::inference::GenerationOptions {
    if (maximum_tokens < 1 || maximum_tokens > MAXIMUM_OUTPUT_TOKENS)
        throw std::runtime_error("Output tokens must be between 1 and 8192");
    kidi::inference::GenerationOptions result;
    result.maximum_new_tokens = maximum_tokens;
    result.context_size = CONTEXT_TOKENS;
    result.prefill_chunk_size = PREFILL_TOKENS;
    result.stream_text = true;
    result.on_image_progress = [](std::string_view stage) { image_progress(std::string(stage).c_str()); };
    return result;
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
        generator.emplace(
            kidi::ops::require(kidi::inference::Generator::load(directory, DEVICE, 0, 128, true, external_tensor)));
        // Selected tokens arrive one call later, so WebGPU work completes while JavaScript awaits between steps.
        kidi::ops::require(generator->configure_serving({.maximum_active = 1,
                                                         .maximum_requests = 1,
                                                         .cache_token_budget = CONTEXT_TOKENS,
                                                         .prefill_tokens_per_step = PREFILL_TOKENS,
                                                         .compact_cache = true,
                                                         .deferred_tokens = true}));
        // Pay first-use weight packing and kernel preparation during load rather than on the first message.
        kidi::ops::require(generator->warm_up());
        return {{"ready", true},
                {"native_qat", generator->native_qat()},
                {"vision_supported", generator->vision_supported()},
                {"threads", configured_threads},
                {"backend", BACKEND}};
    });
}

EMSCRIPTEN_KEEPALIVE auto kidi_enqueue(const char* messages_json, int maximum_tokens, int image_max_pixels) -> const
    char* {
    return answer([&]() -> nlohmann::json {
        const auto request = messages(messages_json);
        auto settings = options(maximum_tokens);
        settings.image_max_pixels = static_cast<std::size_t>(image_max_pixels);
        auto id = kidi::ops::require(loaded().enqueue_chat(request, settings));
        return {{"request_id", id}};
    });
}

EMSCRIPTEN_KEEPALIVE auto kidi_load_asr(const char* directory) -> const char* {
    return answer([&]() -> nlohmann::json {
        if (!configured_threads) throw std::runtime_error("Configure the runtime before loading the speech model");
        transcriber.emplace(
            kidi::ops::require(kidi::inference::Transcriber::load(directory, kidi::tensor::Device::cpu())));
        return {{"ready", true}, {"threads", configured_threads}, {"backend", "wasm-cpu"}};
    });
}

EMSCRIPTEN_KEEPALIVE auto kidi_transcribe(const float* samples_address, std::uint32_t sample_count,
                                          const char* language, int maximum_tokens) -> const char* {
    return answer([&]() -> nlohmann::json {
        if (!samples_address || !sample_count || sample_count > 480000)
            throw std::runtime_error("Speech must contain 1 to 480000 samples");
        if (maximum_tokens < 1 || maximum_tokens > 444)
            throw std::runtime_error("Speech output tokens must be between 1 and 444");
        const auto samples = std::span(samples_address, static_cast<std::size_t>(sample_count));
        auto result = kidi::ops::require(loaded_transcriber().transcribe(
            samples, 16000,
            {.language = language, .task = "transcribe", .maximum_tokens = static_cast<std::size_t>(maximum_tokens)}));
        return {{"text", result.text},
                {"language", result.language},
                {"token_ids", result.token_ids},
                {"feature_ms", result.stats.feature_ns / 1e6},
                {"encode_ms", result.stats.encode_ns / 1e6},
                {"decode_ms", result.stats.decode_ns / 1e6}};
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

EMSCRIPTEN_KEEPALIVE auto kidi_memory_stats() -> const char* {
    return answer([]() -> nlohmann::json {
        const auto usage = kidi::core::heap_usage();
        nlohmann::json result{{"heap_bytes", emscripten_get_heap_size()},
                              {"heap_limit_bytes", emscripten_get_heap_max()},
                              {"malloc_in_use_bytes", usage.in_use},
                              {"malloc_free_bytes", usage.free},
                              {"malloc_peak_footprint_bytes", usage.footprint}};
        if (generator) {
            const auto memory = generator->memory();
            result["kv_cache_bytes"] = memory.cache_bytes;
            result["image_feature_bytes"] = memory.image_bytes;
        }
        auto categories = nlohmann::json::array();
        for (const auto& entry : kidi::core::memory_entries())
            categories.push_back({{"category", entry.category}, {"bytes", entry.bytes}, {"count", entry.count}});
        result["categories"] = std::move(categories);
        return result;
    });
}
}