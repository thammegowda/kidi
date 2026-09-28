#include "kidi/inference/generator.h"
#include "kidi/inference/transcriber.h"
#include "kidi/runtime/ynn/graph.h"

#include <nlohmann/json.hpp>
#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <iostream>

namespace {
using Clock = std::chrono::steady_clock;
using kidi::ops::require;
constexpr std::size_t CONTEXT = 9216;
constexpr std::size_t CHUNK = 32;
constexpr std::size_t OUTPUT_TOKENS = 64;

auto elapsed_ms(Clock::time_point start) -> double {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

auto rounded(double value) -> double { return std::round(value * 100000.0) / 100000.0; }

auto peak_rss_kib() -> long {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
#ifdef __APPLE__
    return usage.ru_maxrss / 1024;
#else
    return usage.ru_maxrss;
#endif
}

auto emit(nlohmann::json record, int threads) -> void {
    record["threads"] = threads;
    record["peak_rss_kib"] = peak_rss_kib();
    std::cout << record.dump() << std::endl;
}

auto benchmark_gemma(const std::filesystem::path& directory, int threads, int repeats) -> void {
    const auto started = Clock::now();
    auto generator = require(kidi::inference::Generator::load(directory, kidi::tensor::Device::cpu(), 0, 128, true));
    require(generator.configure_serving({1, 1, CONTEXT, CHUNK}));
    emit({{"stage", "load"},
          {"model", "gemma"},
          {"ms", rounded(elapsed_ms(started))},
          {"native_qat", generator.native_qat()}},
         threads);

    for (const int copies : {1, 16}) {
        std::string prompt;
        for (int copy = 0; copy < copies; ++copy)
            prompt +=
                "Binary search repeatedly halves a sorted search range. It is useful for lookup and boundary finding. ";
        prompt += "Explain the main idea and give practical examples.";
        const std::array messages{kidi::text::ChatMessage{"user", prompt}};
        for (int iteration = 0; iteration <= repeats; ++iteration) {
            kidi::inference::GenerationOptions options;
            options.maximum_new_tokens = OUTPUT_TOKENS;
            options.context_size = CONTEXT;
            options.prefill_chunk_size = CHUNK;
            options.stream_text = true;
            options.ignore_eos = true;
            require(generator.enqueue_chat(messages, options));
            bool completed = false;
            while (generator.pending_requests()) {
                auto step = require(generator.step());
                for (const auto& event : step.events) {
                    if (!event.completed) continue;
                    const auto& result = *event.completed;
                    const auto& stats = result.stats;
                    if (result.generation.token_ids.size() != OUTPUT_TOKENS || !stats.decode_ns || !stats.prefill_ns)
                        throw std::runtime_error("incomplete benchmark generation");
                    emit({{"stage", "generation"},
                          {"model", "gemma"},
                          {"workload", copies == 1 ? "short" : "long"},
                          {"iteration", iteration},
                          {"warmup", iteration == 0},
                          {"context_tokens", CONTEXT},
                          {"prefill_chunk", CHUNK},
                          {"prompt_tokens", stats.prompt_tokens},
                          {"output_tokens", result.generation.token_ids.size()},
                          {"prefill_ms", rounded(stats.prefill_ns / 1e6)},
                          {"prefill_tps", rounded(stats.prompt_tokens * 1e9 / stats.prefill_ns)},
                          {"decode_ms", rounded(stats.decode_ns / 1e6)},
                          {"decode_tokens", stats.decode_tokens},
                          {"decode_tps", rounded(stats.decode_tokens * 1e9 / stats.decode_ns)},
                          {"first_token_ms", rounded(stats.time_to_first_token_ns / 1e6)},
                          {"generation_ms", rounded(stats.generation_ns / 1e6)},
                          {"preparation_ms", rounded(stats.preparation_ns / 1e6)},
                          {"token_ids", result.generation.token_ids},
                          {"text", result.text}},
                         threads);
                    completed = true;
                }
            }
            if (!completed) throw std::runtime_error("missing generation completion");
        }
    }
}

auto benchmark_chat_turns(const std::filesystem::path& directory, int threads, int repeats) -> void {
    auto generator = require(kidi::inference::Generator::load(directory, kidi::tensor::Device::cpu(), 0, 128, true));
    require(generator.configure_serving({1, 1, CONTEXT, CHUNK}));
    kidi::inference::GenerationOptions options;
    options.maximum_new_tokens = 16;
    options.context_size = CONTEXT;
    options.prefill_chunk_size = CHUNK;
    options.stream_text = true;
    const auto run = [&](const auto& messages, std::size_t budget, const char* phase, int iteration) {
        options.prefix_cache_bytes = budget;
        require(generator.enqueue_chat(messages, options));
        std::optional<kidi::inference::TextGeneration> completed;
        std::string streamed;
        while (generator.pending_requests()) {
            auto step = require(generator.step());
            for (auto& event : step.events) {
                streamed += event.text;
                if (event.completed) completed = std::move(event.completed);
            }
        }
        if (!completed || streamed != completed->text) throw std::runtime_error("incomplete streamed chat turn");
        const auto& stats = completed->stats;
        emit({{"stage", "chat_turn"},
              {"phase", phase},
              {"iteration", iteration},
              {"prompt_tokens", stats.prompt_tokens},
              {"reused_prompt_tokens", stats.reused_prompt_tokens},
              {"prefix_cache_bytes", stats.prefix_cache_bytes},
              {"prefix_reserved_bytes", stats.prefix_reserved_bytes},
              {"prefill_ms", rounded(stats.prefill_ns / 1e6)},
              {"first_token_ms", rounded(stats.time_to_first_token_ns / 1e6)},
              {"generation_ms", rounded(stats.generation_ns / 1e6)},
              {"token_ids", completed->generation.token_ids}},
             threads);
        return std::move(*completed);
    };
    constexpr std::size_t BUDGET = 512 * 1024 * 1024;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        require(generator.configure_serving({1, 1, CONTEXT, CHUNK}));
        std::vector<kidi::text::ChatMessage> conversation{
            {"user",
             "Binary search repeatedly halves a sorted search range. It is useful for lookup and boundary finding. "
             "In a production application the array can contain duplicate values, and callers need the first matching "
             "index rather than an arbitrary match. Briefly explain the main idea."}};
        const auto first = run(conversation, BUDGET, "first", iteration);
        conversation.push_back({"assistant", first.text});
        conversation.push_back({"user", "How should I handle duplicate values?"});
        const auto cached = run(conversation, BUDGET, "cached", iteration);
        const auto fresh = run(conversation, 0, "uncached", iteration);
        if (!cached.stats.reused_prompt_tokens || cached.stats.prefix_reserved_bytes > BUDGET ||
            fresh.stats.reused_prompt_tokens || cached.generation.token_ids != fresh.generation.token_ids ||
            cached.text != fresh.text)
            throw std::runtime_error("cached chat did not preserve tokens or the memory bound");
    }
}

auto benchmark_image(const std::filesystem::path& directory, const std::filesystem::path& image, int threads) -> void {
    const auto start = Clock::now();
    auto generator = require(kidi::inference::Generator::load(directory, kidi::tensor::Device::cpu(), 0, 128, true));
    require(generator.configure_serving({1, 1, CONTEXT, CHUNK}));
    emit({{"stage", "load"}, {"vision", generator.vision_supported()}, {"ms", rounded(elapsed_ms(start))}}, threads);
    kidi::inference::GenerationOptions options;
    options.maximum_new_tokens = 48;
    options.context_size = CONTEXT;
    options.prefill_chunk_size = CHUNK;
    options.prefix_cache_bytes = 512 * 1024 * 1024;
    options.stream_text = true;
    std::vector messages{kidi::text::ChatMessage{"user", "What is in this photo? Give a short answer.", {image}}};
    for (int turn = 0; turn < 2; ++turn) {
        const auto encoded = Clock::now();
        require(generator.enqueue_chat(messages, options));
        emit({{"stage", "image_encoded"}, {"turn", turn}, {"ms", rounded(elapsed_ms(encoded))}}, threads);
        while (generator.pending_requests()) {
            const auto step = require(generator.step());
            for (const auto& event : step.events)
                if (event.completed) {
                    if (turn && !event.completed->stats.reused_prompt_tokens)
                        throw std::runtime_error("image follow-up lost KV reuse");
                    emit({{"stage", "answer"},
                          {"turn", turn},
                          {"text", event.completed->text},
                          {"reused_prompt_tokens", event.completed->stats.reused_prompt_tokens},
                          {"ms", rounded(elapsed_ms(encoded))}},
                         threads);
                    messages.push_back({"assistant", event.completed->text});
                }
        }
        messages.push_back({"user", "What color is it? Answer briefly."});
    }
}

auto benchmark_whisper(const std::filesystem::path& directory, const std::filesystem::path& wav, int threads,
                       int repeats) -> void {
    const auto waveform = require(kidi::audio::load_wav(wav));
    if (waveform.sample_rate != 16000 || waveform.samples.size() < 32000 || waveform.samples.size() > 480000)
        throw std::runtime_error("benchmark WAV must be 2-30 seconds at 16 kHz");
    const auto started = Clock::now();
    auto transcriber = require(kidi::inference::Transcriber::load(directory));
    emit({{"stage", "load"}, {"model", "whisper"}, {"ms", rounded(elapsed_ms(started))}}, threads);
    for (const auto size : {std::size_t{32000}, waveform.samples.size()}) {
        for (int iteration = 0; iteration <= repeats; ++iteration) {
            const auto before = Clock::now();
            const auto result = require(transcriber.transcribe(std::span(waveform.samples).first(size), 16000));
            const auto milliseconds = elapsed_ms(before);
            const auto seconds = size / 16000.0;
            emit({{"stage", "transcription"},
                  {"model", "whisper"},
                  {"workload", size == 32000 ? "draft_2s" : "final"},
                  {"iteration", iteration},
                  {"warmup", iteration == 0},
                  {"audio_seconds", rounded(seconds)},
                  {"wall_ms", rounded(milliseconds)},
                  {"real_time_factor", rounded(milliseconds / 1000 / seconds)},
                  {"feature_ms", rounded(result.stats.feature_ns / 1e6)},
                  {"encode_ms", rounded(result.stats.encode_ns / 1e6)},
                  {"decode_ms", rounded(result.stats.decode_ns / 1e6)},
                  {"token_ids", result.token_ids},
                  {"language", result.language},
                  {"text", result.text}},
                 threads);
        }
    }
}
} // namespace

auto main(int argc, char** argv) -> int {
    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
        std::cout << "usage: kidi_android_baseline gemma|chat|whisper|image MODEL THREADS REPEATS [MEDIA]\n";
        return 0;
    }
    try {
        if (argc < 5) throw std::runtime_error("missing benchmark arguments");
        const std::string mode(argv[1]);
        const auto threads = std::stoi(argv[3]), repeats = std::stoi(argv[4]);
        if (threads < 1 || threads > 8 || repeats < 1 || repeats > 20 ||
            !(((mode == "gemma" || mode == "chat") && argc == 5) ||
              ((mode == "whisper" || mode == "image") && argc == 6)))
            throw std::runtime_error("invalid mode, thread count, repeat count, or WAV argument");
        kidi::runtime::ynn::set_thread_count(threads);
        require(kidi::runtime::ynn::reserve_thread_pool(threads));
        emit({{"stage", "runtime"},
              {"arch", kidi::runtime::ynn::supported_arch_names()},
              {"warm_repeats", repeats},
              {"build", "release-native-cpu"}},
             threads);
        if (mode == "gemma")
            benchmark_gemma(argv[2], threads, repeats);
        else if (mode == "chat")
            benchmark_chat_turns(argv[2], threads, repeats);
        else if (mode == "image")
            benchmark_image(argv[2], argv[5], threads);
        else
            benchmark_whisper(argv[2], argv[5], threads, repeats);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}