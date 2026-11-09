#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <vector>

#include "llama.h"
#include <nlohmann/json.hpp>

namespace {
using Clock = std::chrono::steady_clock;
using nlohmann::json;

auto seconds(Clock::time_point begin, Clock::time_point end) -> double {
    return std::chrono::duration<double>(end - begin).count();
}

auto peak_rss_kib() -> long {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
#ifdef __APPLE__
    return usage.ru_maxrss / 1024;
#else
    return usage.ru_maxrss;
#endif
}

auto emit(json record) -> void {
    record["peak_rss_kib"] = peak_rss_kib();
    std::cout << record.dump() << std::endl;
}

auto greedy_select(llama_context* context, std::size_t vocabulary) -> llama_token {
    const float* values = llama_get_logits_ith(context, -1);
    if (values == nullptr) throw std::runtime_error("llama_get_logits_ith returned null");
    for (std::size_t index = 0; index < vocabulary; ++index) {
        if (!std::isfinite(values[index])) throw std::runtime_error("non-finite logits");
    }
    return static_cast<llama_token>(std::max_element(values, values + vocabulary) - values);
}
} // namespace

auto main(int argc, char** argv) -> int {
    try {
        if (argc != 3) throw std::invalid_argument("usage: llamacpp_gemma4_bench MODEL.gguf PROTOCOL.json");
        std::ifstream stream(argv[2]);
        if (!stream) throw std::runtime_error("cannot open benchmark protocol");
        const auto protocol = json::parse(stream);
        const auto mode = protocol.value("mode", std::string("speed"));
        const auto context = protocol.at("context_tokens").get<std::size_t>();
        const auto threads = protocol.at("threads").get<std::size_t>();
        const auto warmups = protocol.value("warmups", std::size_t{1});
        if (context != 1024 || !warmups || warmups > 20 || !threads || threads > std::thread::hardware_concurrency())
            throw std::invalid_argument("invalid Gemma benchmark protocol");

        ggml_backend_load_all();
        emit({{"stage", "runtime"},
              {"device", "cpu:0"},
              {"backend", "llama.cpp+KleidiAI"},
              {"kleidiai_enabled", true},
              {"threads", threads}});

        const auto loading = Clock::now();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        auto* model = llama_model_load_from_file(argv[1], model_params);
        if (model == nullptr) throw std::runtime_error("llama_model_load_from_file failed");

        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = static_cast<std::uint32_t>(context);
        ctx_params.n_batch = static_cast<std::uint32_t>(context);
        ctx_params.n_ubatch = static_cast<std::uint32_t>(context);
        ctx_params.n_threads = static_cast<std::int32_t>(threads);
        ctx_params.n_threads_batch = static_cast<std::int32_t>(threads);
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        auto* ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) {
            llama_model_free(model);
            throw std::runtime_error("llama_init_from_model failed");
        }
        const auto* vocab = llama_model_get_vocab(model);
        const auto vocabulary = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));
        if (vocabulary != 262144) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("unexpected Gemma 4 E2B vocabulary size");
        }
        emit({{"stage", "load"}, {"ms", seconds(loading, Clock::now()) * 1000}});

        if (mode == "eval") {
            const auto inputs_list = protocol.at("input_token_ids_list").get<std::vector<std::vector<llama_token>>>();
            const auto max_outputs = protocol.at("max_output_tokens").get<std::size_t>();
            if (inputs_list.empty() || !max_outputs) throw std::invalid_argument("invalid Gemma eval protocol");
            auto run_seq = [&](const std::vector<llama_token>& prompt_ids, std::size_t limit) {
                llama_memory_clear(llama_get_memory(ctx), true);
                auto ids = prompt_ids;
                std::vector<llama_token> generated;
                generated.reserve(limit);
                if (llama_decode(ctx, llama_batch_get_one(ids.data(), static_cast<std::int32_t>(ids.size()))) != 0)
                    throw std::runtime_error("llama_decode prefill failed");
                auto tok = greedy_select(ctx, vocabulary);
                for (std::size_t index = 1; index <= limit; ++index) {
                    if (tok == 1 || tok == 106) break;
                    generated.push_back(tok);
                    if (index == limit) break;
                    if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0)
                        throw std::runtime_error("llama_decode step failed");
                    tok = greedy_select(ctx, vocabulary);
                }
                return generated;
            };
            for (std::size_t w = 0; w < warmups; ++w) (void)run_seq(inputs_list.front(), std::min<std::size_t>(8, max_outputs));
            std::vector<std::vector<llama_token>> all_token_ids;
            all_token_ids.reserve(inputs_list.size());
            std::size_t total_prompt_tokens = 0, total_output_tokens = 0;
            const auto eval_started = Clock::now();
            for (const auto& prompt_ids : inputs_list) {
                total_prompt_tokens += prompt_ids.size();
                auto gen = run_seq(prompt_ids, max_outputs);
                total_output_tokens += gen.size();
                all_token_ids.push_back(std::move(gen));
            }
            emit({{"stage", "eval"},
                  {"items", inputs_list.size()},
                  {"eval_seconds", seconds(eval_started, Clock::now())},
                  {"prompt_tokens", total_prompt_tokens},
                  {"output_tokens", total_output_tokens},
                  {"token_ids", all_token_ids}});
            llama_free(ctx);
            llama_model_free(model);
            return 0;
        }

        const auto prompt = protocol.at("input_token_ids").get<std::vector<llama_token>>();
        const auto outputs = protocol.at("output_tokens").get<std::size_t>();
        const auto repeats = protocol.at("repeats").get<std::size_t>();
        if (prompt.empty() || outputs < 2 || prompt.size() + outputs > context || !repeats || repeats > 20 ||
            std::ranges::any_of(prompt, [](auto id) { return id < 0 || id >= 262144; }))
            throw std::invalid_argument("invalid Gemma benchmark protocol");

        for (std::size_t iteration = 0; iteration < warmups + repeats; ++iteration) {
            llama_memory_clear(llama_get_memory(ctx), true);
            auto ids = prompt;
            std::vector<llama_token> generated;
            generated.reserve(outputs);
            const auto submitted = Clock::now();
            if (llama_decode(ctx, llama_batch_get_one(ids.data(), static_cast<std::int32_t>(ids.size()))) != 0) {
                llama_free(ctx);
                llama_model_free(model);
                throw std::runtime_error("llama_decode prefill failed");
            }
            generated.push_back(greedy_select(ctx, vocabulary));
            const auto first = Clock::now();
            for (std::size_t index = 1; index < outputs; ++index) {
                auto token = generated.back();
                if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
                    llama_free(ctx);
                    llama_model_free(model);
                    throw std::runtime_error("llama_decode step failed");
                }
                generated.push_back(greedy_select(ctx, vocabulary));
            }
            const auto last = Clock::now();
            const auto prefill = seconds(submitted, first), decode = seconds(first, last);
            if (prefill <= 0 || decode <= 0) throw std::runtime_error("invalid benchmark timing");
            emit({{"stage", "generation"},
                  {"iteration", iteration},
                  {"warmup", iteration < warmups},
                  {"input_token_ids", prompt},
                  {"token_ids", generated},
                  {"prompt_tokens", prompt.size()},
                  {"output_tokens", generated.size()},
                  {"decode_tokens", generated.size() - 1},
                  {"reused_prompt_tokens", 0},
                  {"first_token_seconds", prefill},
                  {"decode_seconds", decode},
                  {"prefill_through_first_token_tps", prompt.size() / prefill},
                  {"decode_interval_tps", (outputs - 1) / decode}});
        }
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "llama.cpp Gemma benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
