#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/platform/runtime.h>
#include <nlohmann/json.hpp>

namespace {
using Clock = std::chrono::steady_clock;
using executorch::extension::Module;
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

auto execute(Module& module, std::vector<std::int64_t>& ids, std::vector<std::int64_t>& positions) -> std::int64_t {
    const auto length = static_cast<executorch::aten::SizesType>(ids.size());
    auto tokens = executorch::extension::from_blob(ids.data(), {1, length}, executorch::aten::ScalarType::Long);
    auto offsets = executorch::extension::from_blob(positions.data(), {length}, executorch::aten::ScalarType::Long);
    auto result = module.execute("forward", {*tokens, *offsets});
    if (!result.ok()) {
        throw std::runtime_error("ExecuTorch forward failed: " + std::to_string(static_cast<int>(result.error())));
    }
    if (result->size() != 1 || !result->front().isTensor()) throw std::runtime_error("expected one logits tensor");
    const auto& logits = result->front().toTensor();
    if (logits.scalar_type() != executorch::aten::ScalarType::Float || logits.numel() != 262144)
        throw std::runtime_error("expected last-token FP32 Gemma 4 E2B logits");
    const auto* values = logits.const_data_ptr<float>();
    for (std::size_t index = 0; index < logits.numel(); ++index) {
        if (!std::isfinite(values[index])) throw std::runtime_error("non-finite logits");
    }
    return std::max_element(values, values + logits.numel()) - values;
}
} // namespace

auto main(int argc, char** argv) -> int {
    try {
        if (argc != 3) throw std::invalid_argument("usage: arm_gemma4_bench MODEL.pte PROTOCOL.json");
        std::ifstream stream(argv[2]);
        if (!stream) throw std::runtime_error("cannot open benchmark protocol");
        const auto protocol = json::parse(stream);
        const auto mode = protocol.value("mode", std::string("speed"));
        const auto context = protocol.at("context_tokens").get<std::size_t>();
        const auto threads = protocol.at("threads").get<std::size_t>();
        const auto warmups = protocol.value("warmups", std::size_t{1});
        if (context != 1024 || !warmups || warmups > 20 || !threads || threads > std::thread::hardware_concurrency())
            throw std::invalid_argument("invalid Gemma benchmark protocol");

        executorch::runtime::runtime_init();
        auto* pool = executorch::extension::threadpool::get_threadpool();
        // Configure before any model load or worker jobs, as in Arm's LLM-Runner.
        if (!pool->_unsafe_reset_threadpool(static_cast<std::uint32_t>(threads)) || pool->get_thread_count() != threads)
            throw std::runtime_error("ExecuTorch failed to configure the requested thread count");
        emit({{"stage", "runtime"},
              {"device", "cpu:0"},
              {"backend", "XNNPACK"},
              {"kleidiai_enabled", true},
              {"threads", pool->get_thread_count()}});

        const auto loading = Clock::now();
        Module module(argv[1], Module::LoadMode::Mmap);
        const auto loaded = module.load_method("forward");
        if (loaded != executorch::runtime::Error::Ok)
            throw std::runtime_error("ExecuTorch load failed: " + std::to_string(static_cast<int>(loaded)));
        emit({{"stage", "load"}, {"ms", seconds(loading, Clock::now()) * 1000}});

        if (mode == "eval") {
            const auto inputs_list = protocol.at("input_token_ids_list").get<std::vector<std::vector<std::int64_t>>>();
            std::vector<std::size_t> item_ids(inputs_list.size());
            if (protocol.contains("item_ids"))
                item_ids = protocol.at("item_ids").get<std::vector<std::size_t>>();
            else
                std::iota(item_ids.begin(), item_ids.end(), std::size_t{0});
            const auto max_outputs = protocol.at("max_output_tokens").get<std::size_t>();
            if (inputs_list.empty() || !max_outputs || item_ids.size() != inputs_list.size())
                throw std::invalid_argument("invalid Gemma eval protocol");
            auto run_seq = [&](const std::vector<std::int64_t>& prompt_ids, std::size_t limit) {
                auto ids = prompt_ids;
                std::vector<std::int64_t> positions(ids.size());
                std::iota(positions.begin(), positions.end(), 0);
                std::vector<std::int64_t> generated;
                generated.reserve(limit);
                auto tok = execute(module, ids, positions);
                for (std::size_t index = 1; index <= limit; ++index) {
                    if (tok == 1 || tok == 106) break;
                    generated.push_back(tok);
                    if (index == limit) break;
                    ids.assign(1, tok);
                    positions.assign(1, static_cast<std::int64_t>(prompt_ids.size() + index - 1));
                    tok = execute(module, ids, positions);
                }
                return generated;
            };
            for (std::size_t w = 0; w < warmups; ++w) (void)run_seq(inputs_list.front(), std::min<std::size_t>(8, max_outputs));
            std::vector<std::vector<std::int64_t>> all_token_ids;
            all_token_ids.reserve(inputs_list.size());
            std::size_t total_prompt_tokens = 0, total_output_tokens = 0;
            const auto eval_started = Clock::now();
            for (std::size_t idx = 0; idx < inputs_list.size(); ++idx) {
                const auto& prompt_ids = inputs_list[idx];
                const auto item_started = Clock::now();
                total_prompt_tokens += prompt_ids.size();
                auto gen = run_seq(prompt_ids, max_outputs);
                const auto item_seconds = seconds(item_started, Clock::now());
                total_output_tokens += gen.size();
                emit({{"stage", "eval_item"},
                      {"id", item_ids[idx]},
                      {"seconds", item_seconds},
                      {"prompt_tokens", prompt_ids.size()},
                      {"output_tokens", gen.size()},
                      {"token_ids", gen}});
                all_token_ids.push_back(std::move(gen));
            }
            emit({{"stage", "eval"},
                  {"items", inputs_list.size()},
                  {"eval_seconds", seconds(eval_started, Clock::now())},
                  {"prompt_tokens", total_prompt_tokens},
                  {"output_tokens", total_output_tokens},
                  {"token_ids", all_token_ids}});
            return 0;
        }

        const auto prompt = protocol.at("input_token_ids").get<std::vector<std::int64_t>>();
        const auto outputs = protocol.at("output_tokens").get<std::size_t>();
        const auto repeats = protocol.at("repeats").get<std::size_t>();
        if (prompt.empty() || outputs < 2 || prompt.size() + outputs > context || !repeats || repeats > 20 ||
            std::ranges::any_of(prompt, [](auto id) { return id < 0 || id >= 262144; }))
            throw std::invalid_argument("invalid Gemma benchmark protocol");

        for (std::size_t iteration = 0; iteration < warmups + repeats; ++iteration) {
            auto ids = prompt;
            std::vector<std::int64_t> positions(ids.size());
            std::iota(positions.begin(), positions.end(), 0);
            std::vector<std::int64_t> generated;
            generated.reserve(outputs);
            const auto submitted = Clock::now();
            generated.push_back(execute(module, ids, positions));
            const auto first = Clock::now();
            for (std::size_t index = 1; index < outputs; ++index) {
                ids.assign(1, generated.back());
                positions.assign(1, static_cast<std::int64_t>(prompt.size() + index - 1));
                generated.push_back(execute(module, ids, positions));
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
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Arm Gemma benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
