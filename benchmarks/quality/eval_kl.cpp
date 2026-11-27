// Developer quality check for quantized precisions: record a reference precision's teacher-forced top-10 logits with
// --save-ref, then score another precision against them with --ref (KL@10, EAR@10, top-1 agreement, perplexity).
#include "kidi/checkpoint/hub.h"
#include "kidi/cli/argparse.h"
#include "kidi/core/precision.h"
#include "kidi/inference/generator.h"
#include "kidi/ops/context.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/tensor/device.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace kidi;
using ops::require;

constexpr std::size_t TOPK = 10;

struct TopKRecord {
    std::uint32_t ids[TOPK];
    float logits[TOPK];
    float lse;
};

auto parse_device(std::string_view backend) -> tensor::Device {
    if (backend == "ynnpack") return tensor::Device::cpu();
    if (backend == "mps") return tensor::Device::apple_gpu();
    if (const auto device = tensor::parse_cuda_device(backend)) return *device;
    throw std::invalid_argument("unknown backend; expected ynnpack, mps, or cuda[:INDEX]");
}

auto read_sequences(std::istream& input, const text::Tokenizer& tokenizer, std::size_t context_limit)
    -> std::vector<std::vector<std::int32_t>> {
    std::vector<std::vector<std::int32_t>> sequences;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::string formatted = line;
        if (line.front() == '{') {
            const auto parsed = nlohmann::ordered_json::parse(line);
            if (parsed.contains("messages"))
                formatted = require(tokenizer.format_chat(
                    parsed["messages"], parsed.value("template_args", nlohmann::ordered_json::object())));
            else
                formatted = parsed.value("text", std::string{});
        }
        auto ids = tokenizer.encode(formatted);
        if (!ids || ids->size() < 2) continue;
        if (context_limit && ids->size() > context_limit) ids->resize(context_limit);
        sequences.push_back(std::move(*ids));
    }
    return sequences;
}

auto read_reference(const std::filesystem::path& path) -> std::vector<TopKRecord> {
    std::ifstream file(path, std::ios::binary);
    std::uint64_t count = 0;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::vector<TopKRecord> records(static_cast<std::size_t>(count));
    file.read(reinterpret_cast<char*>(records.data()), static_cast<std::streamsize>(count * sizeof(TopKRecord)));
    if (!file) throw std::runtime_error("cannot read reference logit file: " + path.string());
    return records;
}

auto write_reference(const std::filesystem::path& path, const std::vector<TopKRecord>& records) -> void {
    std::ofstream file(path, std::ios::binary);
    const std::uint64_t count = records.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    file.write(reinterpret_cast<const char*>(records.data()), static_cast<std::streamsize>(count * sizeof(TopKRecord)));
    if (!file) throw std::runtime_error("cannot write reference logit file: " + path.string());
}

auto evaluate(const cli::Namespace& arguments) -> int {
    auto model = arguments.get<std::filesystem::path>("model");
    if (model.string().starts_with('@')) model = require(checkpoint::Hub().resolve(model.string()));
    const auto precision = core::parse_precision(arguments.get<std::string>("precision"));
    const auto kv_cache_precision = core::parse_kv_cache_precision(arguments.get<std::string>("kv_cache_precision"));
    if (!precision || !kv_cache_precision) throw std::invalid_argument("invalid precision or KV-cache precision");
    const auto device = parse_device(arguments.get<std::string>("backend"));
    runtime::ynn::set_thread_count(arguments.get<std::int32_t>("threads"));
    const auto& save_ref = arguments.get<std::filesystem::path>("save_ref");
    const auto& ref = arguments.get<std::filesystem::path>("ref");

    const auto load_start = std::chrono::steady_clock::now();
    auto generator = require(inference::Generator::load(
        model, {.device = device, .precision = *precision, .kv_cache_precision = *kv_cache_precision}));
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();

    const auto& input_path = arguments.get<std::string>("input");
    std::ifstream input_file;
    if (input_path != "-") {
        input_file.open(input_path);
        if (!input_file) throw std::runtime_error("cannot open input file: " + input_path);
    }
    const auto sequences = read_sequences(input_path == "-" ? std::cin : input_file, generator.tokenizer(),
                                          arguments.get<std::size_t>("context_size"));
    if (sequences.empty()) throw std::runtime_error("no valid prompts");
    const auto references = ref.empty() ? std::vector<TopKRecord>{} : read_reference(ref);

    std::vector<TopKRecord> saved;
    std::size_t cursor = 0, positions = 0, nll_positions = 0, top1_matches = 0;
    double kl_sum = 0, ear_sum = 0, nll_sum = 0;
    const auto eval_start = std::chrono::steady_clock::now();
    for (const auto& ids : sequences) {
        const auto logits = require(generator.prefill_logits(ids));
        const auto vocab = logits.size(-1);
        const auto data = require(logits.data<float>());
        std::vector<std::uint32_t> order(save_ref.empty() ? 0 : vocab);
        std::iota(order.begin(), order.end(), 0U);
        for (std::size_t t = 0; t < ids.size(); ++t) {
            const float* row = data.data() + t * vocab;
            const auto argmax = static_cast<std::uint32_t>(std::max_element(row, row + vocab) - row);
            const float max_logit = row[argmax];
            double sum_exp = 0;
            for (std::size_t v = 0; v < vocab; ++v) sum_exp += std::exp(static_cast<double>(row[v] - max_logit));
            const double lse = max_logit + std::log(sum_exp);
            if (t > 0 && t + 1 < ids.size() && static_cast<std::size_t>(ids[t + 1]) < vocab) {
                nll_sum += lse - row[ids[t + 1]];
                ++nll_positions;
            }
            if (!save_ref.empty()) {
                const auto k = std::min(TOPK, vocab);
                std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                                  [row](std::uint32_t a, std::uint32_t b) { return row[a] > row[b]; });
                TopKRecord record{};
                record.lse = static_cast<float>(lse);
                for (std::size_t i = 0; i < TOPK; ++i) {
                    record.ids[i] = i < k ? order[i] : 0U;
                    record.logits[i] = i < k ? row[order[i]] : -1e30F;
                }
                saved.push_back(record);
            }
            if (!references.empty()) {
                if (cursor >= references.size())
                    throw std::runtime_error("reference logit file has fewer tokens than the input prompts");
                const auto& reference = references[cursor++];
                // Position 0 is unconditioned (<bos>), so it is excluded like the perplexity sum.
                if (t > 0) {
                    double kl = 0, ear = 0, p_sum = 0;
                    for (std::size_t i = 0; i < TOPK; ++i) {
                        const double log_p = static_cast<double>(reference.logits[i]) - reference.lse;
                        const double log_q = static_cast<double>(row[reference.ids[i]]) - lse;
                        const double p = std::exp(log_p);
                        p_sum += p;
                        kl += p * (log_p - log_q);
                        ear += std::min(p, std::exp(log_q));
                    }
                    kl_sum += std::max(0.0, kl);
                    ear_sum += p_sum > 1e-12 ? ear / p_sum : 0.0;
                    top1_matches += argmax == reference.ids[0];
                }
            }
            if (t > 0) ++positions;
        }
    }
    const double eval_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - eval_start).count();
    if (!save_ref.empty()) write_reference(save_ref, saved);

    const double ppl = nll_positions ? std::exp(nll_sum / static_cast<double>(nll_positions)) : 0.0;
    std::cout << std::fixed << std::setprecision(4) << "precision=" << core::to_string(generator.precision())
              << " | tokens=" << positions << " | load=" << std::setprecision(2) << load_s << "s | prefill=" << eval_s
              << "s | ppl=" << std::setprecision(3) << ppl;
    if (!references.empty() && positions) {
        const auto count = static_cast<double>(positions);
        std::cout << " | KL@10=" << std::setprecision(4) << kl_sum / count << " | EAR@10=" << std::setprecision(2)
                  << 100.0 * ear_sum / count << "% | top1=" << 100.0 * static_cast<double>(top1_matches) / count << "%";
    }
    std::cout << '\n';
    return 0;
}
} // namespace

auto main(int argc, char** argv) -> int {
    kidi::cli::ArgumentParser parser("kidi_eval_kl",
                                     "Teacher-forced KL@10, EAR@10, top-1 agreement, and perplexity of a precision "
                                     "against reference logits recorded with --save-ref.");
    parser.add_argument("-m", "--model")
        .type<std::filesystem::path>()
        .required()
        .metavar("DIR")
        .help("model directory or local @owner/model hub reference");
    parser.add_argument("-i", "--in")
        .dest("input")
        .default_value(std::string("-"))
        .metavar("FILE")
        .help("JSONL ({\"messages\": ...} or {\"text\": ...}) or plain-text prompts, one per line; '-' for stdin");
    parser.add_argument("--save-ref")
        .dest("save_ref")
        .type<std::filesystem::path>()
        .default_value(std::filesystem::path{})
        .metavar("FILE")
        .help("write top-10 reference logits to FILE");
    parser.add_argument("--ref")
        .type<std::filesystem::path>()
        .default_value(std::filesystem::path{})
        .metavar("FILE")
        .help("score against top-10 reference logits from FILE");
    parser.add_argument("--backend").default_value(std::string("ynnpack")).help("ynnpack, mps, or cuda[:INDEX]");
    parser.add_argument("--precision")
        .default_value(std::string("checkpoint"))
        .help("checkpoint, bf16, q8, q4, vq3d-hybrid, or the QAT oracle qat-fp32");
    parser.add_argument("--kv-cache-precision")
        .dest("kv_cache_precision")
        .default_value(std::string("auto"))
        .help("KV-cache storage policy");
    parser.add_argument("--context-size")
        .dest("context_size")
        .default_value<std::size_t>(0)
        .help("truncate prompts to this many tokens (0 keeps them whole)");
    parser.add_argument("-j", "--threads").default_value<std::int32_t>(4).help("CPU worker threads");
    try {
        const auto arguments = parser.parse_args(argc, argv);
        if (arguments.should_exit()) {
            std::cout << arguments.output();
            return 0;
        }
        return evaluate(arguments);
    } catch (const kidi::cli::ParseError& error) {
        std::cerr << error.usage() << "error: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
