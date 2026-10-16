#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "kidi/inference/synthesizer.h"
#include "kidi/runtime/ynn/graph.h"

namespace {
auto positive_integer(std::string_view value, std::string_view name) -> std::size_t {
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size() || !result)
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    return result;
}

auto nonnegative_integer(std::string_view value, std::string_view name) -> std::size_t {
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size())
        throw std::invalid_argument(std::string(name) + " must be a nonnegative integer");
    return result;
}

auto elapsed(std::chrono::steady_clock::time_point start) -> std::uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}

auto token_hash(std::span<const std::int32_t> tokens) -> std::uint64_t {
    std::uint64_t result = 1469598103934665603ULL;
    for (const auto token : tokens) {
        result ^= static_cast<std::uint32_t>(token);
        result *= 1099511628211ULL;
    }
    return result;
}

auto text_chunks(std::string_view argument) -> std::vector<std::string> {
    if (!argument.starts_with('@')) return {std::string(argument)};
    std::ifstream input(std::filesystem::path(argument.substr(1)), std::ios::binary);
    if (!input) throw std::runtime_error("cannot open text file: " + std::string(argument.substr(1)));
    std::string content{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad()) throw std::runtime_error("cannot read text file: " + std::string(argument.substr(1)));
    if (content.empty()) throw std::invalid_argument("text file is empty");
    return {std::move(content)};
}
} // namespace

auto main(int argc, char** argv) -> int {
    if (argc < 2 || argc > 7) {
        std::cerr << "usage: kidi_tts_bench MODEL [cpu|mps|vulkan|npu] [REPETITIONS] [THREADS] [TEXT|@FILE] [STEPS]\n";
        return 2;
    }
    try {
        const std::filesystem::path model(argv[1]);
        const std::string_view backend = argc >= 3 ? argv[2] : "cpu";
        const auto device = backend == "cpu"      ? kidi::tensor::Device::cpu()
                            : backend == "mps"    ? kidi::tensor::Device::apple_gpu()
                            : backend == "vulkan" ? kidi::tensor::Device::vulkan()
                            : backend == "npu"
                                ? kidi::tensor::Device::qualcomm_npu()
                                : throw std::invalid_argument("backend must be cpu, mps, vulkan, or npu");
        const auto repetitions = argc >= 4 ? nonnegative_integer(argv[3], "repetitions") : 3;
        const auto available = std::max(1U, std::thread::hardware_concurrency());
        const auto threads = argc >= 5 ? positive_integer(argv[4], "threads") : std::min<std::size_t>(4, available);
        const auto chunks = text_chunks(argc >= 6 ? argv[5] : "Hello from Kidi.");
        const auto steps = argc >= 7 ? positive_integer(argv[6], "steps") : 32;
        kidi::runtime::ynn::set_thread_count(threads);

        const auto load_start = std::chrono::steady_clock::now();
        auto synthesizer = kidi::inference::Synthesizer::load(model, device);
        if (!synthesizer) throw std::runtime_error(synthesizer.error().message);
        const auto load_ns = elapsed(load_start);
        std::size_t planned_chunks = 0, maximum_bytes = 0, maximum_words = 0;
        for (const auto& input : chunks) {
            auto sentences = synthesizer->segment(input, "English");
            if (!sentences) throw std::runtime_error(sentences.error().message);
            planned_chunks += sentences->size();
            for (const auto& sentence : *sentences) {
                maximum_bytes = std::max(maximum_bytes, sentence.size());
                std::size_t words = 0;
                bool inside = false;
                for (const auto value : sentence) {
                    const auto space = std::isspace(static_cast<unsigned char>(value));
                    if (!space && !inside) ++words;
                    inside = !space;
                }
                maximum_words = std::max(maximum_words, words);
            }
        }
        std::cerr << "kidi_tts_plan|chunks=" << planned_chunks << "|maximum_bytes=" << maximum_bytes
                  << "|maximum_words=" << maximum_words << '\n';

        std::vector<std::int32_t> reference;
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
            kidi::inference::SynthesisOptions options;
            options.language = "English";
            options.steps = steps;
            const auto start = std::chrono::steady_clock::now();
            std::vector<std::int32_t> tokens;
            std::uint64_t generation_ns = 0, decode_ns = 0, preparation_ns = 0;
            std::size_t audio_tokens = 0, samples = 0, synthesized_chunks = 0;
            std::uint32_t sample_rate = 0;
            options.on_chunk = [repetition](std::size_t index, std::size_t total) {
                std::cerr << "kidi_tts_chunk|repetition=" << repetition << "|index=" << index << "|total=" << total
                          << '\n';
            };
            for (std::size_t index = 0; index < chunks.size(); ++index) {
                auto speech = synthesizer->synthesize(chunks[index], options);
                if (!speech) throw std::runtime_error(speech.error().message);
                generation_ns += speech->stats.generation_ns;
                decode_ns += speech->stats.decode_ns;
                preparation_ns += speech->stats.preparation_ns;
                audio_tokens += speech->stats.audio_tokens;
                synthesized_chunks += speech->stats.chunks;
                samples += speech->samples.size();
                sample_rate = speech->sample_rate;
                tokens.insert(tokens.end(), speech->tokens.begin(), speech->tokens.end());
                options.voice_prompt = speech->voice_prompt;
            }
            const auto total_ns = elapsed(start);
            if (repetition == 0)
                reference = tokens;
            else if (tokens != reference)
                throw std::runtime_error("repeated deterministic synthesis changed audio tokens");
            if (repetition == 0) {
                if (const auto* output = std::getenv("KIDI_TTS_TOKENS_OUT"); output && *output) {
                    std::ofstream stream(output);
                    if (!stream) throw std::runtime_error("cannot create token output: " + std::string(output));
                    for (std::size_t index = 0; index < tokens.size(); ++index)
                        stream << (index ? "," : "") << tokens[index];
                    stream << '\n';
                    if (!stream) throw std::runtime_error("cannot write token output: " + std::string(output));
                }
            }
            std::cout << "kidi_tts_bench|backend=" << synthesizer->execution() << "|threads=" << threads
                      << "|steps=" << steps << "|repetition=" << repetition << "|chunks=" << synthesized_chunks
                      << "|load_ns=" << load_ns << "|total_ns=" << total_ns << "|generation_ns=" << generation_ns
                      << "|decode_ns=" << decode_ns << "|preparation_ns=" << preparation_ns
                      << "|audio_tokens=" << audio_tokens << "|samples=" << samples << "|sample_rate=" << sample_rate
                      << "|token_hash=" << token_hash(tokens) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
