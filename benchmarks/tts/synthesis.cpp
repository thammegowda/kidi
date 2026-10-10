#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
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

auto elapsed(std::chrono::steady_clock::time_point start) -> std::uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}
} // namespace

auto main(int argc, char** argv) -> int {
    if (argc < 2 || argc > 6) {
        std::cerr << "usage: kidi_tts_bench MODEL [cpu|vulkan] [REPETITIONS] [THREADS] [TEXT]\n";
        return 2;
    }
    try {
        const std::filesystem::path model(argv[1]);
        const std::string_view backend = argc >= 3 ? argv[2] : "cpu";
        const auto device = backend == "cpu"      ? kidi::tensor::Device::cpu()
                            : backend == "vulkan" ? kidi::tensor::Device::vulkan()
                                                  : throw std::invalid_argument("backend must be cpu or vulkan");
        const auto repetitions = argc >= 4 ? positive_integer(argv[3], "repetitions") : 3;
        const auto available = std::max(1U, std::thread::hardware_concurrency());
        const auto threads = argc >= 5 ? positive_integer(argv[4], "threads") : std::min<std::size_t>(4, available);
        const std::string_view text = argc >= 6 ? argv[5] : "Hello from Kidi.";
        kidi::runtime::ynn::set_thread_count(threads);

        const auto load_start = std::chrono::steady_clock::now();
        auto synthesizer = kidi::inference::Synthesizer::load(model, device);
        if (!synthesizer) throw std::runtime_error(synthesizer.error().message);
        const auto load_ns = elapsed(load_start);

        std::vector<std::int32_t> reference;
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
            kidi::inference::SynthesisOptions options;
            options.language = "English";
            const auto start = std::chrono::steady_clock::now();
            auto speech = synthesizer->synthesize(text, options);
            if (!speech) throw std::runtime_error(speech.error().message);
            const auto total_ns = elapsed(start);
            if (repetition == 0)
                reference = speech->tokens;
            else if (speech->tokens != reference)
                throw std::runtime_error("repeated deterministic synthesis changed audio tokens");
            std::cout << "kidi_tts_bench|backend=" << synthesizer->execution() << "|threads=" << threads
                      << "|repetition=" << repetition << "|load_ns=" << load_ns << "|total_ns=" << total_ns
                      << "|generation_ns=" << speech->stats.generation_ns << "|decode_ns=" << speech->stats.decode_ns
                      << "|preparation_ns=" << speech->stats.preparation_ns
                      << "|audio_tokens=" << speech->stats.audio_tokens << "|samples=" << speech->samples.size()
                      << "|sample_rate=" << speech->sample_rate << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
