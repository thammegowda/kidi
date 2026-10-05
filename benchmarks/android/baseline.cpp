#include <cstdlib>
#include "kidi/checkpoint/config.h"
#include "kidi/inference/generator.h"
#include "kidi/checkpoint/prepare.h"
#include "kidi/image/gemma4.h"
#include "kidi/inference/transcriber.h"
#include "kidi/model/gemma4_vision.h"
#include "kidi/runtime/ynn/graph.h"

#include <nlohmann/json.hpp>
#include <sys/resource.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <utility>

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

/// Gemma device from KIDI_ACCELERATOR (auto, cpu, gpu, or npu); defaults to cpu for reproducible baselines.
auto gemma_device() -> kidi::tensor::Device {
    const auto* accelerator = std::getenv("KIDI_ACCELERATOR");
    return require(kidi::inference::select_device(accelerator ? accelerator : "cpu"));
}

auto emit(nlohmann::json record, int threads) -> void {
    record["threads"] = threads;
    if (const auto* accelerator = std::getenv("KIDI_ACCELERATOR")) record["accelerator"] = accelerator;
    record["peak_rss_kib"] = peak_rss_kib();
    std::cout << record.dump() << std::endl;
}

auto benchmark_gemma(const std::filesystem::path& directory, int threads, int repeats) -> void {
    const auto started = Clock::now();
    auto generator = require(kidi::inference::Generator::load(directory, gemma_device(), 0, 128, true));
    require(generator.configure_serving({1, 1, CONTEXT, CHUNK}));
    emit({{"stage", "load"},
          {"execution", generator.execution()},
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
    auto generator = require(kidi::inference::Generator::load(directory, gemma_device(), 0, 128, true));
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
    auto generator = require(kidi::inference::Generator::load(directory, gemma_device(), 0, 128, true));
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

struct Comparison {
    double relative_rmse, cosine, maximum;
};

auto compare(std::span<const float> actual, std::span<const float> expected) -> Comparison {
    if (actual.size() != expected.size()) throw std::runtime_error("reference tensor shape changed");
    double error = 0, energy = 0, dot = 0, norm = 0, maximum = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
            throw std::runtime_error("vision tensor is not finite");
        const auto delta = static_cast<double>(actual[index]) - expected[index];
        error += delta * delta;
        energy += static_cast<double>(expected[index]) * expected[index];
        dot += static_cast<double>(actual[index]) * expected[index];
        norm += static_cast<double>(actual[index]) * actual[index];
        maximum = std::max(maximum, std::abs(delta));
    }
    return {std::sqrt(error / std::max(energy, 1e-30)),
            dot / std::max(std::sqrt(energy * norm), 1e-30), maximum};
}

auto host_copy(const kidi::tensor::Tensor& tensor) -> kidi::tensor::Tensor {
    const auto bytes = require(tensor.copy_to_host());
    return require(kidi::tensor::Tensor::from_bytes(
        {tensor.shape().begin(), tensor.shape().end()}, tensor.dtype(), bytes));
}

/// Vision-only path: preprocessing, checkpoint binding, the 16-block tower, pooling, and text-space projection.
auto benchmark_vision(const std::filesystem::path& directory, std::span<const std::filesystem::path> images,
                      int threads, int repeats) -> void {
    const auto config = YAML::LoadFile((directory / "config.json").string());
    const auto weights = require(kidi::checkpoint::Weights::load(directory / "model.safetensors"));
    const auto load_started = Clock::now();
    const kidi::ModuleScope scope(kidi::tensor::DType::F32, false, kidi::tensor::Device::cpu());
    auto vision = kidi::model::Gemma4Vision(config["vision_config"],
                                            config["text_config"]["hidden_size"].as<int>(),
                                            static_cast<bool>(config["quantization_config"]));
    const auto* precision_name = std::getenv("KIDI_VISION_PRECISION");
    const auto precision =
        kidi::core::parse_precision(precision_name ? std::string_view(precision_name) : std::string_view("checkpoint"));
    if (!precision) throw std::runtime_error("invalid KIDI_VISION_PRECISION");
    vision->set_precision(*precision);
    require(vision->set_checkpoint(weights));
    emit({{"stage", "vision_load"},
          {"ms", rounded(elapsed_ms(load_started))},
          {"precision", kidi::core::to_string(*precision)}},
         threads);

    const auto* reference_directory = std::getenv("KIDI_VISION_REFERENCE_DIR");
    const auto* dump_directory = std::getenv("KIDI_VISION_DUMP_DIR");
    std::size_t stage_layers = 3;
    if (const auto* value = std::getenv("KIDI_VISION_STAGE_LAYERS")) {
        const auto text = std::string_view(value);
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), stage_layers);
        if (error != std::errc{} || end != text.data() + text.size() || stage_layers > 16)
            throw std::runtime_error("KIDI_VISION_STAGE_LAYERS must be an integer from 0 to 16");
    }
    for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
        const auto& image = images[image_index];
        std::ifstream stream(image, std::ios::binary);
        if (!stream) throw std::runtime_error("unable to read benchmark image");
        const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(stream), {}};
        const auto prepare_started = Clock::now();
        const auto pixels = require(kidi::image::prepare_gemma4(bytes, 280));

        std::optional<kidi::checkpoint::Weights> reference;
        kidi::tensor::Tensor reference_output;
        std::vector<kidi::tensor::Tensor> reference_blocks;
        std::optional<kidi::image::Gemma4Image> reference_pixels;
        nlohmann::json prepare_record{{"stage", "vision_prepare"},
                                      {"image", image.filename().string()},
                                      {"image_index", image_index},
                                      {"ms", rounded(elapsed_ms(prepare_started))},
                                      {"patch_rows", pixels.patch_rows},
                                      {"patch_columns", pixels.patch_columns},
                                      {"patches", static_cast<std::size_t>(pixels.patch_rows) * pixels.patch_columns}};
        if (reference_directory) {
            const auto path =
                std::filesystem::path(reference_directory) / (image.stem().string() + ".safetensors");
            reference.emplace(require(kidi::checkpoint::Weights::load(path)));
            const auto expected_patches = require(reference->tensor("patches"));
            const auto expected = require(expected_patches.data<float>());
            const auto result = compare(pixels.patches, expected);
            const auto expected_positions = require(reference->tensor("position_ids"));
            const auto positions = require(expected_positions.data<std::int64_t>());
            const auto patch_count = static_cast<std::size_t>(pixels.patch_rows) * pixels.patch_columns;
            if (positions.size() != 2 * patch_count)
                throw std::runtime_error("reference position tensor shape changed");
            for (std::size_t patch = 0; patch < patch_count; ++patch)
                if (positions[2 * patch] != static_cast<std::int64_t>(patch % pixels.patch_columns) ||
                    positions[2 * patch + 1] != static_cast<std::int64_t>(patch / pixels.patch_columns))
                    throw std::runtime_error("reference position IDs differ from Kidi patch order");
            prepare_record["patch_relative_rmse"] = result.relative_rmse;
            prepare_record["patch_cosine"] = result.cosine;
            prepare_record["patch_max_abs"] = result.maximum;
            prepare_record["position_ids_match"] = true;
            if (const auto* canonical = std::getenv("KIDI_VISION_USE_REFERENCE_INPUT");
                canonical && std::string_view(canonical) == "1") {
                reference_pixels = pixels;
                reference_pixels->patches.assign(expected.begin(), expected.end());
            }
            reference_output = require(reference->tensor("visual_embeddings"));
            for (int block = 0; block < 16; ++block) {
                std::ostringstream name;
                name << "block_" << std::setfill('0') << std::setw(2) << block;
                reference_blocks.push_back(require(reference->tensor(name.str())));
            }
        }
        prepare_record["input"] = reference_pixels ? "official_reference" : "kidi_preprocess";
        emit(std::move(prepare_record), threads);

        for (int iteration = 0; iteration <= repeats; ++iteration) {
            const auto started = Clock::now();
            const auto output = require(vision->forward(reference_pixels ? *reference_pixels : pixels));
            const auto values = require(output.data<float>());
            double sum = 0, squared = 0, maximum = 0;
            for (const auto value : values) {
                if (!std::isfinite(value)) throw std::runtime_error("vision output is not finite");
                sum += value;
                squared += static_cast<double>(value) * value;
                maximum = std::max(maximum, std::abs(static_cast<double>(value)));
            }
            nlohmann::json record{{"stage", "vision_encode"},
                                  {"image", image.filename().string()},
                                  {"image_index", image_index},
                                  {"iteration", iteration},
                                  {"warmup", iteration == 0},
                                  {"ms", rounded(elapsed_ms(started))},
                                  {"visual_tokens", output.size(1)},
                                  {"width", output.size(2)},
                                  {"sum", sum},
                                  {"rms", std::sqrt(squared / values.size())},
                                  {"max_abs", maximum}};
            if (reference_output.defined()) {
                const auto expected = require(std::as_const(reference_output).data<float>());
                const auto result = compare(values, expected);
                record["relative_rmse"] = result.relative_rmse;
                record["cosine"] = result.cosine;
                record["reference_max_abs"] = result.maximum;
            }
            emit(std::move(record), threads);
        }
        if (!reference_blocks.empty() || dump_directory) {
            kidi::StateDict dump;
            const auto& diagnostic_pixels = reference_pixels ? *reference_pixels : pixels;
            if (dump_directory) {
                const auto length =
                    static_cast<std::int64_t>(diagnostic_pixels.patch_rows) * diagnostic_pixels.patch_columns;
                dump.emplace(
                    "patches",
                    require(kidi::tensor::Tensor::from_host(
                        {1, length, 768}, std::span<const float>(diagnostic_pixels.patches))));
                std::vector<std::int64_t> positions(static_cast<std::size_t>(2 * length));
                for (std::int64_t patch = 0; patch < length; ++patch) {
                    positions[2 * patch] = patch % diagnostic_pixels.patch_columns;
                    positions[2 * patch + 1] = patch / diagnostic_pixels.patch_columns;
                }
                dump.emplace(
                    "position_ids",
                    require(kidi::tensor::Tensor::from_host(
                        {1, length, 2}, std::span<const std::int64_t>(positions))));
            }
            kidi::model::Gemma4VisionImpl::BlockObserver block_observer =
                [&](std::size_t block, const kidi::tensor::Tensor& output) {
                    std::ostringstream name;
                    name << "block_" << std::setfill('0') << std::setw(2) << block;
                    if (dump_directory) dump.emplace(name.str(), host_copy(output));
                    if (!reference_blocks.empty()) {
                        const auto actual = require(output.data<float>());
                        const auto expected = require(std::as_const(reference_blocks.at(block)).data<float>());
                        const auto result = compare(actual, expected);
                        emit({{"stage", "vision_block"},
                              {"image", image.filename().string()},
                              {"image_index", image_index},
                              {"block", block},
                              {"relative_rmse", result.relative_rmse},
                              {"cosine", result.cosine},
                              {"max_abs", result.maximum}},
                             threads);
                    }
                };
            kidi::model::Gemma4VisionImpl::StageObserver stage_observer =
                [&](std::size_t layer, std::string_view stage, const kidi::tensor::Tensor& output) {
                    std::ostringstream name;
                    name << "layer_" << std::setfill('0') << std::setw(2) << layer << '_' << stage;
                    if (dump_directory) dump.emplace(name.str(), host_copy(output));
                    if (reference && reference->contains(name.str())) {
                        const auto expected_tensor = require(reference->tensor(name.str()));
                        const auto actual = require(output.data<float>());
                        const auto expected = require(expected_tensor.data<float>());
                        const auto result = compare(actual, expected);
                        emit({{"stage", "vision_stage"},
                              {"image", image.filename().string()},
                              {"image_index", image_index},
                              {"layer", layer},
                              {"name", stage},
                              {"relative_rmse", result.relative_rmse},
                              {"cosine", result.cosine},
                              {"max_abs", result.maximum}},
                             threads);
                    }
                };
            const auto diagnostic_output = require(vision->forward(
                reference_pixels ? *reference_pixels : pixels, block_observer, stage_observer, stage_layers));
            if (dump_directory) {
                dump.emplace("encoder_hidden", dump.at("block_15"));
                dump.emplace("visual_embeddings", host_copy(diagnostic_output));
                const auto destination =
                    std::filesystem::path(dump_directory) / (image.stem().string() + ".safetensors");
                std::filesystem::create_directories(destination.parent_path());
                require(kidi::checkpoint::Weights::save(destination, dump));
                emit({{"stage", "vision_dump"},
                      {"image", image.filename().string()},
                      {"path", destination.string()},
                      {"tensors", dump.size()}},
                     threads);
            }
        }
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
                  {"preparation_ms", rounded(result.stats.preparation_ns / 1e6)},
                  {"encode_ms", rounded(result.stats.encode_ns / 1e6)},
                  {"decode_ms", rounded(result.stats.decode_ns / 1e6)},
                  {"token_ids", result.token_ids},
                  {"language", result.language},
                  {"text", result.text}},
                 threads);
        }
    }
}
/// The app's dictation path: INT8 preparation, then drafts while recording and a final pass after stopping.
auto benchmark_dictation(const std::filesystem::path& directory, const std::filesystem::path& wav, int threads,
                         int repeats) -> void {
    const auto waveform = require(kidi::audio::load_wav(wav));
    if (waveform.sample_rate != 16000 || waveform.samples.size() < 32000 || waveform.samples.size() > 480000)
        throw std::runtime_error("benchmark WAV must be 2-30 seconds at 16 kHz");
    const auto started = Clock::now();
    const auto prepared = require(kidi::checkpoint::prepare(directory, kidi::model::WhisperImpl::checkpoint_config(),
                                                            kidi::model::WhisperImpl::int8_preparation));
    auto transcriber = require(kidi::inference::Transcriber::load(prepared));
    emit({{"stage", "load"}, {"model", "whisper-int8"}, {"ms", rounded(elapsed_ms(started))}}, threads);
    // Defaults match the app: fitted audio and a warm-up at load. Set either variable to 0 to compare.
    const auto enabled = [](const char* name) {
        const auto* value = std::getenv(name);
        return !value || std::string_view(value) != "0";
    };
    const auto fit = enabled("KIDI_FIT_AUDIO");
    if (enabled("KIDI_WARM_UP")) {
        const auto warm = Clock::now();
        require(transcriber.warm_up());
        emit({{"stage", "warm_up"}, {"ms", rounded(elapsed_ms(warm))}}, threads);
    }
    std::vector<std::size_t> sizes;
    for (std::size_t size = 12800; size < waveform.samples.size(); size += 19200) sizes.push_back(size);
    sizes.push_back(waveform.samples.size());
    for (int iteration = 0; iteration < repeats; ++iteration)
        for (const auto size : sizes) {
            const auto before = Clock::now();
            const auto result = require(transcriber.transcribe(
                std::span(waveform.samples).first(size), 16000,
                {.language = "auto", .maximum_tokens = 128, .fit_audio = fit}));
            emit({{"stage", "transcription"},
                  {"iteration", iteration},
                  {"fit_audio", fit},
                  {"audio_seconds", rounded(size / 16000.0)},
                  {"wall_ms", rounded(elapsed_ms(before))},
                  {"feature_ms", rounded(result.stats.feature_ns / 1e6)},
                  {"encode_ms", rounded(result.stats.encode_ns / 1e6)},
                  {"decode_ms", rounded(result.stats.decode_ns / 1e6)},
                  {"preparation_ms", rounded(result.stats.preparation_ns / 1e6)},
                  {"tokens", result.token_ids.size()},
                  {"text", result.text}},
                 threads);
        }
}
} // namespace

auto main(int argc, char** argv) -> int {
    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
        std::cout << "usage: kidi_android_baseline gemma|chat|whisper|image|vision|dictation MODEL THREADS REPEATS "
                     "[MEDIA ...]\n";
        return 0;
    }
    try {
        if (argc < 5) throw std::runtime_error("missing benchmark arguments");
        const std::string mode(argv[1]);
        const auto threads = std::stoi(argv[3]), repeats = std::stoi(argv[4]);
        if (threads < 1 || threads > 8 || repeats < 1 || repeats > 20 ||
            !(((mode == "gemma" || mode == "chat") && argc == 5) ||
              ((mode == "whisper" || mode == "image" || mode == "dictation") && argc == 6) ||
              (mode == "vision" && argc >= 6)))
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
        else if (mode == "vision") {
            std::vector<std::filesystem::path> images;
            for (int index = 5; index < argc; ++index) images.emplace_back(argv[index]);
            benchmark_vision(argv[2], images, threads, repeats);
        }
        else if (mode == "dictation")
            benchmark_dictation(argv[2], argv[5], threads, repeats);
        else
            benchmark_whisper(argv[2], argv[5], threads, repeats);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}