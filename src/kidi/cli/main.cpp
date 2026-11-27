#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <deque>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "kidi/cli/argparse.h"
#include "kidi/cli/log.h"
#include "kidi/cli/main.h"
#include "kidi/audio/wav.h"
#include "kidi/cli/interactive.h"
#include "kidi/cli/index.h"
#include "kidi/cli/serve.h"
#ifdef KIDI_HAS_CODE
#include "kidi/cli/managed_serve.h"
#endif
#include "kidi/core/version.h"
#include "kidi/checkpoint/package.h"
#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/hub.h"
#include "kidi/checkpoint/weights.h"
#include "kidi/inference/translator.h"
#include "kidi/inference/generator.h"
#include "kidi/inference/synthesizer.h"
#include "kidi/inference/transcriber.h"
#include "kidi/model/omnivoice.h"
#include "kidi/model/kokoro.h"
#include "kidi/model/gemma4.h"
#include "kidi/model/registry.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/tensor/backend.h"
#include "ynnpack/base/topology.h"

namespace kidi::cli {
auto auto_inference_device(std::span<const tensor::BackendInfo> backends, tensor::TransformerSupport required,
                           bool allow_metal) -> tensor::Device {
    for (const auto device : {tensor::Device::cuda(), tensor::Device::apple_gpu()}) {
        if (device.kind == tensor::DeviceKind::A_GPU && !allow_metal) continue;
        if (!device.capabilities().compute.supports(required)) continue;
        if (std::ranges::any_of(backends, [&](const auto& backend) {
                return backend.device_kind == device.kind && backend.storage_available && backend.execution_available;
            }))
            return device;
    }
    return tensor::Device::cpu();
}
} // namespace kidi::cli

namespace {

auto inference_device(const kidi::cli::Namespace& arguments,
                      kidi::tensor::TransformerSupport required = kidi::tensor::TransformerSupport::QUANTIZED,
                      bool allow_metal = true, bool cpu_only = false) -> kidi::tensor::Device {
    const auto& backend = arguments.get<std::string>("backend");
    if (const auto device = kidi::tensor::parse_cuda_device(backend)) return *device;
    if (backend == "vulkan") return kidi::tensor::Device::vulkan();
    if (backend == "npu") return kidi::tensor::Device::qualcomm_npu();
    if (backend != "mps" && backend != "auto" && backend != "ynnpack")
        throw std::invalid_argument("unknown backend; expected auto, ynnpack, mps, vulkan, npu, or cuda[:INDEX]");
    const bool speculative =
        (arguments.contains("assistant_model") && !arguments.get<std::filesystem::path>("assistant_model").empty()) ||
        (arguments.contains("prompt_lookup") && arguments.get<bool>("prompt_lookup"));
    if (backend == "mps") return kidi::tensor::Device::apple_gpu();
    if (backend == "ynnpack" || speculative || cpu_only) return kidi::tensor::Device::cpu();
    return kidi::cli::auto_inference_device(kidi::tensor::BackendRegistry::instance().backends(), required,
                                            allow_metal);
}

/// Stable label for machine-readable stderr records.
auto backend_label(kidi::tensor::Device device) -> std::string_view {
    if (device.kind == kidi::tensor::DeviceKind::CUDA) return "cuda_gpu";
    if (device == kidi::tensor::Device::apple_gpu()) return "mps_gpu";
    if (device == kidi::tensor::Device::vulkan()) return "vulkan_gpu";
    return "ynnpack_cpu";
}

/// Applies the CPU thread count. Commands that compute on the CPU also keep those threads on one NUMA node, so the
/// weights they stream every token are read from local memory.
auto configure_cpu_threads(std::int32_t threads, kidi::tensor::Device device) -> void {
    kidi::runtime::ynn::set_thread_count(threads);
    if (device == kidi::tensor::Device::cpu()) {
        const auto resolved = kidi::runtime::ynn::thread_count();
        ::ynn::confine_to_numa_node(resolved);
        ::ynn::confine_to_performance_cpus(resolved);
        (void)kidi::runtime::ynn::reserve_thread_pool(resolved);
    }
}

auto inspect(const std::filesystem::path& directory) -> int {
    auto document = kidi::checkpoint::load_config(directory);
    if (!document) {
        kidi::cli::log::error("{}", document.error().message);
        return 1;
    }
    if (auto validation = kidi::model::validate_model_config((*document)["model"]); !validation) {
        kidi::cli::log::error("{}", validation.error().message);
        return 1;
    }
    const auto type = (*document)["model"]["type"].as<std::string>();
    if (type == "whisper") {
        auto weights = kidi::checkpoint::Weights::load((*document)["weights_file"].as<std::string>());
        if (!weights) {
            kidi::cli::log::error("{}", weights.error().message);
            return 1;
        }
        const auto parameter = weights->tensor("model.encoder.conv1.weight");
        if (!parameter) {
            kidi::cli::log::error("{}", parameter.error().message);
            return 1;
        }
        std::cout << "format: huggingface\n"
                  << "model: whisper\n"
                  << "weights: " << (*document)["weights_file"].as<std::string>() << '\n'
                  << "weight tensors: " << weights->size() << '\n'
                  << "weight dtype: " << kidi::tensor::to_string(parameter->dtype()) << '\n'
                  << "encoder layers: " << (*document)["model"]["encoder_layers"].as<int>() << '\n'
                  << "decoder layers: " << (*document)["model"]["decoder_layers"].as<int>() << '\n'
                  << "vocabulary: " << (*document)["model"]["vocab_size"].as<int>() << '\n'
                  << "execution device: cpu\n";
        return 0;
    }
    if (type == "gemma4_text") {
        auto weights = kidi::checkpoint::Weights::load((*document)["weights_file"].as<std::string>());
        if (!weights) {
            kidi::cli::log::error("{}", weights.error().message);
            return 1;
        }
        const bool qat = kidi::checkpoint::native_qat((*document)["model"]);
        const auto embedding = weights->tensor(qat ? "model.language_model.embed_tokens.embedding_quantized"
                                                   : "model.language_model.embed_tokens.weight");
        if (!embedding) {
            kidi::cli::log::error("{}", embedding.error().message);
            return 1;
        }
        std::cout << "format: " << (*document)["format_version"].as<int>() << '\n'
                  << "model: gemma4_text\n"
                  << "weights: " << (*document)["weights_file"].as<std::string>() << '\n'
                  << "checkpoint tensors (including unused modalities): " << weights->size() << '\n'
                  << "weight dtype: " << kidi::tensor::to_string(embedding->dtype()) << '\n'
                  << "native mobile QAT: " << (qat ? "yes" : "no") << '\n'
                  << "text layers: " << (*document)["model"]["num_hidden_layers"].as<int>() << '\n'
                  << "vocabulary: " << (*document)["model"]["vocab_size"].as<int>() << '\n'
                  << "default device: " << kidi::tensor::to_string(kidi::module_device) << '\n';
        return 0;
    }
    if (type == "omnivoice") {
        auto weights = kidi::checkpoint::Weights::load((*document)["weights_file"].as<std::string>());
        if (!weights) {
            kidi::cli::log::error("{}", weights.error().message);
            return 1;
        }
        const auto embedding = weights->tensor("llm.embed_tokens.weight");
        if (!embedding) {
            kidi::cli::log::error("{}", embedding.error().message);
            return 1;
        }
        const auto& model = (*document)["model"];
        std::cout << "format: " << (*document)["format_version"].as<int>() << '\n'
                  << "task: tts\n"
                  << "model: omnivoice\n"
                  << "weights: " << (*document)["weights_file"].as<std::string>() << '\n'
                  << "weight tensors: " << weights->size() << '\n'
                  << "weight dtype: " << kidi::tensor::to_string(embedding->dtype()) << '\n'
                  << "text layers: " << model["llm_config"]["num_hidden_layers"].as<int>() << '\n'
                  << "text vocabulary: " << model["llm_config"]["vocab_size"].as<int>() << '\n'
                  << "audio codebooks: " << model["num_audio_codebook"].as<int>() << '\n'
                  << "sample rate: " << model["codec_config"]["sample_rate"].as<int>() << '\n'
                  << "execution device: cpu\n";
        return 0;
    }
    if (type == "kokoro") {
        auto weights = kidi::checkpoint::Weights::load((*document)["weights_file"].as<std::string>());
        if (!weights) {
            kidi::cli::log::error("{}", weights.error().message);
            return 1;
        }
        const auto embedding = weights->tensor("bert.module.embeddings.word_embeddings.weight");
        if (!embedding) {
            kidi::cli::log::error("{}", embedding.error().message);
            return 1;
        }
        const auto& model = (*document)["model"];
        std::cout << "format: " << (*document)["format_version"].as<int>() << '\n'
                  << "task: tts\n"
                  << "model: kokoro\n"
                  << "weights: " << (*document)["weights_file"].as<std::string>() << '\n'
                  << "weight tensors: " << weights->size() << '\n'
                  << "weight dtype: " << kidi::tensor::to_string(embedding->dtype()) << '\n'
                  << "voices: " << model["voices"].size() << '\n'
                  << "sample rate: " << model["sample_rate"].as<int>() << '\n'
                  << "execution device: cpu\n";
        return 0;
    }
    auto package = kidi::checkpoint::Package::load(directory);
    if (!package) {
        kidi::cli::log::error("{}", package.error().message);
        return 1;
    }
    const auto& config = package->config();
    std::cout << "format: " << config["format_version"].as<int>() << '\n'
              << "model: " << config["model"]["type"].as<std::string>() << '\n'
              << "weights: " << config["weights_file"].as<std::string>() << '\n'
              << "weight tensors: " << package->weights().size() << '\n'
              << "vocabularies: " << package->source_tokenizer().vocabulary_size() << ", "
              << package->target_tokenizer().vocabulary_size() << '\n'
              << "beam: " << config["decode"]["beam_size"].as<int>() << '\n'
              << "default device: " << kidi::tensor::to_string(kidi::module_device) << '\n'
              << "cpu features: " << kidi::runtime::ynn::supported_arch_names() << '\n';
    for (const auto& backend : kidi::tensor::BackendRegistry::instance().backends()) {
        std::cout << "tensor backend: " << kidi::tensor::to_string(backend.device_kind) << ' ' << backend.name
                  << " storage=" << (backend.storage_available ? "yes" : "no")
                  << " execution=" << (backend.execution_available ? "yes" : "no");
        if (!backend.storage_available) std::cout << " reason=" << backend.unavailable_reason;
        std::cout << '\n';
    }
    return 0;
}

auto validate_chat_request(const nlohmann::ordered_json& request) -> kidi::Result<void> {
    const auto invalid = [](std::string message) {
        return std::unexpected(kidi::Error{kidi::ErrorCode::INVALID_ARGUMENT, std::move(message)});
    };
    if (!request.is_object() || !request.contains("messages") || !request["messages"].is_array() ||
        request["messages"].empty())
        return invalid("expected an object with a nonempty messages array");
    for (const auto& [key, value] : request.items())
        if (key != "messages" && key != "id" && key != "max_tokens" && key != "template_args")
            return invalid("unsupported request field: " + key);
    if (request.contains("id") && !request["id"].is_string() && !request["id"].is_number_integer())
        return invalid("id must be a string or integer");
    if (request.contains("max_tokens") &&
        (!request["max_tokens"].is_number_integer() || request["max_tokens"] <= 0 || request["max_tokens"] > INT32_MAX))
        return invalid("max_tokens must be a positive integer within int32 bounds");
    if (request.contains("template_args") && !request["template_args"].is_object())
        return invalid("template_args must be an object");
    return {};
}

struct ChatSession {
    kidi::inference::Generator generator;
    kidi::inference::GenerationOptions options;
    std::int64_t load_ns;
    kidi::tensor::Device device;
};

auto load_chat_session(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory,
                       const std::filesystem::path& assistant_directory) -> kidi::Result<ChatSession> {
    const auto device = inference_device(arguments);
    const auto precision = kidi::core::parse_precision(arguments.get<std::string>("precision"));
    if (!precision) return std::unexpected(kidi::Error{kidi::ErrorCode::INVALID_ARGUMENT, "invalid precision"});
    const auto kv_cache_precision =
        kidi::core::parse_kv_cache_precision(arguments.get<std::string>("kv_cache_precision"));
    if (!kv_cache_precision)
        return std::unexpected(kidi::Error{kidi::ErrorCode::INVALID_ARGUMENT, "invalid KV-cache precision"});
    const auto prompt_lookup = arguments.get<bool>("prompt_lookup");
    if (arguments.contains("draft_tokens") && assistant_directory.empty() && !prompt_lookup)
        return std::unexpected(
            kidi::Error{kidi::ErrorCode::INVALID_ARGUMENT,
                        "--draft-tokens requires --assistant-model or --prompt-lookup"});
    const auto draft_tokens =
        arguments.contains("draft_tokens") ? arguments.get<std::size_t>("draft_tokens") : std::size_t{3};
    const auto started = std::chrono::steady_clock::now();
    auto loaded =
        kidi::inference::Generator::load(directory, {.device = device,
                                                     .precision = *precision,
                                                     .kv_cache_precision = *kv_cache_precision,
                                                     .packed_prefill = arguments.get<bool>("packed_prefill"),
                                                     .dense_lm_head = arguments.get<bool>("dense_lm_head"),
                                                     .assistant_model = assistant_directory,
                                                     .prompt_lookup = prompt_lookup,
                                                     .draft_tokens = draft_tokens});
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    const auto load_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
    return ChatSession{std::move(*loaded),
                       {.maximum_new_tokens = arguments.get<std::size_t>("max_new_tokens"),
                        .context_size = arguments.get<std::size_t>("context_size"),
                        .prefill_chunk_size = arguments.get<std::size_t>("prefill_chunk_size"),
                        .ignore_eos = arguments.get<bool>("ignore_eos"),
                        .full_attention_cache = arguments.get<bool>("full_attention_cache")},
                       load_ns,
                       device};
}

auto generate_chat(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory,
                   const std::filesystem::path& assistant_directory, std::istream& input, std::ostream& output) -> int {
    try {
        const auto started = std::chrono::steady_clock::now();
        auto session = load_chat_session(arguments, directory, assistant_directory);
        if (!session) {
            kidi::cli::log::error("{}", session.error().message);
            return 1;
        }
        auto& generator = session->generator;
        const auto& options = session->options;
        const auto device = session->device;
        const auto load_ns = session->load_ns;
        const auto max_active = arguments.contains("max_active")
                                    ? arguments.get<std::size_t>("max_active")
                                    : (generator.speculative() ? std::size_t{1} : std::size_t{4});
        const kidi::inference::ServingOptions limits{.maximum_active = max_active,
                                                     .maximum_requests = arguments.get<std::size_t>("queue_size"),
                                                     .cache_token_budget = arguments.get<std::size_t>("cache_tokens"),
                                                     .prefill_tokens_per_step = options.prefill_chunk_size};
        auto configured = generator.configure_serving(limits);
        if (!configured) {
            kidi::cli::log::error("{}", configured.error().message);
            return 2;
        }
        bool ended = false;
        struct Response {
            nlohmann::json id;
            std::optional<nlohmann::json> completed;
        };
        std::deque<Response> ordered;
        std::uint64_t next_output = 1;
        std::size_t line_number = 0;
        std::uint64_t prefill_ns = 0, decode_ns = 0, preparation_ns = 0;
        std::size_t completed = 0, peak_cache_tokens = 0;
        while (!ended || !ordered.empty()) {
            while (!ended && ordered.size() < limits.maximum_requests) {
                std::string line;
                if (!std::getline(input, line)) {
                    if (!input.eof()) {
                        kidi::cli::log::error("failed reading chat input");
                        return 1;
                    }
                    ended = true;
                    break;
                }
                ++line_number;
                const auto request = nlohmann::ordered_json::parse(line, nullptr, false);
                auto valid = validate_chat_request(request);
                if (!valid) {
                    kidi::cli::log::error("input line {}: {}", line_number, valid.error().message);
                    return 2;
                }
                auto request_options = options;
                if (request.contains("max_tokens"))
                    request_options.maximum_new_tokens = request["max_tokens"].get<std::size_t>();
                auto template_args = request.value("template_args", nlohmann::ordered_json::object());
                if (!request.contains("template_args")) template_args["enable_thinking"] = false;
                auto queued = generator.enqueue_chat(request["messages"], request_options, template_args);
                if (!queued) {
                    kidi::cli::log::error("input line {}: {}", line_number, queued.error().message);
                    return queued.error().code == kidi::ErrorCode::INVALID_ARGUMENT ? 2 : 1;
                }
                ordered.push_back(
                    {request.contains("id") ? nlohmann::json(request["id"]) : nlohmann::json(*queued), {}});
            }
            if (!generator.pending_requests()) break;
            auto step = generator.step();
            if (!step) {
                kidi::cli::log::error("{}", step.error().message);
                return 1;
            }
            const auto& result = *step;
            prefill_ns += result.prefill_ns;
            decode_ns += result.decode_ns;
            preparation_ns += result.preparation_ns;
            peak_cache_tokens = std::max(peak_cache_tokens, result.reserved_cache_tokens);
            for (const auto& event : result.events) {
                if (!event.completed) continue;
                const auto& generation = *event.completed;
                auto& response = ordered.at(event.request_id - next_output);
                nlohmann::json record{{"request_id", event.request_id},
                                      {"id", response.id},
                                      {"message", {{"role", "assistant"}, {"content", generation.text}}},
                                      {"token_ids", generation.generation.token_ids},
                                      {"decoder_steps", generation.generation.decoder_steps}};
                if (arguments.get<bool>("profile")) {
                    record["ttft_ns"] = generation.stats.time_to_first_token_ns;
                    record["generation_ns"] = generation.stats.generation_ns;
                    record["prompt_tokens"] = generation.stats.prompt_tokens;
                    record["prefill_ns"] = generation.stats.prefill_ns;
                    record["decode_tokens"] = generation.stats.decode_tokens;
                    record["speculative"] = generation.stats.speculative;
                    record["draft_tokens"] = generation.stats.draft_tokens;
                    record["accepted_draft_tokens"] = generation.stats.accepted_draft_tokens;
                    record["verification_steps"] = generation.stats.verification_steps;
                    record["draft_ns"] = generation.stats.draft_ns;
                    record["verify_ns"] = generation.stats.verify_ns;
                    if (limits.maximum_active == 1) {
                        record["decode_ns"] = generation.stats.decode_ns;
                        record["preparation_ns"] = generation.stats.preparation_ns;
                    }
                }
                response.completed = std::move(record);
            }
            while (!ordered.empty() && ordered.front().completed) {
                output << ordered.front().completed->dump() << '\n';
                ordered.pop_front();
                ++next_output;
                ++completed;
            }
            output.flush();
            if (!output) {
                kidi::cli::log::error("failed writing generation results");
                return 1;
            }
        }
        if (arguments.get<bool>("profile")) {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
                    .count();
            std::cerr << "kidi_serving|requests=" << completed << "|max_active=" << limits.maximum_active
                      << "|backend=" << backend_label(device) << "|device=" << kidi::tensor::to_string(device)
                      << "|load_ns=" << load_ns << "|native_qat=" << generator.native_qat()
                      << "|packed_prefill=" << arguments.get<bool>("packed_prefill")
                      << "|cache_tokens=" << limits.cache_token_budget
                      << "|peak_reserved_cache_tokens=" << peak_cache_tokens << "|prefill_ns=" << prefill_ns
                      << "|decode_ns=" << decode_ns << "|preparation_ns=" << preparation_ns
                      << "|generation_ns=" << elapsed << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        kidi::cli::log::error("{}", error.what());
        return 1;
    }
}
auto generate_translation(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory,
                          std::istream& input, std::ostream& output) -> int {
    kidi::inference::InferenceStats profile;
    const auto batch_size = arguments.get<std::int32_t>("batch_size");
    const auto batch_window =
        arguments.contains("batch_window") ? arguments.get<std::int32_t>("batch_window") : batch_size;
    if (batch_size < 1 || batch_size > 256) {
        kidi::cli::log::error("batch size must be between 1 and 256");
        return 2;
    }
    if (batch_window < batch_size || batch_window > 4096) {
        kidi::cli::log::error("batch window must be between batch-size and 4096");
        return 2;
    }
    auto* profile_ptr = arguments.get<bool>("profile") ? &profile : nullptr;
    const auto device = inference_device(arguments, kidi::tensor::TransformerSupport::GENERAL);
    auto translator = kidi::inference::Translator::load(directory, device, profile_ptr, batch_size);
    if (!translator) {
        kidi::cli::log::error("{}", translator.error().message);
        return 1;
    }

    kidi::inference::DecodeOptions options;
    if (arguments.contains("beam_size")) options.beam_size = arguments.get<std::int32_t>("beam_size");
    if (arguments.contains("maximum_extra_tokens")) {
        options.maximum_extra_tokens = arguments.get<std::int32_t>("maximum_extra_tokens");
    }
    if (arguments.contains("length_penalty")) options.length_penalty = arguments.get<float>("length_penalty");
    options.compute_score = arguments.get<bool>("score");

    std::size_t line_number = 0;
    std::size_t translated_items = 0;
    std::size_t target_tokens = 0;
    std::vector<std::string> pending;
    const auto flush = [&]() -> bool {
        std::vector<std::string> sources;
        for (const auto& line : pending)
            if (!line.empty()) sources.push_back(line);
        auto translations = translator->translate_batch(sources, options, profile_ptr);
        if (!translations) {
            kidi::cli::log::error("batch ending at input line {}: {}", line_number, translations.error().message);
            return false;
        }
        std::size_t index = 0;
        for (const auto& line : pending) {
            if (!line.empty()) {
                const auto& translation = (*translations)[index++];
                output << translation.text;
                if (arguments.get<bool>("score")) output << '\t' << translation.score;
                ++translated_items;
                target_tokens += translation.token_ids.size();
            }
            output << '\n';
        }
        output.flush();
        pending.clear();
        if (!output) {
            kidi::cli::log::error("cannot write output: {}", arguments.get<std::string>("output"));
            return false;
        }
        return true;
    };
    for (std::string line; std::getline(input, line);) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        pending.push_back(std::move(line));
        if (pending.size() == static_cast<std::size_t>(batch_window) && !flush()) return 1;
    }
    if (!pending.empty() && !flush()) return 1;
    if (input.bad()) {
        kidi::cli::log::error("cannot read input: {}", arguments.get<std::string>("input"));
        return 1;
    }
    if (arguments.get<bool>("stats")) {
        std::cerr << "kidi_metrics|input_lines=" << line_number << "|translated_items=" << translated_items
                  << "|target_tokens=" << target_tokens << '\n';
    }
    if (profile_ptr) {
        std::cerr << "kidi_profile|backend=" << backend_label(device)
                  << "|threads=" << kidi::runtime::ynn::thread_count()
                  << "|ynn_arch_flags=" << kidi::runtime::ynn::supported_arch_flags()
                  << "|ynn_arch=" << kidi::runtime::ynn::supported_arch_names()
                  << "|package_load_ns=" << profile.package_load_ns << "|graph_compile_ns=" << profile.graph_compile_ns
                  << "|source_tokenize_ns=" << profile.source_tokenize_ns << "|encoder_ns=" << profile.encoder_ns
                  << "|source_embedding_ns=" << profile.source_embedding_ns << "|decoder_ns=" << profile.decoder_ns
                  << "|source_projection_ns=" << profile.source_projection_ns
                  << "|target_embedding_ns=" << profile.target_embedding_ns
                  << "|host_search_ns=" << profile.host_search_ns << "|target_decode_ns=" << profile.target_decode_ns
                  << "|translate_ns=" << profile.translate_ns << "|decoder_steps=" << profile.decoder_steps << '\n';
    }
    return 0;
}

/// Validates the thread count and that the package holds the model family this command serves.
auto prepare_command(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory,
                     kidi::model::Task task, std::string* model_type = nullptr) -> int {
    const auto threads = arguments.get<std::int32_t>("threads");
    if (threads <= 0) {
        kidi::cli::log::error("thread count must be positive");
        return 2;
    }
    auto config = kidi::checkpoint::load_config(directory);
    if (!config) {
        kidi::cli::log::error("{}", config.error().message);
        return 1;
    }
    const auto type = (*config)["model"]["type"].as<std::string>();
    if (!kidi::model::supports_task(type, task)) {
        kidi::cli::log::error("{} expects a model supporting {}, but this package is {}",
                              arguments.get<std::string>("command"), kidi::model::task_name(task), type);
        return 2;
    }
    if (model_type) *model_type = type;
    const auto required = task == kidi::model::Task::TRANSLATION ? kidi::tensor::TransformerSupport::GENERAL
                                                                 : kidi::tensor::TransformerSupport::QUANTIZED;
    const auto device = inference_device(arguments, required, true, task == kidi::model::Task::TTS && type == "kokoro");
    kidi::cli::log::info("inference device: {} (backend={})", kidi::tensor::to_string(device),
                         arguments.get<std::string>("backend"));
    configure_cpu_threads(threads, device);
    return 0;
}

/// Opens the -i/-o files, defaulting to standard input and output.
auto run_with_streams(const kidi::cli::Namespace& arguments,
                      const std::function<int(std::istream&, std::ostream&)>& body) -> int {
    const auto& input_path = arguments.get<std::string>("input");
    const auto& output_path = arguments.get<std::string>("output");
    if (input_path != "-" && output_path != "-" && std::filesystem::exists(output_path) &&
        std::filesystem::equivalent(input_path, output_path)) {
        kidi::cli::log::error("input and output must be different files");
        return 2;
    }
    std::ifstream input_file;
    std::ofstream output_file;
    std::istream* input = &std::cin;
    std::ostream* output = &std::cout;
    if (input_path != "-") {
        input_file.open(input_path);
        if (!input_file) {
            kidi::cli::log::error("cannot open input file: {}", input_path);
            return 1;
        }
        input = &input_file;
    }
    if (output_path != "-") {
        output_file.open(output_path);
        if (!output_file) {
            kidi::cli::log::error("cannot open output file: {}", output_path);
            return 1;
        }
        output = &output_file;
    }
    return body(*input, *output);
}

auto translate_command(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory) -> int {
    if (const auto status = prepare_command(arguments, directory, kidi::model::Task::TRANSLATION)) return status;
    return run_with_streams(arguments, [&](std::istream& input, std::ostream& output) {
        return generate_translation(arguments, directory, input, output);
    });
}

auto resolve_hub_argument(const std::filesystem::path& path, const std::filesystem::path& cache,
                          std::filesystem::path& resolved_out) -> int {
    if (path.empty()) {
        resolved_out.clear();
        return 0;
    }
    const auto text = path.string();
    if (!text.starts_with('@')) {
        resolved_out = path;
        return 0;
    }
    try {
        const kidi::checkpoint::Hub hub{"https://huggingface.co", cache};
        auto resolved = hub.download(std::string_view(text).substr(1));
        if (!resolved) {
            kidi::cli::log::error("{}", resolved.error().message);
            return 1;
        }
        resolved_out = std::move(*resolved);
        return 0;
    } catch (const std::exception& error) {
        kidi::cli::log::error("{}", error.what());
        return 1;
    }
}

auto generate_command(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory) -> int {
    if (const auto status = prepare_command(arguments, directory, kidi::model::Task::TEXT_GENERATION)) return status;
    std::filesystem::path assistant_directory;
    if (const auto status = resolve_hub_argument(arguments.get<std::filesystem::path>("assistant_model"),
                                                 arguments.get<std::filesystem::path>("cache"), assistant_directory))
        return status;
    return run_with_streams(arguments, [&](std::istream& input, std::ostream& output) {
        return generate_chat(arguments, directory, assistant_directory, input, output);
    });
}

auto chat_command(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory) -> int {
    if (const auto status = prepare_command(arguments, directory, kidi::model::Task::TEXT_GENERATION)) return status;
    std::filesystem::path assistant_directory;
    if (const auto status = resolve_hub_argument(arguments.get<std::filesystem::path>("assistant_model"),
                                                 arguments.get<std::filesystem::path>("cache"), assistant_directory))
        return status;
    auto session = load_chat_session(arguments, directory, assistant_directory);
    if (!session) {
        kidi::cli::log::error("{}", session.error().message);
        return 1;
    }
    return kidi::cli::interactive_chat(session->generator, session->options, arguments, session->load_ns);
}

#ifdef KIDI_HAS_CODE
auto managed_context_size(kidi::inference::Generator& generator, kidi::tensor::Device device)
    -> kidi::Result<std::size_t> {
    auto backend = kidi::tensor::BackendRegistry::instance().backend(device);
    if (!backend) return std::unexpected(backend.error());
    const auto free = (*backend)->free_memory(device);
    // Leave a quarter of free memory for activations, other processes and allocator slack.
    if (!free) return std::size_t{0}; // unknown: keep the model's configured context
    auto fitted = generator.fit_context_size(free / 4 * 3);
    if (fitted)
        kidi::cli::log::info("context: {} tokens ({} GiB free on {})", *fitted, free >> 30,
                             kidi::tensor::to_string(device));
    return fitted;
}
#endif

auto serve_command(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory) -> int {
    if (const auto status = prepare_command(arguments, directory, kidi::model::Task::TEXT_GENERATION)) return status;
    std::filesystem::path assistant_directory;
    if (const auto status = resolve_hub_argument(arguments.get<std::filesystem::path>("assistant_model"),
                                                 arguments.get<std::filesystem::path>("cache"), assistant_directory))
        return status;
    auto session = load_chat_session(arguments, directory, assistant_directory);
    if (!session) {
        kidi::cli::log::error("{}", session.error().message);
        return 1;
    }
    auto cache_tokens = arguments.get<std::size_t>("cache_tokens");
    session->options.prefix_cache_bytes = arguments.get<std::size_t>("prefix_cache_mib") << 20;
    bool compact_cache = false;
#ifdef KIDI_HAS_CODE
    // Kidi Code sessions grow long: allocate each request's KV cache on demand and, unless the user chose a
    // context size, use the largest context that fits in free memory (up to the model limit).
    if (!arguments.get<std::filesystem::path>("managed_dir").empty()) {
        compact_cache = true;
        if (!session->options.context_size) {
            auto fitted = managed_context_size(session->generator, session->device);
            if (!fitted) {
                kidi::cli::log::error("{}", fitted.error().message);
                return 2;
            }
            session->options.context_size = *fitted;
        }
        cache_tokens = std::max(cache_tokens, session->options.context_size);
        // Agent turns resend the whole conversation; keep the last state so only the new suffix is prefilled.
        session->options.prefix_cache_bytes = std::numeric_limits<std::size_t>::max();
    }
#endif
    auto configured = session->generator.configure_serving(
        {.maximum_active = arguments.get<std::size_t>("max_active"),
         .maximum_requests = arguments.get<std::size_t>("queue_size"),
         .cache_token_budget = cache_tokens,
         .prefill_tokens_per_step = session->options.prefill_chunk_size,
         .compact_cache = compact_cache});
    if (!configured) {
        kidi::cli::log::error("{}", configured.error().message);
        return 2;
    }
    auto name = arguments.get<std::string>("served_model_name");
    if (name.empty()) name = arguments.get<std::filesystem::path>("model").string();
    if (name.starts_with('@')) name.erase(0, 1);
    const kidi::cli::ServeOptions options{.host = arguments.get<std::string>("host"),
                                          .port = arguments.get<std::int32_t>("port"),
                                          .model_name = std::move(name),
                                          .generation = session->options};
#ifdef KIDI_HAS_CODE
    if (!arguments.get<std::filesystem::path>("managed_dir").empty())
        return kidi::cli::serve_managed(session->generator, options, arguments);
#endif
    return kidi::cli::serve(session->generator, options);
}

auto transcribe_command(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory) -> int {
    const auto threads = arguments.get<std::int32_t>("threads");
    if (threads <= 0) {
        kidi::cli::log::error("thread count must be positive");
        return 2;
    }
    const auto& backend = arguments.get<std::string>("backend");
    if (backend != "auto" && backend != "ynnpack" && !kidi::tensor::parse_cuda_device(backend)) {
        kidi::cli::log::error("Whisper requires the YNNPACK or CUDA backend");
        return 2;
    }
    const auto chunk_seconds = arguments.get<std::size_t>("chunk_seconds");
    const auto maximum_tokens = arguments.get<std::size_t>("max_new_tokens");
    if (!chunk_seconds || chunk_seconds > 30 || !maximum_tokens || maximum_tokens > 444) {
        kidi::cli::log::error("Whisper requires --chunk-seconds from 1 to 30 and --max-new-tokens from 1 to 444");
        return 2;
    }
    const auto& input_path = arguments.get<std::filesystem::path>("input");
    auto reader = kidi::audio::WavReader::open(input_path);
    if (!reader) {
        kidi::cli::log::error("{}", reader.error().message);
        return 2;
    }
    if (reader->sample_rate() != 16000) {
        kidi::cli::log::error("Whisper requires a 16 kHz PCM WAV; resample the input before transcription");
        return 2;
    }
    const auto& output_path = arguments.get<std::string>("output");
    if (output_path != "-" && std::filesystem::exists(output_path) &&
        std::filesystem::equivalent(input_path, output_path)) {
        kidi::cli::log::error("transcript output must not overwrite the input audio");
        return 2;
    }
    const auto device = inference_device(arguments, kidi::tensor::TransformerSupport::QUANTIZED, false);
    kidi::cli::log::info("inference device: {} (backend={})", kidi::tensor::to_string(device), backend);
    configure_cpu_threads(threads, device);
    const auto started = std::chrono::steady_clock::now();
    auto transcriber = kidi::inference::Transcriber::load(directory, device);
    if (!transcriber) {
        kidi::cli::log::error("{}", transcriber.error().message);
        return 1;
    }
    const auto load_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
    const kidi::inference::TranscriptionOptions options{
        .language = arguments.get<std::string>("language"),
        .task = arguments.get<std::string>("task"),
        .maximum_tokens = maximum_tokens,
    };
    std::ofstream file;
    std::ostream* output = &std::cout;
    if (output_path != "-") {
        file.open(output_path);
        if (!file) {
            kidi::cli::log::error("cannot open output file: {}", output_path);
            return 1;
        }
        output = &file;
    }
    const auto chunk_samples = chunk_seconds * 16000;
    const auto chunks = (reader->sample_count() + chunk_samples - 1) / chunk_samples;
    kidi::inference::TranscriptionStats stats;
    std::size_t tokens = 0;
    std::string language;
    for (std::uint64_t chunk = 0; chunk < chunks; ++chunk) {
        auto samples = reader->read(chunk_samples);
        if (!samples) {
            kidi::cli::log::error("{}", samples.error().message);
            return 2;
        }
        if (chunks > 1) kidi::cli::log::info("transcribing chunk {}/{}", chunk + 1, chunks);
        auto result = transcriber->transcribe(*samples, reader->sample_rate(), options);
        if (!result) {
            kidi::cli::log::error("chunk {}: {}", chunk + 1, result.error().message);
            return result.error().code == kidi::ErrorCode::INVALID_ARGUMENT ? 2 : 1;
        }
        if (result->token_ids.size() == maximum_tokens) {
            kidi::cli::log::warn(
                "chunk {} reached --max-new-tokens {}; output may be incomplete. "
                "Increase the token limit (up to 444) or reduce --chunk-seconds",
                chunk + 1, maximum_tokens);
            if (chunks > 1) return 1;
        }
        *output << result->text << '\n' << std::flush;
        if (!*output) {
            kidi::cli::log::error("cannot write transcription: {}", output_path);
            return 1;
        }
        stats.feature_ns += result->stats.feature_ns;
        stats.encode_ns += result->stats.encode_ns;
        stats.decode_ns += result->stats.decode_ns;
        stats.preparation_ns += result->stats.preparation_ns;
        tokens += result->token_ids.size();
        if (language.empty())
            language = result->language;
        else if (language != result->language)
            language = "mixed";
    }
    if (arguments.get<bool>("profile"))
        std::cerr << "kidi_profile|backend=" << backend_label(device) << "|device=" << kidi::tensor::to_string(device)
                  << "|threads=" << threads << "|model_load_ns=" << load_ns << "|feature_ns=" << stats.feature_ns
                  << "|encoder_ns=" << stats.encode_ns << "|decode_ns=" << stats.decode_ns
                  << "|preparation_ns=" << stats.preparation_ns << "|tokens=" << tokens << "|language=" << language
                  << "|chunks=" << chunks << '\n';
    return 0;
}

auto synthesize_command(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory) -> int {
    std::string model_type;
    if (const auto status = prepare_command(arguments, directory, kidi::model::Task::TTS, &model_type)) return status;
    const auto device =
        inference_device(arguments, kidi::tensor::TransformerSupport::QUANTIZED, true, model_type == "kokoro");
    std::optional<float> duration_seconds;
    const auto duration = arguments.get<std::string>("duration");
    if (duration != "auto") {
        auto seconds = kidi::inference::parse_synthesis_number(duration);
        if (!seconds || *seconds <= 0) {
            kidi::cli::log::error("duration must be 'auto' or a positive number of seconds");
            return 2;
        }
        duration_seconds = *seconds;
    }
    auto voice = kidi::inference::parse_voice_attributes(arguments.get<std::vector<std::string>>("voice"));
    if (!voice) {
        kidi::cli::log::error("{}", voice.error().message);
        return 2;
    }
    const auto started = std::chrono::steady_clock::now();
    auto synthesizer = kidi::inference::Synthesizer::load(directory, device);
    if (!synthesizer) {
        kidi::cli::log::error("{}", synthesizer.error().message);
        return 1;
    }
    const auto load_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
    kidi::inference::SynthesisOptions options{
        .duration_seconds = duration_seconds,
        .language = arguments.get<std::string>("language"),
        .voice = std::move(*voice),
        .steps = arguments.get<std::size_t>("steps"),
        .guidance_scale = arguments.get<float>("guidance_scale"),
        .time_shift = arguments.get<float>("time_shift"),
        .layer_penalty = arguments.get<float>("layer_penalty"),
        .position_temperature = arguments.get<float>("position_temperature"),
        .seed = arguments.get<std::uint64_t>("seed"),
    };
    auto result = synthesizer->synthesize(arguments.get<std::string>("text"), options);
    if (!result) {
        kidi::cli::log::error("{}", result.error().message);
        return result.error().code == kidi::ErrorCode::INVALID_ARGUMENT ? 2 : 1;
    }
    const auto output = arguments.get<std::filesystem::path>("output");
    auto saved = kidi::audio::save_wav(output, result->samples, result->sample_rate);
    if (!saved) {
        kidi::cli::log::error("{}", saved.error().message);
        return 1;
    }
    if (arguments.contains("tokens_output")) {
        const auto path = arguments.get<std::filesystem::path>("tokens_output");
        std::ofstream tokens(path);
        if (!tokens) {
            kidi::cli::log::error("cannot create audio-token output: {}", path.string());
            return 1;
        }
        tokens << nlohmann::json(result->tokens).dump() << '\n';
        if (!tokens) {
            kidi::cli::log::error("cannot write audio-token output: {}", path.string());
            return 1;
        }
    }
    if (arguments.get<bool>("profile"))
        std::cerr << "kidi_profile|backend=" << synthesizer->execution()
                  << "|threads=" << kidi::runtime::ynn::thread_count() << "|model_load_ns=" << load_ns
                  << "|generation_ns=" << result->stats.generation_ns << "|decode_ns=" << result->stats.decode_ns
                  << "|preparation_ns=" << result->stats.preparation_ns
                  << "|audio_tokens=" << result->stats.audio_tokens << "|chunks=" << result->stats.chunks
                  << "|samples=" << result->samples.size() << "|sample_rate=" << result->sample_rate << '\n';
    return 0;
}

} // namespace

auto kidi::cli::main(int argc, const char* const argv[]) -> int {
    kidi::cli::log::init_from_env();
    kidi::cli::ArgumentParser parser("kidi", "Run translation, transcription, text generation and TTS models.");
    parser.version("kidi " + std::string(kidi::version()));
    auto& commands = parser.add_subparsers().required();
#ifdef KIDI_HAS_CODE
    auto& code_parser = commands.add_parser("code", "native coding agent with shared background inference");
    kidi::cli::configure_managed_code_parser(code_parser);
#endif
    auto& index_parser = commands.add_parser("index", "build a reusable CPU HNSW index for a model");
    kidi::cli::configure_index_parser(index_parser);
    auto& translate_parser = commands.add_parser("translate", "translate one Moses-tokenized line per input line");
    translate_parser.description("Translate with an RTG model. Output preserves input order.");
    auto& generate_parser = commands.add_parser("generate", "generate one reply per JSONL chat request");
    generate_parser.description("Generate with a chat model reading JSONL messages. Output preserves input order.");
    auto& chat_parser = commands.add_parser("chat", "interactive multi-turn chat with streaming replies");
    chat_parser.description("Chat in the terminal with a loaded model. Type /help for shell commands.");
    auto& serve_parser = commands.add_parser("serve", "serve an OpenAI-compatible chat completions API");
    serve_parser.description("Serve /v1/chat/completions with streaming, reasoning, and tool calls over HTTP.");
    serve_parser.add_argument("--host").default_value(std::string("127.0.0.1"));
    serve_parser.add_argument("--port").default_value<std::int32_t>(8000);
    serve_parser.add_argument("--served-model-name")
        .dest("served_model_name")
        .default_value(std::string{})
        .help("model id reported to clients (default: --model)");
    serve_parser.add_argument("--max-active").dest("max_active").default_value<std::size_t>(1);
    serve_parser.add_argument("--queue-size").dest("queue_size").default_value<std::size_t>(64);
    serve_parser.add_argument("--prefix-cache-mib")
        .dest("prefix_cache_mib")
        .default_value<std::size_t>(0)
        .help("keep the last request's KV state (up to this size) to reuse shared prompt prefixes");
#ifdef KIDI_HAS_CODE
    kidi::cli::configure_managed_serve_parser(serve_parser);
#endif
    auto& transcribe_parser = commands.add_parser("transcribe", "transcribe a 16 kHz WAV file with Whisper");
    transcribe_parser.description(
        "Transcribe or translate speech with Whisper Tiny, Base, Small, Large-v3, or Large-v3-Turbo. Long WAV files "
        "are processed "
        "in bounded windows with one loaded model.");
    auto& synthesize_parser = commands.add_parser("synthesize", "synthesize speech from text");
    synthesize_parser.description("Synthesize a mono WAV file with a TTS model.");
    chat_parser.add_argument("--system").default_value(std::string{}).help("system instruction");
    chat_parser.add_argument("--color")
        .default_value(std::string("auto"))
        .choices({"auto", "always", "never"})
        .help("terminal colors; auto respects NO_COLOR");
    generate_parser.add_argument("--max-active")
        .dest("max_active")
        .type<std::size_t>()
        .metavar("N")
        .help("active concurrent requests (default: 1 with --assistant-model, else 4)");
    generate_parser.add_argument("--queue-size").dest("queue_size").default_value<std::size_t>(64);
    for (auto* command_parser :
         {&translate_parser, &generate_parser, &chat_parser, &serve_parser, &transcribe_parser, &synthesize_parser}) {
        command_parser->add_argument("-m", "--model")
            .type<std::filesystem::path>()
            .required()
            .metavar("MODEL")
            .help("local directory or @owner/model");
        command_parser->add_argument("-c", "--cache")
            .type<std::filesystem::path>()
            .default_value(kidi::checkpoint::Hub::default_cache())
            .help("Hugging Face model cache directory");
        command_parser->add_argument("--backend")
            .default_value(std::string("auto"))
            .help("auto (CUDA, Metal, then CPU), ynnpack, mps, vulkan, npu, or cuda[:INDEX]");
        command_parser->add_argument("-j", "--threads").default_value<std::int32_t>(4);
        command_parser->add_argument("--profile").action(kidi::cli::Action::STORE_TRUE);
    }
    transcribe_parser.add_argument("-i", "--in")
        .dest("input")
        .type<std::filesystem::path>()
        .required()
        .metavar("WAV")
        .help("16 kHz PCM WAV input");
    transcribe_parser.add_argument("-o", "--out")
        .dest("output")
        .default_value(std::string("-"))
        .metavar("FILE")
        .help("transcript file, or '-' for stdout");
    transcribe_parser.add_argument("--language").default_value(std::string("auto")).metavar("CODE");
    transcribe_parser.add_argument("--task")
        .default_value(std::string("transcribe"))
        .choices({"transcribe", "translate"});
    transcribe_parser.add_argument("--max-new-tokens").dest("max_new_tokens").default_value<std::size_t>(128);
    transcribe_parser.add_argument("--chunk-seconds")
        .dest("chunk_seconds")
        .default_value<std::size_t>(30)
        .help("audio seconds per window (1 to 30); output is flushed after each window");
    synthesize_parser.add_argument("--text").required().help("text to synthesize");
    synthesize_parser.add_argument("-o", "--out")
        .dest("output")
        .type<std::filesystem::path>()
        .required()
        .metavar("WAV")
        .help("24 kHz mono WAV output");
    synthesize_parser.add_argument("--language").default_value(std::string{}).metavar("NAME");
    synthesize_parser.add_argument("--duration")
        .default_value(std::string("auto"))
        .metavar("auto|SECONDS")
        .help("automatic speech-length estimate or an exact duration in seconds");
    synthesize_parser.add_argument("--voice")
        .action(kidi::cli::Action::APPEND)
        .metavar("KEY=VALUE")
        .help("repeatable model-specific voice attribute");
    synthesize_parser.add_argument("--steps").default_value<std::size_t>(32);
    synthesize_parser.add_argument("--guidance-scale").dest("guidance_scale").default_value(2.F);
    synthesize_parser.add_argument("--time-shift").dest("time_shift").default_value(0.1F);
    synthesize_parser.add_argument("--layer-penalty").dest("layer_penalty").default_value(5.F);
    synthesize_parser.add_argument("--position-temperature").dest("position_temperature").default_value(5.F);
    synthesize_parser.add_argument("--seed").default_value<std::uint64_t>(0);
    synthesize_parser.add_argument("--tokens-out")
        .dest("tokens_output")
        .type<std::filesystem::path>()
        .metavar("JSON")
        .help("optional generated audio-token dump");
    for (auto* command_parser : {&generate_parser, &chat_parser, &serve_parser}) {
        command_parser->add_argument("--cache-tokens")
            .dest("cache_tokens")
            .default_value<std::size_t>(16384)
            .help("budget for reserved dense context tokens");
        command_parser->add_argument("--max-new-tokens").dest("max_new_tokens").default_value<std::size_t>(0);
        command_parser->add_argument("--context-size").dest("context_size").default_value<std::size_t>(0);
        command_parser->add_argument("--prefill-chunk-size").dest("prefill_chunk_size").default_value<std::size_t>(128);
        command_parser->add_argument("--ignore-eos").dest("ignore_eos").action(kidi::cli::Action::STORE_TRUE);
        command_parser->add_argument("--full-attention-cache")
            .dest("full_attention_cache")
            .action(kidi::cli::Action::STORE_TRUE)
            .help("disable CPU local-history cropping for the previous numerical path");
        command_parser->add_argument("--precision")
            .default_value(std::string("checkpoint"))
            .choices({"checkpoint", "bf16", "q8", "q4", "vq3d-hybrid"})
            .help(
                "weight and compute policy: checkpoint (stored format), bf16 (reference), q8 (INT8 per channel, "
                "packed at load), q4 (INT4 in groups of 128, packed at load), vq3d-hybrid (Inkling-2 VQ experts)");
        command_parser->add_argument("--kv-cache-precision")
            .dest("kv_cache_precision")
            .default_value(std::string("auto"))
            .choices({"auto", "fp32", "bf16", "int8"})
            .help("KV-cache storage policy; auto picks the model and device default");
        command_parser->add_argument("--packed-prefill")
            .dest("packed_prefill")
            .action(kidi::cli::Action::STORE_TRUE)
            .help("use packed GEMM for prefill; default may cache FP16 matrices for native QAT");
        command_parser->add_argument("--dense-lm-head")
            .dest("dense_lm_head")
            .action(kidi::cli::Action::STORE_TRUE)
            .help("disable checkpoint lm_head.hnsw discovery and use the exact dense head");
        command_parser->add_argument("--assistant-model")
            .dest("assistant_model")
            .default_value(std::filesystem::path{})
            .metavar("MODEL")
            .help("local directory or @owner/model assistant for speculative decoding");
        command_parser->add_argument("--draft-tokens")
            .dest("draft_tokens")
            .type<std::size_t>()
            .metavar("N")
            .help("adaptive draft cap, 1 to 16; starts at 1 (default: 3; requires assistant or prompt lookup)");
        command_parser->add_argument("--prompt-lookup")
            .dest("prompt_lookup")
            .action(kidi::cli::Action::STORE_TRUE)
            .help("draft repeated prompt/history tokens for CPU Inkling-2; greedy decoding only");
    }

    auto& inspect_parser = commands.add_parser("inspect", "inspect a model package");
    inspect_parser.description("Inspect an RTG, Gemma 4, OmniVoice, Kokoro, or direct Whisper model package.");
    inspect_parser.add_argument("-m", "--model")
        .type<std::filesystem::path>()
        .required()
        .metavar("MODEL")
        .help("local directory or @owner/model");
    inspect_parser.add_argument("-c", "--cache")
        .type<std::filesystem::path>()
        .default_value(kidi::checkpoint::Hub::default_cache())
        .help("Hugging Face model cache directory");

    translate_parser.add_argument("-b", "--beam-size").type<std::int32_t>().metavar("N").help("RTG beam size");
    translate_parser.add_argument("-x", "--max-extra-tokens")
        .dest("maximum_extra_tokens")
        .type<std::int32_t>()
        .metavar("N")
        .help("maximum target tokens beyond source length");
    translate_parser.add_argument("-a", "--length-penalty")
        .type<float>()
        .metavar("ALPHA")
        .help("Wu length penalty exponent");
    translate_parser.add_argument("--score").action(kidi::cli::Action::STORE_TRUE).help("append RTG hypothesis score");
    translate_parser.add_argument("--stats")
        .action(kidi::cli::Action::STORE_TRUE)
        .help("emit machine-readable generation stats to stderr");
    translate_parser.add_argument("--batch-size")
        .type<std::int32_t>()
        .default_value(std::int32_t{1})
        .metavar("N")
        .help("independent sentences per batch (1 to 256; CPU executes serially)");
    translate_parser.add_argument("--batch-window")
        .type<std::int32_t>()
        .metavar("N")
        .help("bounded input window for length grouping (default: batch-size; maximum 4096)");
    for (auto* command_parser : {&translate_parser, &generate_parser}) {
        command_parser->add_argument("-i", "--in")
            .dest("input")
            .default_value(std::string("-"))
            .metavar("FILE")
            .help("input file, or '-' for stdin");
        command_parser->add_argument("-o", "--out")
            .dest("output")
            .default_value(std::string("-"))
            .metavar("FILE")
            .help("output file, or '-' for stdout");
    }

    try {
        const auto arguments = parser.parse_args(argc, argv);
        if (arguments.should_exit()) {
            std::cout << arguments.output();
            if (arguments.status() == kidi::cli::ParseStatus::VERSION) {
                std::vector<std::string> unavailable;
                for (const auto& backend : kidi::tensor::BackendRegistry::instance().backends()) {
                    if (!backend.execution_available) {
                        unavailable.push_back(backend.name);
                        continue;
                    }
                    std::cout << "available: " << backend.name;
#ifdef KIDI_CUDA_TOOLKIT_VERSION
                    if (backend.device_kind == kidi::tensor::DeviceKind::CUDA)
                        std::cout << ' ' << KIDI_CUDA_TOOLKIT_VERSION;
#endif
                    std::cout << '\n';
                }
                if (!unavailable.empty()) {
                    std::cout << "unavailable: ";
                    for (std::size_t i = 0; i < unavailable.size(); ++i)
                    {
                        std::cout << (i ? ", " : "") << unavailable[i];
                    }
                    std::cout << '\n';
                }
            }
            return 0;
        }
        const auto& command = arguments.get<std::string>("command");
#ifdef KIDI_HAS_CODE
        if (command == "code") return kidi::cli::managed_code_command(arguments);
        if (command == "serve") kidi::cli::protect_managed_lock(arguments);
#endif
        std::filesystem::path directory;
        const auto cache = arguments.contains("cache") ? arguments.get<std::filesystem::path>("cache")
                                                       : kidi::checkpoint::Hub::default_cache();
        if (const auto status = resolve_hub_argument(arguments.get<std::filesystem::path>("model"), cache, directory))
            return status;
        if (command == "index") return kidi::cli::index_command(arguments, directory);
        if (command == "inspect") return inspect(directory);
        if (command == "translate") return translate_command(arguments, directory);
        if (command == "generate") return generate_command(arguments, directory);
        if (command == "chat") return chat_command(arguments, directory);
        if (command == "serve") return serve_command(arguments, directory);
        if (command == "transcribe") return transcribe_command(arguments, directory);
        if (command == "synthesize") return synthesize_command(arguments, directory);
        throw std::logic_error("unhandled command: " + command);
    } catch (const kidi::cli::ParseError& error) {
        std::cerr << error.usage();
        kidi::cli::log::error("{}", error.what());
        return 2;
    } catch (const std::exception& error) {
        kidi::cli::log::error("{}", error.what());
        return 1;
    }
}