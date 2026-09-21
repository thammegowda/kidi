#include <cstddef>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <deque>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "kidi/cli/argparse.h"
#include "kidi/cli/main.h"
#include "kidi/cli/interactive.h"
#include "kidi/core/version.h"
#include "kidi/model/package.h"
#include "kidi/model/config.h"
#include "kidi/inference/translator.h"
#include "kidi/inference/generator.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/tensor/backend.h"

namespace {

auto inference_device(const kidi::cli::Namespace& arguments) -> kidi::tensor::Device {
    const auto& backend = arguments.get<std::string>("backend");
    return backend == "mps" || (backend == "auto" && kidi::module_device == kidi::tensor::Device::apple_gpu())
               ? kidi::tensor::Device::apple_gpu()
               : kidi::tensor::Device::cpu();
}

/// Stable label for machine-readable stderr records.
auto backend_label(kidi::tensor::Device device) -> std::string_view {
    return device == kidi::tensor::Device::apple_gpu() ? "mps_gpu" : "ynnpack_cpu";
}

auto inspect(const std::filesystem::path& directory) -> int {
    auto document = kidi::model::load_config(directory / "model.yaml");
    if (!document) {
        spdlog::error("{}", document.error().message);
        return 1;
    }
    if ((*document)["model"]["type"].as<std::string>() == "gemma4_text") {
        auto validation = kidi::model::Gemma4Impl::validate_config((*document)["model"]);
        if (!validation) {
            spdlog::error("{}", validation.error().message);
            return 1;
        }
        auto weights = kidi::model::Weights::load((*document)["weights_file"].as<std::string>());
        if (!weights) {
            spdlog::error("{}", weights.error().message);
            return 1;
        }
        const bool qat = static_cast<bool>((*document)["model"]["quantization_config"]);
        const auto embedding = weights->tensor(qat ? "model.language_model.embed_tokens.embedding_quantized"
                                                   : "model.language_model.embed_tokens.weight");
        if (!embedding) {
            spdlog::error("{}", embedding.error().message);
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
    auto package = kidi::model::Package::load(directory);
    if (!package) {
        spdlog::error("{}", package.error().message);
        return 1;
    }
    const auto& config = package->config();
    auto validation = kidi::model::TransformerImpl::validate_config(config["model"]);
    if (!validation) {
        spdlog::error("{}", validation.error().message);
        return 1;
    }
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

auto chat_messages(const nlohmann::json& request) -> kidi::Result<std::vector<kidi::text::ChatMessage>> {
    const auto invalid = [](std::string message) {
        return std::unexpected(kidi::Error{kidi::ErrorCode::INVALID_ARGUMENT, std::move(message)});
    };
    if (!request.is_object() || !request.contains("messages") || !request["messages"].is_array() ||
        request["messages"].empty())
        return invalid("expected an object with a nonempty messages array");
    for (const auto& [key, value] : request.items())
        if (key != "messages" && key != "id" && key != "max_tokens")
            return invalid("unsupported request field: " + key);
    if (request.contains("id") && !request["id"].is_string() && !request["id"].is_number_integer())
        return invalid("id must be a string or integer");
    if (request.contains("max_tokens") &&
        (!request["max_tokens"].is_number_integer() || request["max_tokens"] <= 0 || request["max_tokens"] > INT32_MAX))
        return invalid("max_tokens must be a positive integer within int32 bounds");
    std::vector<kidi::text::ChatMessage> messages;
    for (const auto& message : request["messages"]) {
        if (!message.is_object() || message.size() != 2 || !message.contains("role") || !message["role"].is_string() ||
            !message.contains("content"))
            return invalid("each message must contain role and content only");
        std::string content;
        if (message["content"].is_string()) {
            content = message["content"].get<std::string>();
        } else if (message["content"].is_array()) {
            for (const auto& part : message["content"]) {
                if (!part.is_object() || part.size() != 2 || !part.contains("type") || part["type"] != "text" ||
                    !part.contains("text") || !part["text"].is_string())
                    return invalid("content parts must be text objects; multimodal input is unsupported");
                content += part["text"].get<std::string>();
            }
        } else {
            return invalid("content must be a string or an array of text parts");
        }
        messages.push_back({message["role"].get<std::string>(), std::move(content)});
    }
    return messages;
}

struct ChatSession {
    kidi::inference::Generator generator;
    kidi::inference::GenerationOptions options;
    std::int64_t load_ns;
    kidi::tensor::Device device;
};

auto load_chat_session(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory)
    -> kidi::Result<ChatSession> {
    const auto device = inference_device(arguments);
    const auto started = std::chrono::steady_clock::now();
    auto loaded = kidi::inference::Generator::load(directory, device, arguments.get<std::int32_t>("weight_bits"),
                                                   arguments.get<std::int32_t>("group_size"),
                                                   arguments.get<bool>("packed_prefill"));
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

auto generate_chat(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory, std::istream& input,
                   std::ostream& output) -> int {
    try {
        const auto started = std::chrono::steady_clock::now();
        auto session = load_chat_session(arguments, directory);
        if (!session) {
            spdlog::error("{}", session.error().message);
            return 1;
        }
        auto& generator = session->generator;
        const auto& options = session->options;
        const auto device = session->device;
        const auto load_ns = session->load_ns;
        const kidi::inference::ServingOptions limits{.maximum_active = arguments.get<std::size_t>("max_active"),
                                                     .maximum_requests = arguments.get<std::size_t>("queue_size"),
                                                     .cache_token_budget = arguments.get<std::size_t>("cache_tokens"),
                                                     .prefill_tokens_per_step = options.prefill_chunk_size};
        auto configured = generator.configure_serving(limits);
        if (!configured) {
            spdlog::error("{}", configured.error().message);
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
                        spdlog::error("failed reading chat input");
                        return 1;
                    }
                    ended = true;
                    break;
                }
                ++line_number;
                const auto request = nlohmann::json::parse(line, nullptr, false);
                auto messages = chat_messages(request);
                if (!messages) {
                    spdlog::error("input line {}: {}", line_number, messages.error().message);
                    return 2;
                }
                auto request_options = options;
                if (request.contains("max_tokens"))
                    request_options.maximum_new_tokens = request["max_tokens"].get<std::size_t>();
                auto queued = generator.enqueue_chat(*messages, request_options);
                if (!queued) {
                    spdlog::error("input line {}: {}", line_number, queued.error().message);
                    return queued.error().code == kidi::ErrorCode::INVALID_ARGUMENT ? 2 : 1;
                }
                ordered.push_back({request.value("id", nlohmann::json(*queued)), {}});
            }
            if (!generator.pending_requests()) break;
            auto step = generator.step();
            if (!step) {
                spdlog::error("{}", step.error().message);
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
                spdlog::error("failed writing generation results");
                return 1;
            }
        }
        if (arguments.get<bool>("profile")) {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
                    .count();
            std::cerr << "kidi_serving|requests=" << completed << "|max_active=" << limits.maximum_active
                      << "|backend=" << backend_label(device) << "|load_ns=" << load_ns
                      << "|native_qat=" << generator.native_qat()
                      << "|packed_prefill=" << arguments.get<bool>("packed_prefill")
                      << "|cache_tokens=" << limits.cache_token_budget
                      << "|peak_reserved_cache_tokens=" << peak_cache_tokens << "|prefill_ns=" << prefill_ns
                      << "|decode_ns=" << decode_ns << "|preparation_ns=" << preparation_ns
                      << "|generation_ns=" << elapsed << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
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
        spdlog::error("batch size must be between 1 and 256");
        return 2;
    }
    if (batch_window < batch_size || batch_window > 4096) {
        spdlog::error("batch window must be between batch-size and 4096");
        return 2;
    }
    auto* profile_ptr = arguments.get<bool>("profile") ? &profile : nullptr;
    const auto device = inference_device(arguments);
    auto translator = kidi::inference::Translator::load(directory, device, profile_ptr, batch_size);
    if (!translator) {
        spdlog::error("{}", translator.error().message);
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
            spdlog::error("batch ending at input line {}: {}", line_number, translations.error().message);
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
            spdlog::error("cannot write output: {}", arguments.get<std::string>("output"));
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
        spdlog::error("cannot read input: {}", arguments.get<std::string>("input"));
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
                     std::string_view expected_type) -> int {
    const auto threads = arguments.get<std::int32_t>("threads");
    if (threads <= 0) {
        spdlog::error("thread count must be positive");
        return 2;
    }
    kidi::runtime::ynn::set_thread_count(threads);
    auto config = kidi::model::load_config(directory / "model.yaml");
    if (!config) {
        spdlog::error("{}", config.error().message);
        return 1;
    }
    const auto type = (*config)["model"]["type"].as<std::string>();
    if (type != expected_type) {
        spdlog::error("{} expects a {} model, but this package is {}", arguments.get<std::string>("command"),
                      expected_type, type);
        return 2;
    }
    return 0;
}

/// Opens the -i/-o files, defaulting to standard input and output.
auto run_with_streams(const kidi::cli::Namespace& arguments,
                      const std::function<int(std::istream&, std::ostream&)>& body) -> int {
    const auto& input_path = arguments.get<std::string>("input");
    const auto& output_path = arguments.get<std::string>("output");
    if (input_path != "-" && output_path != "-" && std::filesystem::exists(output_path) &&
        std::filesystem::equivalent(input_path, output_path)) {
        spdlog::error("input and output must be different files");
        return 2;
    }
    std::ifstream input_file;
    std::ofstream output_file;
    std::istream* input = &std::cin;
    std::ostream* output = &std::cout;
    if (input_path != "-") {
        input_file.open(input_path);
        if (!input_file) {
            spdlog::error("cannot open input file: {}", input_path);
            return 1;
        }
        input = &input_file;
    }
    if (output_path != "-") {
        output_file.open(output_path);
        if (!output_file) {
            spdlog::error("cannot open output file: {}", output_path);
            return 1;
        }
        output = &output_file;
    }
    return body(*input, *output);
}

auto translate_command(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory) -> int {
    if (const auto status = prepare_command(arguments, directory, "rtg_transformer_nmt")) return status;
    return run_with_streams(arguments, [&](std::istream& input, std::ostream& output) {
        return generate_translation(arguments, directory, input, output);
    });
}

auto generate_command(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory) -> int {
    if (const auto status = prepare_command(arguments, directory, "gemma4_text")) return status;
    return run_with_streams(arguments, [&](std::istream& input, std::ostream& output) {
        return generate_chat(arguments, directory, input, output);
    });
}

auto chat_command(const kidi::cli::Namespace& arguments, const std::filesystem::path& directory) -> int {
    if (const auto status = prepare_command(arguments, directory, "gemma4_text")) return status;
    auto session = load_chat_session(arguments, directory);
    if (!session) {
        spdlog::error("{}", session.error().message);
        return 1;
    }
    return kidi::cli::interactive_chat(session->generator, session->options, arguments, session->load_ns);
}

} // namespace

auto kidi::cli::main(int argc, const char* const argv[], const ModelResolver& resolve_model) -> int {
    auto logger = spdlog::get("kidi");
    if (!logger) logger = spdlog::stderr_color_mt("kidi");
    spdlog::set_default_logger(std::move(logger));
    spdlog::set_pattern("[%n] [%l] %v");
    spdlog::cfg::load_env_levels();
    kidi::cli::ArgumentParser parser("kidi", "Run translation and text generation models.");
    parser.version("kidi " + std::string(kidi::version()));
    auto& commands = parser.add_subparsers().required();
    auto& translate_parser = commands.add_parser("translate", "translate one Moses-tokenized line per input line");
    translate_parser.description("Translate with an RTG model. Output preserves input order.");
    auto& generate_parser = commands.add_parser("generate", "generate one reply per JSONL chat request");
    generate_parser.description("Generate with a chat model reading JSONL messages. Output preserves input order.");
    auto& chat_parser = commands.add_parser("chat", "interactive multi-turn chat with streaming replies");
    chat_parser.description("Chat in the terminal with a loaded model. Type /help for shell commands.");
    chat_parser.add_argument("--system").default_value(std::string{}).help("system instruction");
    chat_parser.add_argument("--color")
        .default_value(std::string("auto"))
        .choices({"auto", "always", "never"})
        .help("terminal colors; auto respects NO_COLOR");
    generate_parser.add_argument("--max-active").dest("max_active").default_value<std::size_t>(4);
    generate_parser.add_argument("--queue-size").dest("queue_size").default_value<std::size_t>(64);
    for (auto* command_parser : {&translate_parser, &generate_parser, &chat_parser}) {
        command_parser->add_argument("-m", "--model")
            .type<std::filesystem::path>()
            .required()
            .metavar("MODEL")
            .help("local directory or @owner/model (Python launcher with kidi[hf])");
        command_parser->add_argument("-c", "--cache")
            .type<std::filesystem::path>()
            .default_value(std::filesystem::path("~/.cache/kidi/model-hub"))
            .help("Hugging Face model cache directory");
        command_parser->add_argument("--backend")
            .default_value(std::string("auto"))
            .choices({"auto", "ynnpack", "mps"});
        command_parser->add_argument("-j", "--threads").default_value<std::int32_t>(4);
        command_parser->add_argument("--profile").action(kidi::cli::Action::STORE_TRUE);
    }
    for (auto* command_parser : {&generate_parser, &chat_parser}) {
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
        command_parser->add_argument("--weight-bits")
            .dest("weight_bits")
            .default_value<std::int32_t>(0)
            .help("0: original weights; 4 or 8: load-time groupwise packing");
        command_parser->add_argument("--group-size").dest("group_size").default_value<std::int32_t>(128);
        command_parser->add_argument("--packed-prefill")
            .dest("packed_prefill")
            .action(kidi::cli::Action::STORE_TRUE)
            .help("use packed GEMM for prefill; default may cache FP16 matrices for native QAT");
    }

    auto& inspect_parser = commands.add_parser("inspect", "inspect a model package");
    inspect_parser.description("Inspect an RTG or Gemma 4 model package.");
    inspect_parser.add_argument("-m", "--model")
        .type<std::filesystem::path>()
        .required()
        .metavar("MODEL")
        .help("local directory or @owner/model (Python launcher with kidi[hf])");
    inspect_parser.add_argument("-c", "--cache")
        .type<std::filesystem::path>()
        .default_value(std::filesystem::path("~/.cache/kidi/model-hub"))
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
            return 0;
        }
        const auto& command = arguments.get<std::string>("command");
        auto directory = arguments.get<std::filesystem::path>("model");
        const auto reference = directory.string();
        if (reference.starts_with('@')) {
            if (!resolve_model) {
                spdlog::error("Hub models require the Python launcher: install 'kidi[hf]' and use python -m kidi");
                return 2;
            }
            auto resolved =
                resolve_model(std::string_view(reference).substr(1), arguments.get<std::filesystem::path>("cache"));
            if (!resolved) {
                spdlog::error("{}", resolved.error().message);
                return 1;
            }
            directory = std::move(*resolved);
        }
        if (command == "inspect") return inspect(directory);
        if (command == "translate") return translate_command(arguments, directory);
        if (command == "generate") return generate_command(arguments, directory);
        if (command == "chat") return chat_command(arguments, directory);
        throw std::logic_error("unhandled command: " + command);
    } catch (const kidi::cli::ParseError& error) {
        std::cerr << error.usage();
        spdlog::error("{}", error.what());
        return 2;
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return 1;
    }
}