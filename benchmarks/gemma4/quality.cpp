#include "kidi/model/gemma4.h"
#include "kidi/model/config.h"
#include "kidi/text/tokenizer.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/inference/generator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

auto main(int argc, char** argv) -> int {
    using namespace kidi;
    try {
        if (argc < 5 || argc > 7) {
            std::cerr
                << "usage: kidi_gemma4_quality MODEL cpu|gpu BITS GROUP [REFERENCE_JSON_OR_SAFETENSORS [CHUNK]]\n";
            return 2;
        }
        const std::filesystem::path directory(argv[1]);
        const auto device = std::string_view(argv[2]) == "cpu" ? tensor::Device::cpu() : tensor::Device::apple_gpu();
        const auto bits = std::stoi(argv[3]), group = std::stoi(argv[4]);
        const auto chunk = argc == 7 ? std::stoi(argv[6]) : 1;
        if ((std::string_view(argv[2]) != "cpu" && std::string_view(argv[2]) != "gpu") ||
            (bits != 0 && bits != 4 && bits != 8) || group <= 0 || chunk <= 0)
            return 2;
        runtime::ynn::set_thread_count(4);
        if (argc == 6 && std::string_view(argv[5]) == "serving") {
            auto generator = ops::require(inference::Generator::load(directory, device, bits, group));
            const std::array<std::string, 4> prompts{
                "What is the capital of France?", "What is two plus three?",
                "Explain why sunlight contains many colors and how light scatters in the atmosphere.",
                "Translate bonjour into English."};
            std::array<inference::GenerationOptions, 4> options;
            std::vector<inference::TextGeneration> serial;
            for (std::size_t row = 0; row < prompts.size(); ++row) {
                options[row] = {.maximum_new_tokens = row == 0 ? 3U : 12U,
                                .context_size = 128,
                                .prefill_chunk_size = 8,
                                .stream_text = true};
                serial.push_back(ops::require(generator.generate(prompts[row], options[row])));
            }
            ops::require(generator.configure_serving(
                {.maximum_active = 2, .maximum_requests = 3, .cache_token_budget = 256, .prefill_tokens_per_step = 8}));
            std::array<std::uint64_t, 4> ids{};
            for (std::size_t row = 0; row < 3; ++row)
                ids[row] = ops::require(generator.enqueue(prompts[row], options[row]));
            if (generator.enqueue(prompts[3], options[3])) throw std::runtime_error("serving backpressure failed");
            std::array<std::vector<std::int32_t>, 4> streamed;
            std::array<std::string, 4> streamed_text;
            std::array<bool, 4> finished{};
            std::size_t completed = 0, steps = 0;
            while (generator.pending_requests()) {
                if (++steps > 100) throw std::runtime_error("serving failed to make bounded progress");
                const auto result = ops::require(generator.step());
                if (result.active_requests > 2 || result.reserved_cache_tokens > 256)
                    throw std::runtime_error("serving reservation exceeded");
                for (const auto& event : result.events) {
                    const auto found = std::ranges::find(ids, event.request_id);
                    if (found == ids.end()) throw std::runtime_error("unknown serving result id");
                    const auto row = static_cast<std::size_t>(found - ids.begin());
                    if (finished[row]) throw std::runtime_error("duplicate completion or post-completion token");
                    if (event.token) streamed[row].push_back(*event.token);
                    streamed_text[row] += event.text;
                    if (event.completed) {
                        if (event.completed->generation.token_ids != serial[row].generation.token_ids ||
                            event.completed->generation.decoder_steps != serial[row].generation.decoder_steps ||
                            streamed[row] != serial[row].generation.token_ids ||
                            streamed_text[row] != serial[row].text || event.completed->text != streamed_text[row])
                            throw std::runtime_error("serving differs from serial at row " + std::to_string(row));
                        finished[row] = true;
                        ++completed;
                    }
                }
                if (completed && !ids[3]) ids[3] = ops::require(generator.enqueue(prompts[3], options[3]));
            }
            if (completed != 4) throw std::runtime_error("serving dropped a request");
            const auto running = ops::require(generator.enqueue(prompts[2], options[2]));
            ops::require(generator.step());
            const auto waiting = ops::require(generator.enqueue(prompts[0], options[0]));
            ops::require(generator.cancel(running));
            ops::require(generator.cancel(waiting));
            if (generator.pending_requests() || generator.cancel(running))
                throw std::runtime_error("cancellation failed");
            const auto empty = ops::require(generator.step());
            if (!empty.events.empty() || empty.reserved_cache_tokens)
                throw std::runtime_error("cancel leaked reservation");
            ops::require(generator.configure_serving(
                {.maximum_active = 2, .maximum_requests = 3, .cache_token_budget = 192, .prefill_tokens_per_step = 8}));
            auto oversized = options[0];
            oversized.context_size = 256;
            if (generator.enqueue(prompts[0], oversized) || generator.pending_requests())
                throw std::runtime_error("oversized reservation accepted");
            const auto first = ops::require(generator.enqueue(prompts[2], options[2]));
            const auto second = ops::require(generator.enqueue(prompts[0], options[0]));
            const auto constrained = ops::require(generator.step());
            if (constrained.active_requests != 1 || constrained.waiting_requests != 1 ||
                constrained.reserved_cache_tokens != 128)
                throw std::runtime_error("token budget did not constrain admission");
            if (generator.configure_serving({}) || generator.generate(prompts[0], options[0]))
                throw std::runtime_error("outstanding serving requests were not isolated");
            ops::require(generator.cancel(first));
            const auto replacement = ops::require(generator.step());
            if (replacement.active_requests != 1 || replacement.waiting_requests ||
                replacement.reserved_cache_tokens != 128)
                throw std::runtime_error("cancelled slot was not reclaimed");
            ops::require(generator.cancel(second));
            std::cout << "{\"comparison\":\"continuous_admission\",\"backend\":\"" << argv[2]
                      << "\",\"exact_outputs\":true,\"requests\":4,\"steps\":" << steps << "}\n";
            return 0;
        }
        if (argc == 6 && std::string_view(argv[5]) == "generation-batch") {
            auto generator = ops::require(inference::Generator::load(directory, device, bits, group));
            const std::array<std::string, 4> prompts{"What is the capital of France?", "What is two plus three?",
                                                     "Complete the sentence with one word: Water freezes into",
                                                     "Translate bonjour into English."};
            inference::GenerationOptions options{.maximum_new_tokens = 24, .context_size = 128};
            struct Record {
                std::uint64_t serial_ns, batch_ns, serial_decode_ns, batch_decode_ns, serial_preparation_ns,
                    batch_preparation_ns;
            };
            std::vector<Record> records;
            std::size_t tokens = 0;
            for (int repeat = 0; repeat < 5; ++repeat) {
                inference::GenerationBatch serial, batch;
                const auto run_serial = [&] {
                    for (const auto& prompt : prompts) {
                        serial.rows.push_back(ops::require(generator.generate(prompt, options)));
                        const auto& stats = serial.rows.back().stats;
                        serial.generation_ns += stats.generation_ns;
                        serial.decode_ns += stats.decode_ns;
                        serial.preparation_ns += stats.preparation_ns;
                    }
                };
                const auto run_batch = [&] { batch = ops::require(generator.generate_batch(prompts, options)); };
                if (repeat % 2) {
                    run_batch();
                    run_serial();
                } else {
                    run_serial();
                    run_batch();
                }
                if (batch.rows.size() != serial.rows.size()) throw std::runtime_error("batch result count mismatch");
                tokens = 0;
                for (std::size_t row = 0; row < serial.rows.size(); ++row) {
                    if (batch.rows[row].generation.token_ids != serial.rows[row].generation.token_ids ||
                        batch.rows[row].generation.decoder_steps != serial.rows[row].generation.decoder_steps)
                        throw std::runtime_error("batch generation differs from serial at row " + std::to_string(row));
                    tokens += batch.rows[row].stats.decode_tokens;
                }
                records.push_back({serial.generation_ns, batch.generation_ns, serial.decode_ns, batch.decode_ns,
                                   serial.preparation_ns, batch.preparation_ns});
            }
            std::cout << "{\"comparison\":\"generation_batch\",\"backend\":\"" << argv[2]
                      << "\",\"exact_outputs\":true,\"rows\":4,\"decode_tokens\":" << tokens << ",\"records\":[";
            for (std::size_t index = 0; index < records.size(); ++index) {
                if (index) std::cout << ',';
                const auto& record = records[index];
                std::cout << "{\"warmup\":" << (index < 2 ? "true" : "false") << ",\"serial_ns\":" << record.serial_ns
                          << ",\"batch_ns\":" << record.batch_ns << ",\"serial_decode_ns\":" << record.serial_decode_ns
                          << ",\"batch_decode_ns\":" << record.batch_decode_ns
                          << ",\"serial_preparation_ns\":" << record.serial_preparation_ns
                          << ",\"batch_preparation_ns\":" << record.batch_preparation_ns << '}';
            }
            std::cout << "]}\n";
            return 0;
        }
        if (argc == 6 && std::string_view(argv[5]) == "prefix") {
            auto generator = ops::require(inference::Generator::load(directory, device, bits, group));
            inference::GenerationOptions options{
                .maximum_new_tokens = 8, .context_size = 256, .prefill_chunk_size = 8, .ignore_eos = true};
            const std::string prompt =
                "Explain how sunlight contains many colors and why the sky appears blue. Compare blue and red light "
                "scattering in clear air.";
            const auto uncached = ops::require(generator.generate(prompt, options));
            options.prefix_cache_bytes = 4 * 1024 * 1024;
            const auto cold = ops::require(generator.generate(prompt, options));
            const auto hit = ops::require(generator.generate(prompt, options));
            if (cold.stats.reused_prompt_tokens || !hit.stats.reused_prompt_tokens ||
                hit.stats.prefix_cache_bytes > options.prefix_cache_bytes ||
                hit.stats.prefix_reserved_bytes > options.prefix_cache_bytes ||
                hit.stats.prefix_cache_bytes > hit.stats.prefix_reserved_bytes ||
                hit.generation.token_ids != uncached.generation.token_ids ||
                cold.generation.token_ids != uncached.generation.token_ids)
                throw std::runtime_error("prefix cold/hit contract failed");
            const auto suffix = prompt + " Give a concise explanation.";
            const auto reused_suffix = ops::require(generator.generate(suffix, options));
            if (!reused_suffix.stats.reused_prompt_tokens) throw std::runtime_error("shared prefix was not reused");
            options.prefix_cache_bytes = 0;
            const auto uncached_suffix = ops::require(generator.generate(suffix, options));
            if (uncached_suffix.stats.prefix_cache_bytes || uncached_suffix.stats.prefix_reserved_bytes ||
                uncached_suffix.stats.reused_prompt_tokens ||
                uncached_suffix.generation.token_ids != reused_suffix.generation.token_ids)
                throw std::runtime_error("prefix disable or changed-suffix parity failed");
            options.prefix_cache_bytes = 4 * 1024 * 1024;
            ops::require(generator.generate(prompt, options));
            options.prefix_cache_bytes = 1;
            const auto tiny = ops::require(generator.generate(prompt, options));
            if (tiny.stats.prefix_cache_bytes || tiny.stats.prefix_reserved_bytes || tiny.stats.reused_prompt_tokens)
                throw std::runtime_error("prefix budget shrink did not evict");
            options.prefix_cache_bytes = 4 * 1024 * 1024;
            ops::require(generator.generate(prompt, options));
            options.prefill_chunk_size = 4;
            const auto rechunked = ops::require(generator.generate(prompt, options));
            if (rechunked.stats.reused_prompt_tokens)
                throw std::runtime_error("incompatible chunk policy reused prefix");
            options.full_attention_cache = true;
            const auto recropped = ops::require(generator.generate(prompt, options));
            if (recropped.stats.reused_prompt_tokens)
                throw std::runtime_error("incompatible attention policy reused prefix");
            std::cout << "{\"comparison\":\"prefix_regression\",\"backend\":\"" << argv[2]
                      << "\",\"reused_tokens\":" << hit.stats.reused_prompt_tokens
                      << ",\"cached_bytes\":" << hit.stats.prefix_cache_bytes << ",\"exact_outputs\":true}\n";
            return 0;
        }
        const auto config = ops::require(model::load_config(directory / "model.yaml"));
        const auto tokenizer = ops::require(text::Tokenizer::load(config["tokenizer_file"].as<std::string>()));
        const auto weights = ops::require(model::Weights::load(config["weights_file"].as<std::string>()));
        const auto embedding = ops::require(weights.tensor(config["model"]["quantization_config"]
                                                               ? "model.language_model.norm.weight"
                                                               : "model.language_model.embed_tokens.weight"));
        if (argc >= 6 && std::string_view(argv[5]) == "cache-shape") {
            const auto count = argc == 7 ? chunk : 128;
            if (count < 2 || count > 1024) throw std::runtime_error("cache shape probe requires 2..1024 tokens");
            const ModuleScope construction(embedding.dtype(), false, device);
            auto model = ops::require(model::Gemma4Impl::create(config["model"]));
            ops::require(model->set_checkpoint(weights, bits, group));
            const auto seed =
                ops::require(tokenizer.encode("<bos>The teacher asked the students to explain why sunlight contains "
                                              "many colors and how blue light scatters in the atmosphere."));
            std::vector<std::int32_t> sequence(count);
            for (std::size_t index = 0; index < sequence.size(); ++index) sequence[index] = seed[index % seed.size()];
            auto full = ops::require(model->create_state(count));
            auto partial = ops::require(model->create_state(count));
            ops::require(model->prefill(sequence, full));
            ops::require(model->prefill(std::span(sequence).first(count - 1), partial));
            std::cout << "{\"comparison\":\"cache_shape\",\"backend\":\"" << argv[2] << "\",\"tokens\":" << count
                      << ",\"producers\":[";
            for (std::size_t producer = 0; producer < full.layers.size(); ++producer) {
                std::size_t mismatches = 0, values = 0;
                double maximum_error = 0;
                for (const auto& [left, right] :
                     std::array{std::pair{&full.layers[producer].key, &partial.layers[producer].key},
                                std::pair{&full.layers[producer].value, &partial.layers[producer].value}}) {
                    const auto reference = ops::require(left->data<float>()).first((count - 1) * left->size(-1));
                    const auto actual = ops::require(right->data<float>()).first(reference.size());
                    for (std::size_t index = 0; index < reference.size(); ++index) {
                        if (!std::isfinite(reference[index]) || !std::isfinite(actual[index]))
                            throw std::runtime_error("nonfinite cache probe");
                        mismatches += reference[index] != actual[index];
                        maximum_error = std::max(maximum_error, std::abs(double(reference[index]) - actual[index]));
                    }
                    values += reference.size();
                }
                if (producer) std::cout << ',';
                std::cout << "{\"producer\":" << producer << ",\"values\":" << values
                          << ",\"mismatches\":" << mismatches << ",\"max_error\":" << maximum_error << '}';
            }
            std::cout << "]}\n";
            return 0;
        }
        if (argc >= 6 && std::string_view(argv[5]) == "batch") {
            const auto batch_size = argc == 7 ? chunk : 4;
            if (batch_size > 16) throw std::runtime_error("batch probe is limited to 16 requests");
            const ModuleScope construction(embedding.dtype(), false, device);
            auto model = ops::require(model::Gemma4Impl::create(config["model"]));
            ops::require(model->set_checkpoint(weights, bits, group));
            const auto sequence =
                ops::require(tokenizer.encode("<bos>The teacher asked the students to compare sunlight with artificial "
                                              "light and explain their observations carefully."));
            std::vector<model::Gemma4State> serial, batched;
            std::vector<model::Gemma4State*> states;
            for (int row = 0; row < batch_size; ++row) {
                auto seed = ops::require(model->create_state(128));
                ops::require(model->prefill(std::span(sequence).first(4 + row % 4), seed));
                serial.push_back(ops::require(model->fork_state(seed, seed.position, 128)));
                batched.push_back(ops::require(model->fork_state(seed, seed.position, 128)));
            }
            for (auto& state : batched) states.push_back(&state);
            std::vector<std::int32_t> ids(batch_size);
            std::vector<double> serial_times, batched_times;
            std::size_t agreements = 0, positions = 0;
            double maximum_error = 0, divergence = 0;
            for (int step = 0; step < 10; ++step) {
                for (int row = 0; row < batch_size; ++row) ids[row] = sequence[(row + step + 8) % sequence.size()];
                std::vector<tensor::Tensor> reference;
                reference.reserve(batch_size);
                tensor::Tensor actual;
                double serial_ms = 0, batched_ms = 0;
                const auto run_serial = [&] {
                    const auto started = std::chrono::steady_clock::now();
                    for (int row = 0; row < batch_size; ++row)
                        reference.push_back(
                            ops::require(model->forward(std::span<const std::int32_t>(&ids[row], 1), serial[row])));
                    serial_ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                };
                const auto run_batched = [&] {
                    const auto started = std::chrono::steady_clock::now();
                    actual = ops::require(model->forward_batch(ids, states));
                    batched_ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                };
                if (step % 2) {
                    run_batched();
                    run_serial();
                } else {
                    run_serial();
                    run_batched();
                }
                if (step < 2) continue;
                serial_times.push_back(serial_ms);
                batched_times.push_back(batched_ms);
                const auto values = ops::require(actual.data<float>());
                for (int row = 0; row < batch_size; ++row) {
                    const auto expected = ops::require(reference[row].data<float>());
                    const auto observed = values.subspan(row * expected.size(), expected.size());
                    const auto best = std::ranges::max_element(expected), found = std::ranges::max_element(observed);
                    agreements += best - expected.begin() == found - observed.begin();
                    ++positions;
                    double expected_sum = 0, observed_sum = 0;
                    for (std::size_t token = 0; token < expected.size(); ++token) {
                        if (!std::isfinite(observed[token])) throw std::runtime_error("nonfinite batch logits");
                        maximum_error = std::max(maximum_error, std::abs(double(expected[token]) - observed[token]));
                        expected_sum += std::exp(double(expected[token] - *best));
                        observed_sum += std::exp(double(observed[token] - *found));
                    }
                    const double expected_z = *best + std::log(expected_sum),
                                 observed_z = *found + std::log(observed_sum);
                    for (std::size_t token = 0; token < expected.size(); ++token)
                        divergence += std::exp(expected[token] - expected_z) *
                                      (expected[token] - expected_z - observed[token] + observed_z);
                }
            }
            std::ranges::sort(serial_times);
            std::ranges::sort(batched_times);
            std::cout << std::setprecision(9) << "{\"comparison\":\"batched_forced_decode\",\"backend\":\"" << argv[2]
                      << "\",\"requests\":" << batch_size << ",\"positions\":" << positions
                      << ",\"top1_agreement\":" << double(agreements) / positions
                      << ",\"max_logit_error\":" << maximum_error << ",\"mean_kl\":" << divergence / positions
                      << ",\"serial_step_ms\":" << serial_times[serial_times.size() / 2]
                      << ",\"batched_step_ms\":" << batched_times[batched_times.size() / 2] << "}\n";
            return 0;
        }
        const std::array texts{
            "The capital of France is Paris. The river Seine flows through the city, which is known for its museums "
            "and architecture.",
            "Water freezes at zero degrees Celsius at standard atmospheric pressure. Heating ice provides energy that "
            "melts the solid into liquid water.",
            "A triangle has three sides. A square has four equal sides and four right angles. The area of a rectangle "
            "is its width multiplied by its height.",
            "def factorial(number):\n    if number <= 1:\n        return 1\n    return number * factorial(number - "
            "1)\n",
            "The teacher asked the students to compare the two results and explain their reasoning. They checked the "
            "measurements before drawing a conclusion.",
            "Translate from French to English: Bonjour, comment allez-vous ? The English translation is: Hello, how "
            "are you?"};
        std::vector<std::vector<std::int32_t>> tokens;
        for (auto text : texts) tokens.push_back(ops::require(tokenizer.encode(std::string("<bos>") + text)));
        const bool reference_probe = argc >= 6 && std::filesystem::path(argv[5]).extension() == ".safetensors";
        const bool last_prefix_probe = std::getenv("KIDI_QUALITY_LAST_PREFIX") != nullptr;
        if (last_prefix_probe && !reference_probe)
            throw std::runtime_error("last-prefix quality requires an independent reference");
        const bool window_probe = argc >= 6 && !reference_probe;
        std::size_t prefill_count = 0;
        if (window_probe) {
            const auto document = YAML::LoadFile(argv[5]);
            auto sequence = document["prompt_tokens"].as<std::vector<std::int32_t>>();
            prefill_count = sequence.size();
            const auto continuation = document["records"][0]["token_ids"].as<std::vector<std::int32_t>>();
            if (!prefill_count || continuation.size() < 2) throw std::runtime_error("empty window comparison sequence");
            sequence.insert(sequence.end(), continuation.begin(), continuation.end());
            tokens = {std::move(sequence)};
        }
        std::vector<std::vector<float>> baseline;
        if (reference_probe) {
            const auto reference = ops::require(model::Weights::load(argv[5]));
            const auto token_tensor = ops::require(reference.tensor("tokens"));
            const auto expected = ops::require(reference.tensor("logits"));
            const auto ids = ops::require(token_tensor.data<std::int32_t>());
            const auto values = ops::require(expected.data<float>());
            if (values.size() != ids.size() * tokenizer.vocabulary_size())
                throw std::runtime_error("reference shape mismatch");
            tokens = {{ids.begin(), ids.end()}};
            tokens[0].push_back(1);
            baseline.emplace_back(values.begin(), values.end());
        }
        double reference_nll = 0, quantized_nll = 0, divergence = 0;
        std::size_t count = 0, agreements = 0;
        for (int pass : {0, 1}) {
            if (reference_probe && pass == 0) continue;
            const ModuleScope construction(embedding.dtype(), false, device);
            auto model = ops::require(model::Gemma4Impl::create(config["model"]));
            ops::require(model->set_checkpoint(weights, window_probe || pass ? bits : 0, group));
            for (std::size_t item = 0; item < tokens.size(); ++item) {
                auto state = ops::require(model->create_state(((tokens[item].size() + 127) / 128) * 128));
                state.crop_local_attention = !window_probe || pass;
                const auto input = std::span(tokens[item]).first(tokens[item].size() - 1);
                for (std::size_t offset = 0; offset < prefill_count; offset += 128)
                    ops::require(model->prefill(
                        input.subspan(offset, std::min<std::size_t>(128, prefill_count - offset)), state));
                std::vector<float> stored;
                for (std::size_t position = prefill_count; position < input.size(); position += chunk) {
                    const auto length = std::min<std::size_t>(chunk, input.size() - position);
                    if (last_prefix_probe)
                        state = ops::require(model->create_state(((tokens[item].size() + 127) / 128) * 128));
                    const auto output = ops::require(model->forward(
                        last_prefix_probe ? input.first(position + length) : input.subspan(position, length), state,
                        !last_prefix_probe));
                    const auto values = ops::require(output.data<float>());
                    stored.insert(stored.end(), values.begin(), values.end());
                }
                const std::span<const float> values(stored);
                if (!pass) {
                    baseline.emplace_back(values.begin(), values.end());
                    continue;
                }
                const auto vocabulary = tokenizer.vocabulary_size();
                for (std::size_t row = 0; row < values.size() / vocabulary; ++row) {
                    const auto reference_row =
                        last_prefix_probe ? std::min<std::size_t>((row + 1) * chunk, input.size()) - 1 : row;
                    const auto reference = std::span(baseline[item]).subspan(reference_row * vocabulary, vocabulary);
                    const auto actual = values.subspan(row * vocabulary, vocabulary);
                    const auto best = std::ranges::max_element(reference);
                    const auto predicted = std::ranges::max_element(actual);
                    agreements += (best - reference.begin()) == (predicted - actual.begin());
                    double reference_sum = 0, actual_sum = 0;
                    for (std::size_t token = 0; token < vocabulary; ++token) {
                        if (!std::isfinite(actual[token])) throw std::runtime_error("non-finite quantized logits");
                        reference_sum += std::exp(double(reference[token] - *best));
                        actual_sum += std::exp(double(actual[token] - *predicted));
                    }
                    const double reference_z = *best + std::log(reference_sum),
                                 actual_z = *predicted + std::log(actual_sum);
                    reference_nll += reference_z - reference[tokens[item][prefill_count + reference_row + 1]];
                    quantized_nll += actual_z - actual[tokens[item][prefill_count + reference_row + 1]];
                    for (std::size_t token = 0; token < vocabulary; ++token) {
                        const auto probability = std::exp(reference[token] - reference_z);
                        divergence += probability * ((reference[token] - reference_z) - (actual[token] - actual_z));
                    }
                    ++count;
                }
            }
        }
        std::cout << std::setprecision(9) << "{\"comparison\":\""
                  << (reference_probe ? "independent_reference"
                      : window_probe  ? "local_window"
                                      : "quantization")
                  << "\",\"prefill_tokens\":" << prefill_count << ",\"backend\":\"" << argv[2] << "\",\"bits\":" << bits
                  << ",\"group_size\":" << group << ",\"evaluation_chunk_size\":" << chunk << ",\"positions\":" << count
                  << ",\"fresh_prefix_last_logits\":" << (last_prefix_probe ? "true" : "false")
                  << ",\"top1_agreement\":" << double(agreements) / count
                  << ",\"baseline_nll\":" << reference_nll / count << ",\"quantized_nll\":" << quantized_nll / count
                  << ",\"nll_delta\":" << (quantized_nll - reference_nll) / count
                  << ",\"mean_kl\":" << divergence / count << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}