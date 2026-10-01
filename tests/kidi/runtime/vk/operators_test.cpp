#include "kidi/ops/context.h"
#include "kidi/checkpoint/config.h"
#include "kidi/inference/generator.h"
#include "kidi/model/gemma4.h"
#include "kidi/tensor/backend.h"
#include "kidi/text/tokenizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using kidi::ops::require;
using kidi::tensor::Device;
using kidi::tensor::DType;
using kidi::tensor::Tensor;

auto available() -> bool {
    const auto backend = kidi::tensor::BackendRegistry::instance().backend(Device::vulkan());
    return backend && (*backend)->is_available(Device::vulkan()) && (*backend)->supports_execution();
}

auto tensor(std::vector<std::int64_t> shape, std::span<const float> values, Device device = Device::cpu()) -> Tensor {
    return require(Tensor::from_host(std::move(shape), values, device));
}

auto integers(std::vector<std::int64_t> shape, std::span<const std::int32_t> values,
              Device device = Device::cpu()) -> Tensor {
    return require(Tensor::from_host(std::move(shape), values, device));
}

auto bytes(std::vector<std::int64_t> shape, std::span<const std::uint8_t> values,
           Device device = Device::cpu()) -> Tensor {
    return require(Tensor::from_host(std::move(shape), values, device));
}

auto signed_bytes(std::vector<std::int64_t> shape, std::span<const std::int8_t> values,
                  Device device = Device::cpu()) -> Tensor {
    return require(Tensor::from_host(std::move(shape), values, device));
}

auto to_vulkan(const Tensor& input) -> Tensor { return require(input.to(Device::vulkan())); }

auto values(const Tensor& input) -> std::vector<float> {
    const auto host = require(input.to(Device::cpu()));
    const auto span = require(host.data<float>());
    return {span.begin(), span.end()};
}

auto int_values(const Tensor& input) -> std::vector<std::int32_t> {
    const auto host = require(input.to(Device::cpu()));
    const auto span = require(host.data<std::int32_t>());
    return {span.begin(), span.end()};
}

auto byte_values(const Tensor& input) -> std::vector<std::int8_t> {
    const auto host = require(input.to(Device::cpu()));
    const auto span = require(host.data<std::int8_t>());
    return {span.begin(), span.end()};
}

auto assert_close(std::string_view name, const Tensor& expected, const Tensor& actual,
                  float tolerance = 2e-4F) -> void {
    const auto left = values(expected), right = values(actual);
    if (left.size() != right.size()) throw std::runtime_error(std::string(name) + " size mismatch");
    for (std::size_t index = 0; index < left.size(); ++index)
        if (!std::isfinite(right[index]) || std::abs(left[index] - right[index]) > tolerance)
            throw std::runtime_error(std::string(name) + " mismatch at " + std::to_string(index) + ": " +
                                     std::to_string(left[index]) + " != " + std::to_string(right[index]));
}

auto assert_equal(std::string_view name, const Tensor& expected, const Tensor& actual) -> void {
    const auto left = int_values(expected), right = int_values(actual);
    if (left != right) throw std::runtime_error(std::string(name) + " integer mismatch");
}

auto make_sequence(std::size_t count, float scale = 0.03125F) -> std::vector<float> {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) result[index] = (static_cast<int>(index % 23) - 11) * scale;
    return result;
}

auto packed(std::span<const std::int8_t> unpacked, std::size_t rows, std::size_t columns,
            int bits) -> std::vector<std::uint8_t> {
    const auto per_byte = static_cast<std::size_t>(8 / bits);
    const auto mask = static_cast<std::uint8_t>((1 << bits) - 1);
    std::vector<std::uint8_t> result(rows * columns / per_byte);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column) {
            const auto raw = static_cast<std::uint8_t>(unpacked[row * columns + column]) & mask;
            result[row * (columns / per_byte) + column / per_byte] |=
                static_cast<std::uint8_t>(raw << ((column % per_byte) * bits));
        }
    return result;
}

auto check_pointwise(std::size_t rows) -> void {
    kidi::ops::Context cpu(Device::cpu()), vk(Device::vulkan());
    const auto left_values = make_sequence(rows * 8), right_values = make_sequence(8, 0.0625F);
    const auto left = tensor({1, static_cast<std::int64_t>(rows), 8}, left_values);
    const auto right = tensor({8}, right_values);
    assert_close("add", cpu.add(left, right), vk.add(to_vulkan(left), to_vulkan(right)));
    assert_close("multiply", cpu.multiply(left, right), vk.multiply(to_vulkan(left), to_vulkan(right)));
    assert_close("tanh", cpu.tanh(left), vk.tanh(to_vulkan(left)));
    assert_close("static_round", cpu.static_round(left, 0.125F), vk.static_round(to_vulkan(left), 0.125F));
    const auto rounded = vk.cast(to_vulkan(left), DType::I8, 0.125F);
    const auto expected = cpu.cast(left, DType::I8, 0.125F);
    if (byte_values(expected) != byte_values(rounded)) throw std::runtime_error("calibrated cast mismatch");
}

auto check_linear(std::size_t rows) -> void {
    kidi::ops::Context cpu(Device::cpu()), vk(Device::vulkan());
    const auto input_values = make_sequence(rows * 16);
    const auto weight_values = make_sequence(12 * 16, 0.015625F);
    const auto bias_values = make_sequence(12, 0.125F);
    const auto input = tensor({1, static_cast<std::int64_t>(rows), 16}, input_values);
    const auto weight = tensor({12, 16}, weight_values);
    const auto bias = tensor({12}, bias_values);
    assert_close("linear", cpu.linear(input, weight, bias, true),
                 vk.linear(to_vulkan(input), to_vulkan(weight), to_vulkan(bias), true));

    std::vector<std::int8_t> unpacked(16 * 16);
    for (std::size_t index = 0; index < unpacked.size(); ++index)
        unpacked[index] = static_cast<std::int8_t>(static_cast<int>(index % 15) - 7);
    const auto packed_values = packed(unpacked, 16, 16, 4);
    const std::vector<float> scales(16, 0.03125F);
    const auto packed_weight = bytes({16, 8}, packed_values);
    const auto packed_scales = tensor({16, 1}, scales);
    assert_close(
        "packed linear", cpu.packed_linear(input, packed_weight, packed_scales, 4, 16, 0.125F, 0.0625F),
        vk.packed_linear(to_vulkan(input), to_vulkan(packed_weight), to_vulkan(packed_scales), 4, 16, 0.125F, 0.0625F),
        1e-5F);
}

auto check_packed_projection(std::size_t rows, std::size_t width, std::size_t columns, int bits) -> void {
    kidi::ops::Context cpu(Device::cpu()), vk(Device::vulkan());
    const auto previous = std::getenv("KIDI_VULKAN_PACKED");
    const std::string saved = previous ? previous : "";
    setenv("KIDI_VULKAN_PACKED", "1", 1);
    const auto input_values = make_sequence(rows * width, 0.125F);
    std::vector<std::int8_t> unpacked(columns * width);
    const auto low = -(1 << (bits - 1));
    const auto high = (1 << (bits - 1)) - 1;
    for (std::size_t index = 0; index < unpacked.size(); ++index)
        unpacked[index] = static_cast<std::int8_t>(low + static_cast<int>(index % (high - low + 1)));
    std::vector<float> scales(columns);
    for (std::size_t column = 0; column < columns; ++column)
        scales[column] = 0.00390625F * static_cast<float>(1 + column % 7);
    const auto input = tensor({1, static_cast<std::int64_t>(rows), static_cast<std::int64_t>(width)}, input_values);
    const auto packed_weight =
        bytes({static_cast<std::int64_t>(columns), static_cast<std::int64_t>(width / (8 / bits))},
              packed(unpacked, columns, width, bits));
    const auto scale = tensor({static_cast<std::int64_t>(columns), 1}, scales);
    assert_close(
        "wide packed linear", cpu.packed_linear(input, packed_weight, scale, bits, width, 0.125F, 0.0625F),
        vk.packed_linear(to_vulkan(input), to_vulkan(packed_weight), to_vulkan(scale), bits, width, 0.125F, 0.0625F),
        1e-5F);
    if (previous)
        setenv("KIDI_VULKAN_PACKED", saved.c_str(), 1);
    else
        unsetenv("KIDI_VULKAN_PACKED");
}

auto check_gated_feed_forward(std::size_t rows) -> void {
    kidi::ops::Context cpu(Device::cpu()), vk(Device::vulkan());
    constexpr std::size_t WIDTH = 256;
    constexpr std::size_t INTERMEDIATE = 256;
    constexpr int BITS = 4;
    const auto input_values = make_sequence(rows * WIDTH, 0.125F);
    std::vector<std::int8_t> gate_up_values(2 * INTERMEDIATE * WIDTH), down_values(WIDTH * INTERMEDIATE);
    for (std::size_t index = 0; index < gate_up_values.size(); ++index)
        gate_up_values[index] = static_cast<std::int8_t>(static_cast<int>(index % 15) - 7);
    for (std::size_t index = 0; index < down_values.size(); ++index)
        down_values[index] = static_cast<std::int8_t>(static_cast<int>((index * 3) % 15) - 7);
    std::vector<float> gate_up_scales(2 * INTERMEDIATE), down_scales(WIDTH);
    for (std::size_t index = 0; index < gate_up_scales.size(); ++index)
        gate_up_scales[index] = 0.00390625F * static_cast<float>(1 + index % 5);
    for (std::size_t index = 0; index < down_scales.size(); ++index)
        down_scales[index] = 0.00390625F * static_cast<float>(1 + index % 7);
    const auto input = tensor({1, static_cast<std::int64_t>(rows), WIDTH}, input_values);
    const auto gate_up =
        bytes({2 * INTERMEDIATE, WIDTH / (8 / BITS)}, packed(gate_up_values, 2 * INTERMEDIATE, WIDTH, BITS));
    const auto down = bytes({WIDTH, INTERMEDIATE / (8 / BITS)}, packed(down_values, WIDTH, INTERMEDIATE, BITS));
    const auto gate_scale = tensor({2 * INTERMEDIATE, 1}, gate_up_scales);
    const auto down_scale = tensor({WIDTH, 1}, down_scales);
    assert_close(
        "gated feed-forward",
        cpu.gated_feed_forward(input, gate_up, gate_scale, down, down_scale, BITS, WIDTH, INTERMEDIATE, 0.125F, 0.0625F,
                               0.03125F, 0.125F),
        vk.gated_feed_forward(to_vulkan(input), to_vulkan(gate_up), to_vulkan(gate_scale), to_vulkan(down),
                              to_vulkan(down_scale), BITS, WIDTH, INTERMEDIATE, 0.125F, 0.0625F, 0.03125F, 0.125F),
        2e-4F);
}

auto check_embedding() -> void {
    kidi::ops::Context cpu(Device::cpu()), vk(Device::vulkan());
    const std::array<std::int32_t, 4> ids{0, 2, 3, 1};
    const auto id_tensor = integers({4}, ids);
    const auto weight_values = make_sequence(4 * 8);
    const auto weight = tensor({4, 8}, weight_values);
    assert_close("embedding", cpu.embedding(id_tensor, weight, {}, 8, 0, 2.F),
                 vk.embedding(to_vulkan(id_tensor), to_vulkan(weight), {}, 8, 0, 2.F));

    std::vector<std::int8_t> unpacked(4 * 8);
    for (std::size_t index = 0; index < unpacked.size(); ++index)
        unpacked[index] = static_cast<std::int8_t>(static_cast<int>(index % 7) - 3);
    const auto packed_values = packed(unpacked, 4, 8, 4);
    const std::vector<float> scales(4, 0.25F);
    const auto packed_weight = bytes({4, 4}, packed_values);
    const auto scale = tensor({4, 1}, scales);
    assert_close("packed embedding", cpu.embedding(id_tensor, packed_weight, scale, 8, 4, 1.F),
                 vk.embedding(to_vulkan(id_tensor), to_vulkan(packed_weight), to_vulkan(scale), 8, 4, 1.F));
}

auto check_norm_rotary(std::size_t rows) -> void {
    kidi::ops::Context cpu(Device::cpu()), vk(Device::vulkan());
    const auto input_values = make_sequence(rows * 8, 0.0625F);
    const auto scale_values = make_sequence(8, 0.03125F);
    const std::vector<float> positive_scale(scale_values.begin(), scale_values.end());
    const auto input = tensor({1, static_cast<std::int64_t>(rows), 8}, input_values);
    const auto scale = tensor({8}, positive_scale);
    const auto residual_values = make_sequence(rows * 8, 0.015625F);
    const auto residual = tensor({1, static_cast<std::int64_t>(rows), 8}, residual_values);
    assert_close("rms norm", cpu.rms_norm(input, scale, 1e-5F), vk.rms_norm(to_vulkan(input), to_vulkan(scale), 1e-5F));
    assert_close("rms residual", cpu.rms_norm_residual(input, scale, residual, 1e-5F),
                 vk.rms_norm_residual(to_vulkan(input), to_vulkan(scale), to_vulkan(residual), 1e-5F));

    const auto rotary_values = make_sequence(rows * 2 * 8);
    const auto rotary_input = tensor({1, static_cast<std::int64_t>(rows), 2, 8}, rotary_values);
    std::vector<float> cosine(rows * 4), sine(rows * 4);
    for (std::size_t index = 0; index < cosine.size(); ++index) {
        cosine[index] = std::cos(static_cast<float>(index) * 0.01F);
        sine[index] = std::sin(static_cast<float>(index) * 0.01F);
    }
    const auto cos_tensor = tensor({1, static_cast<std::int64_t>(rows), 1, 4}, cosine);
    const auto sin_tensor = tensor({1, static_cast<std::int64_t>(rows), 1, 4}, sine);
    assert_close("rotary", cpu.rotary(rotary_input, cos_tensor, sin_tensor),
                 vk.rotary(to_vulkan(rotary_input), to_vulkan(cos_tensor), to_vulkan(sin_tensor)));
    assert_close(
        "rms rotary", cpu.rms_rotary(rotary_input, scale, cos_tensor, sin_tensor, 1e-5F),
        vk.rms_rotary(to_vulkan(rotary_input), to_vulkan(scale), to_vulkan(cos_tensor), to_vulkan(sin_tensor), 1e-5F));
}

auto check_attention(std::size_t rows) -> void {
    kidi::ops::Context cpu(Device::cpu()), vk(Device::vulkan());
    const auto query_values = make_sequence(rows * 4);
    const auto key_values = make_sequence((rows + 3) * 2);
    const auto value_values = make_sequence((rows + 3) * 2, 0.047F);
    const auto query = tensor({1, static_cast<std::int64_t>(rows), 4}, query_values);
    const auto key = tensor({1, static_cast<std::int64_t>(rows + 3), 2}, key_values);
    const auto value = tensor({1, static_cast<std::int64_t>(rows + 3), 2}, value_values);
    std::vector<float> mask(rows * (rows + 3), 0.F);
    const auto mask_tensor = tensor({1, 1, static_cast<std::int64_t>(rows), static_cast<std::int64_t>(rows + 3)}, mask);
    assert_close("attention", cpu.grouped_query_attention(query, key, value, 2, 1, mask_tensor, 0.5F, 0, {}, {}),
                 vk.grouped_query_attention(to_vulkan(query), to_vulkan(key), to_vulkan(value), 2, 1,
                                            to_vulkan(mask_tensor), 0.5F, 0, {}, {}),
                 1e-4F);

    std::vector<std::int8_t> key_bytes((rows + 3) * 2), value_bytes((rows + 3) * 2);
    for (std::size_t index = 0; index < key_bytes.size(); ++index) {
        key_bytes[index] = static_cast<std::int8_t>(static_cast<int>(index % 11) - 5);
        value_bytes[index] = static_cast<std::int8_t>(static_cast<int>(index % 13) - 6);
    }
    const auto qk = signed_bytes({1, static_cast<std::int64_t>(rows + 3), 2}, key_bytes);
    const auto qv = signed_bytes({1, static_cast<std::int64_t>(rows + 3), 2}, value_bytes);
    const kidi::ops::BlockwiseQuantization quantization{{0.125F}, {0}, 2};
    assert_close("int8 attention",
                 cpu.grouped_query_attention(query, qk, qv, 2, 1, mask_tensor, 0.5F, 0, quantization, quantization),
                 vk.grouped_query_attention(to_vulkan(query), to_vulkan(qk), to_vulkan(qv), 2, 1,
                                            to_vulkan(mask_tensor), 0.5F, 0, quantization, quantization),
                 1e-4F);
}

auto check_indexing() -> void {
    kidi::ops::Context cpu(Device::cpu()), vk(Device::vulkan());
    const auto input_values = make_sequence(6);
    const auto input = tensor({2, 3}, input_values);
    const std::array<std::int32_t, 2> index_values{2, 0};
    const auto indices = integers({2}, index_values);
    assert_close("gather", cpu.gather(input, indices, 1), vk.gather(to_vulkan(input), to_vulkan(indices), 1));
    assert_close("slice", cpu.slice(input, 1, 1, 2), vk.slice(to_vulkan(input), 1, 1, 2));
    const std::array cpu_concat{input, input};
    const std::array vk_concat{to_vulkan(input), to_vulkan(input)};
    assert_close("concat", cpu.concat(cpu_concat, 0), vk.concat(vk_concat, 0));

    const auto destination_values = make_sequence(10);
    const auto update_values = make_sequence(4, 0.5F);
    const auto destination = tensor({1, 5, 2}, destination_values);
    const auto updates = tensor({1, 2, 2}, update_values);
    const std::array<std::int32_t, 2> scatter_indices{1, 3};
    const auto scatter_index = integers({2}, scatter_indices);
    assert_close("scatter", cpu.scatter(destination, updates, scatter_index),
                 vk.scatter(to_vulkan(destination), to_vulkan(updates), to_vulkan(scatter_index)));

    auto copied = to_vulkan(require(Tensor::zeros({1, 5, 2}, DType::F32)));
    vk.copy_slice_(copied, to_vulkan(updates), 1, 2);
    vk.synchronize();
    auto cpu_copied = require(Tensor::zeros({1, 5, 2}, DType::F32));
    cpu.copy_slice_(cpu_copied, updates, 1, 2);
    assert_close("copy slice", cpu_copied, copied);
}

auto check_selection() -> void {
    kidi::ops::Context cpu(Device::cpu()), vk(Device::vulkan());
    const std::array<float, 8> logits{1.F, 3.F, 3.F, 2.F, 0.F, INFINITY, 1.F, 2.F};
    const auto input = tensor({2, 4}, logits);
    assert_equal("greedy", cpu.greedy_token(input), vk.greedy_token(to_vulkan(input)));
}

auto argmax_last_row(std::span<const float> logits, std::size_t vocabulary) -> std::int32_t {
    if (logits.size() < vocabulary || vocabulary == 0) throw std::runtime_error("empty logits");
    auto row = logits.last(vocabulary);
    if (std::ranges::any_of(row, [](float value) { return std::isnan(value) || value == INFINITY; })) return -1;
    return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

auto last_row_value(std::span<const float> logits, std::size_t vocabulary, std::int32_t token) -> float {
    if (token < 0 || static_cast<std::size_t>(token) >= vocabulary) return -INFINITY;
    return logits.last(vocabulary)[static_cast<std::size_t>(token)];
}

auto last_row_margin(std::span<const float> logits, std::size_t vocabulary, std::int32_t token) -> float {
    auto row = logits.last(vocabulary);
    float second = -INFINITY;
    for (std::size_t index = 0; index < row.size(); ++index)
        if (static_cast<std::int32_t>(index) != token) second = std::max(second, row[index]);
    return last_row_value(logits, vocabulary, token) - second;
}

auto compare_last_row(std::span<const float> expected, std::span<const float> actual,
                      std::size_t vocabulary) -> std::pair<double, double> {
    if (expected.size() < vocabulary || actual.size() < vocabulary) throw std::runtime_error("invalid logits");
    expected = expected.last(vocabulary);
    actual = actual.last(vocabulary);
    double square = 0.0, maximum = 0.0;
    for (std::size_t index = 0; index < vocabulary; ++index) {
        const auto delta = static_cast<double>(actual[index]) - static_cast<double>(expected[index]);
        square += delta * delta;
        maximum = std::max(maximum, std::abs(delta));
    }
    return {std::sqrt(square / static_cast<double>(vocabulary)), maximum};
}

auto load_gemma(const std::filesystem::path& directory, const YAML::Node& config,
                const kidi::checkpoint::Weights& weights, Device device) -> kidi::model::Gemma4 {
    const auto parameter =
        require(weights.tensor(config["quantization_config"] ? "model.language_model.norm.weight"
                                                             : "model.language_model.embed_tokens.weight"));
    const kidi::ModuleScope construction(parameter.dtype(), false, device);
    auto model = require(kidi::model::Gemma4Impl::create(config));
    const auto* packed = std::getenv("KIDI_DIAG_PACKED_PREFILL");
    require(model->set_checkpoint(weights, 0, 128, !packed || std::string_view(packed) != "0"));
    (void)directory;
    return model;
}

auto run_gemma_teacher_forced(const std::filesystem::path& directory, int copies) -> int {
    constexpr std::size_t CONTEXT = 9216;
    constexpr std::size_t STEPS = 64;
    const auto chunk = std::getenv("KIDI_DIAG_CHUNK") ? std::strtoull(std::getenv("KIDI_DIAG_CHUNK"), nullptr, 10) : 32;
    const auto full_config = require(kidi::checkpoint::load_config(directory / "model.yaml"));
    const auto config = full_config["model"];
    auto tokenizer = require(kidi::text::Tokenizer::load(full_config["tokenizer_file"].as<std::string>()));
    auto weights = require(kidi::checkpoint::Weights::load(full_config["weights_file"].as<std::string>()));
    auto cpu = load_gemma(directory, config, weights, Device::cpu());
    auto vk = load_gemma(directory, config, weights, Device::vulkan());

    std::string prompt;
    for (int copy = 0; copy < copies; ++copy)
        prompt +=
            "Binary search repeatedly halves a sorted search range. It is useful for lookup and boundary finding. ";
    prompt += "Explain the main idea and give practical examples.";
    const std::array messages{kidi::text::ChatMessage{"user", prompt}};
    const auto serialized = require(tokenizer.format_chat(messages));
    auto prompt_tokens = require(tokenizer.encode(serialized));
    auto cpu_state = require(cpu->create_state(CONTEXT));
    auto vk_state = require(vk->create_state(CONTEXT));
    std::size_t offset = 0;
    while (prompt_tokens.size() - offset > chunk) {
        require(cpu->prefill(std::span(prompt_tokens).subspan(offset, chunk), cpu_state));
        require(vk->prefill(std::span(prompt_tokens).subspan(offset, chunk), vk_state));
        offset += chunk;
    }
    const auto vocabulary = tokenizer.vocabulary_size();
    std::vector<std::int32_t> generated;
    generated.reserve(STEPS);
    std::int32_t forced = 0;
    bool first = true;
    std::size_t mismatches = 0;
    for (std::size_t step = 0; step < STEPS; ++step) {
        std::span<const std::int32_t> input;
        std::array<std::int32_t, 1> token{forced};
        if (first) {
            input = std::span(prompt_tokens).subspan(offset);
            first = false;
        } else {
            input = token;
        }
        auto select_state = require(vk->fork_state(vk_state, vk_state.position, CONTEXT));
        const auto vk_selected = require(vk->forward_token(input, select_state));
        const auto cpu_logits = require(cpu->forward(input, cpu_state));
        const auto vk_logits = require(vk->forward(input, vk_state));
        const auto cpu_values = require(cpu_logits.data<float>());
        const auto vk_values = values(vk_logits);
        const auto cpu_top = argmax_last_row(cpu_values, vocabulary);
        const auto vk_top = argmax_last_row(vk_values, vocabulary);
        const auto [rmse, max_abs] = compare_last_row(cpu_values, vk_values, vocabulary);
        if (cpu_top != vk_top || cpu_top != vk_selected) ++mismatches;
        std::cout << "step=" << step << " input=" << input.size() << " cpu_top=" << cpu_top << " vk_top=" << vk_top
                  << " vk_selected=" << vk_selected << " rmse=" << rmse << " max_abs=" << max_abs
                  << " cpu_margin=" << last_row_margin(cpu_values, vocabulary, cpu_top)
                  << " vk_margin=" << last_row_margin(vk_values, vocabulary, vk_top) << " cpu_top_vk_delta="
                  << (last_row_value(vk_values, vocabulary, vk_top) - last_row_value(vk_values, vocabulary, cpu_top))
                  << '\n';
        generated.push_back(cpu_top);
        forced = cpu_top;
    }
    std::cout << "teacher_forced_summary copies=" << copies << " prompt_tokens=" << prompt_tokens.size()
              << " mismatches=" << mismatches << " generated=";
    for (const auto token : generated) std::cout << (token == generated.front() ? "" : ",") << token;
    std::cout << '\n';
    return mismatches == 0 ? 0 : 2;
}

} // namespace

auto main(int argc, char** argv) -> int {
    try {
        if (argc >= 3 && std::string_view(argv[1]) == "gemma-diagnostic") {
            const int copies = argc >= 4 ? std::atoi(argv[3]) : 1;
            return run_gemma_teacher_forced(argv[2], copies);
        }
        if (!available()) {
            std::cout << "Vulkan backend unavailable; skipping\n";
            return 0;
        }
        for (const auto rows : {std::size_t{1}, std::size_t{32}, std::size_t{128}}) {
            check_pointwise(rows);
            check_linear(rows);
        }
        check_packed_projection(1, 1536, 512, 4);
        check_packed_projection(1, 6144, 512, 2);
        check_gated_feed_forward(1);
        check_gated_feed_forward(32);
        check_embedding();
        for (const auto rows : {std::size_t{1}, std::size_t{32}}) {
            check_norm_rotary(rows);
            check_attention(rows);
        }
        check_indexing();
        check_selection();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
