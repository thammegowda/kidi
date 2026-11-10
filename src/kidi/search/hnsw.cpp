#include "kidi/search/hnsw.h"
#include "kidi/search/search.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <tuple>

namespace kidi::search {
namespace {
constexpr std::size_t HEADER_BYTES = 128, HEADER_CRC_OFFSET = 52;
constexpr std::uint32_t MAX_NODES = 1U << 24;
constexpr std::uint32_t FORMAT_VERSION = 2;

struct Layer {
    std::vector<std::uint32_t> nodes, offsets, edges;
};

struct Graph {
    std::uint32_t width = 0, m = 0, entry = 0, weight_type = 0, weights_crc32 = 0;
    std::size_t candidates = 8, ef_search = 800;
    std::uint64_t disk_bytes = 0;
    std::vector<std::uint32_t> labels, base_offsets, base_edges;
    std::vector<std::uint8_t> levels;
    std::vector<Layer> upper;

    auto neighbors(std::uint32_t node, std::size_t level) const -> std::span<const std::uint32_t>;
    auto runtime_bytes() const noexcept -> std::size_t;
};

auto validate_graph(const Graph& graph) -> void {
    const auto nodes = graph.labels.size();
    invalid(!nodes || nodes > MAX_NODES || !graph.width || graph.m < 2 || graph.m > 65535 || graph.entry >= nodes ||
                graph.levels.size() != nodes || graph.base_offsets.size() != nodes + 1 || graph.base_offsets.front() ||
                graph.base_offsets.back() != graph.base_edges.size() || !graph.candidates || graph.candidates > nodes ||
                graph.ef_search < graph.candidates || graph.ef_search > nodes ||
                graph.base_edges.size() > std::numeric_limits<std::uint32_t>::max() ||
                (graph.weight_type != 1 && graph.weight_type != 2),
            "invalid HNSW dimensions, search defaults, or weight type");
    std::vector<bool> labels(nodes);
    for (auto label : graph.labels) {
        invalid(label >= nodes || labels[label], "HNSW labels are not a full permutation");
        labels[label] = true;
    }
    for (std::size_t node = 0; node < nodes; ++node) {
        const auto start = graph.base_offsets[node], end = graph.base_offsets[node + 1];
        invalid(start > end || end > graph.base_edges.size() || end - start > 2 * graph.m,
                "invalid HNSW base offsets or degree");
    }
    for (auto edge : graph.base_edges) invalid(edge >= nodes, "dangling HNSW base edge");
    std::vector<std::uint8_t> levels(nodes);
    for (std::size_t level = 1; level <= graph.upper.size(); ++level) {
        const auto& layer = graph.upper[level - 1];
        invalid(layer.nodes.empty() || layer.offsets.size() != layer.nodes.size() + 1 || layer.offsets.front() ||
                    layer.offsets.back() != layer.edges.size() ||
                    layer.edges.size() > std::numeric_limits<std::uint32_t>::max(),
                "invalid HNSW upper-layer offsets");
        for (std::size_t index = 0; index < layer.nodes.size(); ++index) {
            const auto node = layer.nodes[index];
            invalid(node >= nodes || (index && node <= layer.nodes[index - 1]) || levels[node] != level - 1 ||
                        layer.offsets[index] > layer.offsets[index + 1] ||
                        layer.offsets[index + 1] > layer.edges.size() ||
                        layer.offsets[index + 1] - layer.offsets[index] > graph.m,
                    "invalid HNSW upper-layer membership, ordering, or degree");
            levels[node] = static_cast<std::uint8_t>(level);
        }
        for (auto edge : layer.edges) invalid(edge >= nodes || levels[edge] != level, "dangling HNSW upper-layer edge");
    }
    invalid(levels != graph.levels || graph.levels[graph.entry] != graph.upper.size(),
            "HNSW levels or entry point are inconsistent");
}

auto decode(std::span<const std::byte> bytes) -> Graph {
    invalid(bytes.size() < HEADER_BYTES, "truncated HNSW header");
    invalid(std::memcmp(bytes.data(), "KIDIHNSW", 8) != 0, "invalid HNSW magic");
    std::array<std::byte, HEADER_BYTES> header;
    std::ranges::copy(bytes.first(HEADER_BYTES), header.begin());
    std::fill_n(header.begin() + HEADER_CRC_OFFSET, 4, std::byte{0});
    Reader reader(bytes);
    reader.take(8);
    const auto version = reader.integer(4), header_bytes = reader.integer(4);
    const auto nodes = reader.integer(4);
    Graph graph;
    graph.width = static_cast<std::uint32_t>(reader.integer(4));
    reader.integer(4);
    graph.m = static_cast<std::uint32_t>(reader.integer(4));
    const auto maximum_level = reader.integer(4);
    graph.entry = static_cast<std::uint32_t>(reader.integer(4));
    graph.weight_type = static_cast<std::uint32_t>(reader.integer(4));
    graph.weights_crc32 = static_cast<std::uint32_t>(reader.integer(4));
    const auto payload_crc32 = reader.integer(4), header_crc32 = reader.integer(4);
    const auto map_offset = reader.integer(8), base_offset = reader.integer(8), upper_offset = reader.integer(8);
    graph.disk_bytes = reader.integer(8);
    const auto metadata = reader.take(32);
    reader.integer(8);
    invalid(version != FORMAT_VERSION || header_bytes != HEADER_BYTES || hnsw_crc32(header) != header_crc32,
            "unsupported HNSW version or invalid header checksum");
    invalid(!nodes || nodes > MAX_NODES || !graph.width || graph.m < 2 || graph.m > 65535 || maximum_level > 255 ||
                graph.entry >= nodes || map_offset != HEADER_BYTES || base_offset != map_offset + 4 * nodes ||
                upper_offset < base_offset + 4 * (nodes + 1) || upper_offset > bytes.size() ||
                graph.disk_bytes != bytes.size(),
            "invalid HNSW dimensions, offsets, or file size");
    invalid(hnsw_crc32(bytes.subspan(HEADER_BYTES)) != payload_crc32, "HNSW payload checksum mismatch");
    Reader settings(metadata);
    graph.candidates = settings.integer(4);
    graph.ef_search = settings.integer(4);
    graph.labels.resize(nodes);
    graph.levels.resize(nodes);
    for (auto& label : graph.labels) label = static_cast<std::uint32_t>(reader.integer(4));
    invalid(reader.position() != base_offset, "HNSW label section length mismatch");
    invalid(upper_offset < base_offset + 4 * (nodes + 1) || (upper_offset - base_offset - 4 * (nodes + 1)) % 3,
            "invalid sparse HNSW base section");
    graph.base_offsets.resize(nodes + 1);
    for (auto& offset : graph.base_offsets) offset = static_cast<std::uint32_t>(reader.integer(4));
    graph.base_edges.resize((upper_offset - reader.position()) / 3);
    for (auto& edge : graph.base_edges) edge = static_cast<std::uint32_t>(reader.integer(3));
    invalid(reader.position() != upper_offset, "HNSW base section length mismatch");
    graph.upper.reserve(maximum_level);
    for (std::size_t level = 1; level <= maximum_level; ++level) {
        const auto count = reader.integer(4);
        invalid(!count || count > nodes || count > reader.remaining() / 5,
                "invalid HNSW upper-layer active-node count");
        Layer layer;
        layer.nodes.reserve(count);
        layer.offsets.reserve(count + 1);
        layer.offsets.push_back(0);
        for (std::size_t index = 0; index < count; ++index) {
            const auto node = static_cast<std::uint32_t>(reader.integer(3));
            const auto length = reader.integer(2);
            invalid(node >= nodes || (!layer.nodes.empty() && node <= layer.nodes.back()) ||
                        graph.levels[node] != level - 1 || length > graph.m || length > reader.remaining() / 3 ||
                        length > std::numeric_limits<std::uint32_t>::max() - layer.edges.size(),
                    "invalid HNSW upper-layer membership or list");
            layer.nodes.push_back(node);
            graph.levels[node] = static_cast<std::uint8_t>(level);
            for (std::size_t neighbor = 0; neighbor < length; ++neighbor)
                layer.edges.push_back(static_cast<std::uint32_t>(reader.integer(3)));
            layer.offsets.push_back(static_cast<std::uint32_t>(layer.edges.size()));
        }
        graph.upper.push_back(std::move(layer));
    }
    invalid(reader.remaining(), "unexpected HNSW trailing bytes");
    validate_graph(graph);
    return graph;
}

auto encode(const Graph& graph) -> std::vector<std::byte> {
    validate_graph(graph);
    std::vector<std::byte> payload;
    payload.reserve(graph.labels.size() * 4 + graph.base_offsets.size() * 4 + graph.base_edges.size() * 3);
    for (auto label : graph.labels) append(payload, label, 4);
    const auto base_offset = HEADER_BYTES + payload.size();
    for (auto offset : graph.base_offsets) append(payload, offset, 4);
    for (auto edge : graph.base_edges) append(payload, edge, 3);
    const auto upper_offset = HEADER_BYTES + payload.size();
    for (const auto& layer : graph.upper) {
        append(payload, layer.nodes.size(), 4);
        for (std::size_t index = 0; index < layer.nodes.size(); ++index) {
            append(payload, layer.nodes[index], 3);
            const auto count = layer.offsets[index + 1] - layer.offsets[index];
            append(payload, count, 2);
            for (auto edge : std::span(layer.edges).subspan(layer.offsets[index], count)) append(payload, edge, 3);
        }
    }
    std::vector<std::byte> result(HEADER_BYTES);
    std::memcpy(result.data(), "KIDIHNSW", 8);
    write(result, 8, FORMAT_VERSION, 4);
    write(result, 12, HEADER_BYTES, 4);
    write(result, 16, graph.labels.size(), 4);
    write(result, 20, graph.width, 4);
    const auto maximum_degree = [&] {
        std::uint32_t value = 0;
        for (std::size_t node = 0; node < graph.labels.size(); ++node)
            value = std::max(value, graph.base_offsets[node + 1] - graph.base_offsets[node]);
        return value;
    }();
    write(result, 24, maximum_degree, 4);
    write(result, 28, graph.m, 4);
    write(result, 32, graph.upper.size(), 4);
    write(result, 36, graph.entry, 4);
    write(result, 40, graph.weight_type, 4);
    write(result, 44, graph.weights_crc32, 4);
    write(result, 48, hnsw_crc32(payload), 4);
    write(result, 56, HEADER_BYTES, 8);
    write(result, 64, base_offset, 8);
    write(result, 72, upper_offset, 8);
    write(result, 80, HEADER_BYTES + payload.size(), 8);
    write(result, 88, graph.candidates, 4);
    write(result, 92, graph.ef_search, 4);
    write(result, HEADER_CRC_OFFSET, hnsw_crc32(result), 4);
    result.insert(result.end(), payload.begin(), payload.end());
    return result;
}

struct BuildNode {
    std::vector<std::vector<std::uint32_t>> links;
};

auto Graph::neighbors(std::uint32_t node, std::size_t level) const -> std::span<const std::uint32_t> {
    if (node >= labels.size() || level > upper.size())
        throw Failure({ErrorCode::INVALID_ARGUMENT, "HNSW node or level is out of bounds"});
    if (!level) return std::span(base_edges).subspan(base_offsets[node], base_offsets[node + 1] - base_offsets[node]);
    const auto& layer = upper[level - 1];
    const auto found = std::ranges::lower_bound(layer.nodes, node);
    if (found == layer.nodes.end() || *found != node)
        throw Failure({ErrorCode::INVALID_MANIFEST, "HNSW node is absent from its upper level"});
    const auto index = static_cast<std::size_t>(found - layer.nodes.begin());
    return std::span(layer.edges).subspan(layer.offsets[index], layer.offsets[index + 1] - layer.offsets[index]);
}

auto Graph::runtime_bytes() const noexcept -> std::size_t {
    auto bytes = (labels.size() + base_offsets.size() + base_edges.size()) * sizeof(std::uint32_t) + levels.size();
    for (const auto& layer : upper)
        bytes += (layer.nodes.size() + layer.offsets.size() + layer.edges.size()) * sizeof(std::uint32_t);
    return bytes;
}

} // namespace

struct HnswIndex::State {
    Graph graph;
    tensor::Tensor vectors;
    std::span<const std::byte> bytes;
    tensor::Tensor scales, zeros;
    std::span<const std::byte> scale_bytes, zero_bytes;
    std::int32_t bits = 0, group_size = 0;
    float input_scale = 0.F;
    bool scale_f16 = false, zero_f16 = false, has_zero = false;
    std::size_t row_bytes = 0;
    Dot dot;
    RowDot row_dot;

    State(Graph topology, tensor::Tensor table)
        : graph(std::move(topology)),
          vectors(std::move(table)),
          bytes(require(std::as_const(vectors).host_bytes())),
          row_bytes(graph.width * tensor::element_size(vectors.dtype())) {
        std::tie(dot, row_dot) = dot_kernels(vectors.dtype());
    }

    State(Graph topology, tensor::Tensor weight_table, tensor::Tensor scale_table, tensor::Tensor zero_table,
          std::int32_t weight_bits, std::int32_t group, float act_scale)
        : graph(std::move(topology)),
          vectors(std::move(weight_table)),
          bytes(require(std::as_const(vectors).host_bytes())),
          scales(std::move(scale_table)),
          zeros(std::move(zero_table)),
          scale_bytes(require(std::as_const(scales).host_bytes())),
          zero_bytes(zeros.defined() ? require(std::as_const(zeros).host_bytes()) : std::span<const std::byte>{}),
          bits(weight_bits),
          group_size(group),
          input_scale(act_scale),
          scale_f16(scales.dtype() == tensor::DType::F16),
          zero_f16(zeros.defined() && zeros.dtype() == tensor::DType::F16),
          has_zero(zeros.defined()),
          row_bytes(graph.width / (8 / static_cast<std::size_t>(weight_bits))),
          dot(nullptr),
          row_dot(nullptr) {}
};

HnswIndex::HnswIndex(std::unique_ptr<State> state) : state_(std::move(state)) {}
HnswIndex::HnswIndex(HnswIndex&&) noexcept = default;
auto HnswIndex::operator=(HnswIndex&&) noexcept -> HnswIndex& = default;
HnswIndex::~HnswIndex() = default;
auto HnswIndex::size() const noexcept -> std::size_t { return state_->graph.labels.size(); }
auto HnswIndex::width() const noexcept -> std::size_t { return state_->graph.width; }
auto HnswIndex::candidates() const noexcept -> std::size_t { return state_->graph.candidates; }
auto HnswIndex::ef_search() const noexcept -> std::size_t { return state_->graph.ef_search; }
auto HnswIndex::runtime_bytes() const noexcept -> std::size_t { return state_->graph.runtime_bytes(); }
auto HnswIndex::vectors() const noexcept -> const tensor::Tensor& { return state_->vectors; }

auto HnswIndex::load(const std::filesystem::path& path, tensor::Tensor vectors) -> Result<HnswIndex> {
    auto bytes = read_file(path);
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    try {
        auto graph = decode(*bytes);
        if (!vectors.defined() || !vectors.is_host_accessible() || !vectors.is_contiguous() ||
            vectors.dimensions() != 2 || vectors.size(0) != graph.labels.size() || vectors.size(1) != graph.width ||
            (vectors.dtype() != tensor::DType::BF16 && vectors.dtype() != tensor::DType::F32) ||
            graph.weight_type != (vectors.dtype() == tensor::DType::BF16 ? 1U : 2U))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "HNSW index does not match the vector table"});
        if (hnsw_crc32(require(std::as_const(vectors).host_bytes())) != graph.weights_crc32)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "HNSW vector-table checksum mismatch"});
        return HnswIndex(std::make_unique<State>(std::move(graph), std::move(vectors)));
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}

auto HnswIndex::load_quantized(const std::filesystem::path& path, tensor::Tensor weights, tensor::Tensor scales,
                               tensor::Tensor zeros, std::int32_t bits, std::int32_t group_size, float input_scale)
    -> Result<HnswIndex> {
    auto bytes = read_file(path);
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    try {
        auto graph = decode(*bytes);
        if ((bits != 2 && bits != 4 && bits != 8) || !std::isfinite(input_scale) || input_scale < 0 ||
            group_size <= 0 || graph.width % group_size != 0 || group_size % 32 != 0 || !weights.defined() ||
            !weights.is_host_accessible() || !weights.is_contiguous() || weights.dtype() != tensor::DType::U8 ||
            weights.dimensions() != 2 || weights.size(0) != graph.labels.size() ||
            weights.size(1) != graph.width / (8 / bits) || !scales.defined() || !scales.is_host_accessible() ||
            !scales.is_contiguous() || scales.dimensions() != 2 || scales.size(0) != graph.labels.size() ||
            scales.size(1) != graph.width / group_size ||
            (scales.dtype() != tensor::DType::F32 && scales.dtype() != tensor::DType::F16) ||
            (zeros.defined() &&
             (bits != 4 || !zeros.is_host_accessible() || !zeros.is_contiguous() || zeros.dimensions() != 2 ||
              zeros.size(0) != scales.size(0) || zeros.size(1) != scales.size(1) ||
              (zeros.dtype() != tensor::DType::F32 && zeros.dtype() != tensor::DType::F16))))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "HNSW index does not match quantized vector table"});
        return HnswIndex(std::make_unique<State>(std::move(graph), std::move(weights), std::move(scales),
                                                 std::move(zeros), bits, group_size, input_scale));
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}

auto HnswIndex::from_hnswlib(const std::filesystem::path& path, tensor::Tensor vectors, std::size_t candidates,
                             std::size_t ef_search) -> Result<HnswIndex> {
    try {
        if (!vectors.defined() || !vectors.is_host_accessible() || !vectors.is_contiguous() ||
            vectors.dimensions() != 2 || !vectors.size(0) || vectors.size(0) > MAX_NODES || !vectors.size(1) ||
            vectors.size(1) > std::numeric_limits<std::uint32_t>::max() ||
            (vectors.dtype() != tensor::DType::F32 && vectors.dtype() != tensor::DType::BF16))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid HNSW import vector table"});
        const auto data = require(read_file(path));
        Reader source(data);
        const auto level0_offset = source.integer(8), capacity = source.integer(8), nodes = source.integer(8);
        const auto record_bytes = source.integer(8), label_offset = source.integer(8),
                   vector_offset = source.integer(8);
        const auto maximum_level = source.integer(4), entry = source.integer(4);
        const auto maximum_m = source.integer(8), base_capacity = source.integer(8), m = source.integer(8);
        const auto multiplier = std::bit_cast<double>(source.integer(8));
        const auto construction_ef = source.integer(8);
        invalid(level0_offset || capacity < nodes || nodes != vectors.size(0) || maximum_level > 255 ||
                    entry >= nodes || m < 2 || m > 65535 || maximum_m != m || base_capacity > 2 * m ||
                    vector_offset != 4 + 4 * base_capacity || label_offset != vector_offset + 4 * vectors.size(1) ||
                    record_bytes != label_offset + 8 || nodes > source.remaining() / record_bytes ||
                    !std::isfinite(multiplier) || multiplier <= 0 || construction_ef < m,
                "unsupported or invalid original-hnswlib layout");
        Graph graph;
        graph.width = static_cast<std::uint32_t>(vectors.size(1));
        graph.m = static_cast<std::uint32_t>(m);
        graph.entry = static_cast<std::uint32_t>(entry);
        graph.weight_type = vectors.dtype() == tensor::DType::BF16 ? 1U : 2U;
        graph.candidates = candidates;
        graph.ef_search = ef_search;
        const auto table = require(std::as_const(vectors).host_bytes());
        graph.weights_crc32 = hnsw_crc32(table);
        graph.labels.resize(nodes);
        graph.levels.resize(nodes);
        graph.base_offsets.reserve(nodes + 1);
        graph.base_offsets.push_back(0);
        const auto row_bytes = vectors.size(1) * tensor::element_size(vectors.dtype());
        const bool bfloat = vectors.dtype() == tensor::DType::BF16;
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto record = source.take(record_bytes);
            Reader labels(record.subspan(label_offset));
            const auto label = labels.integer(8);
            invalid(label >= nodes, "hnswlib label is outside the supplied vector table");
            graph.labels[node] = static_cast<std::uint32_t>(label);
            Reader neighbors(record.first(vector_offset));
            const auto count = neighbors.integer(4);
            invalid(count > base_capacity, "invalid or deleted hnswlib base list");
            for (std::size_t edge = 0; edge < count; ++edge)
                graph.base_edges.push_back(static_cast<std::uint32_t>(neighbors.integer(4)));
            if (graph.base_edges.size() > std::numeric_limits<std::uint32_t>::max())
                throw Failure({ErrorCode::UNSUPPORTED, "imported HNSW graph exceeds uint32 offset capacity"});
            graph.base_offsets.push_back(static_cast<std::uint32_t>(graph.base_edges.size()));
            Reader values(record.subspan(vector_offset, 4 * vectors.size(1)));
            const auto* row = table.data() + label * row_bytes;
            for (std::size_t column = 0; column < vectors.size(1); ++column) {
                const auto actual = static_cast<std::uint32_t>(values.integer(4));
                const auto expected = weight_element(row, column, bfloat);
                invalid(!std::isfinite(expected) || actual != std::bit_cast<std::uint32_t>(expected),
                        "hnswlib vectors differ from the original unnormalized table");
            }
        }
        graph.upper.resize(maximum_level);
        for (auto& layer : graph.upper) layer.offsets.push_back(0);
        const auto upper_record_bytes = 4 + 4 * m;
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto length = source.integer(4);
            invalid(length % upper_record_bytes || length > source.remaining() ||
                        length / upper_record_bytes > maximum_level,
                    "invalid hnswlib upper-list length");
            const auto levels = length / upper_record_bytes;
            graph.levels[node] = static_cast<std::uint8_t>(levels);
            for (std::size_t level = 0; level < levels; ++level) {
                Reader neighbors(source.take(upper_record_bytes));
                const auto count = neighbors.integer(4);
                invalid(count > m, "invalid hnswlib upper-neighbor count");
                auto& layer = graph.upper[level];
                layer.nodes.push_back(static_cast<std::uint32_t>(node));
                for (std::size_t edge = 0; edge < count; ++edge)
                    layer.edges.push_back(static_cast<std::uint32_t>(neighbors.integer(4)));
                if (layer.edges.size() > std::numeric_limits<std::uint32_t>::max())
                    throw Failure({ErrorCode::UNSUPPORTED, "imported HNSW layer exceeds uint32 offset capacity"});
                layer.offsets.push_back(static_cast<std::uint32_t>(layer.edges.size()));
            }
        }
        invalid(source.remaining(), "unexpected hnswlib trailing data");
        validate_graph(graph);
        return HnswIndex(std::make_unique<State>(std::move(graph), std::move(vectors)));
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}

auto HnswIndex::save(const std::filesystem::path& path) const -> Result<void> {
    try {
        const auto bytes = encode(state_->graph);
        return save_atomic_file(path, bytes);
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::system_error& error) {
        return std::unexpected(Error{ErrorCode::IO, "cannot save HNSW index: " + std::string(error.what())});
    }
}

auto HnswIndex::prefetch_node(std::uint32_t node) const noexcept -> void {
    const auto& state = *state_;
    const auto label = static_cast<std::size_t>(state.graph.labels[node]);
    const auto* ptr = state.bytes.data() + label * state.row_bytes;
    const auto groups = state.bits ? static_cast<std::size_t>(state.graph.width / state.group_size) : 0;
    prefetch_row(ptr, state.row_bytes, state.bits, label, groups, state.scale_bytes, state.scale_f16, state.zero_bytes,
                 state.zero_f16, state.has_zero);
}

auto HnswIndex::prepare_query(std::span<const float> query, HnswWorkspace& workspace) const -> void {
    const auto& state = *state_;
    prepare_quantized_query(query, state.graph.width, state.bits, state.group_size, state.input_scale, workspace.q_i8,
                            workspace.q_scale, workspace.q_bsum);
}

auto HnswIndex::query_distance(std::span<const float> query, const HnswWorkspace& workspace, std::uint32_t node) const
    -> float {
    const auto& state = *state_;
    const auto label = static_cast<std::size_t>(state.graph.labels[node]);
    if (!state.bits) {
        const auto row = label * state.row_bytes;
        const auto result = 1.F - state.dot(query.data(), state.bytes.data() + row, state.graph.width);
        if (!std::isfinite(result)) throw Failure({ErrorCode::RUNTIME, "nonfinite HNSW query distance"});
        return result;
    }
    const auto* w_row = reinterpret_cast<const std::uint8_t*>(state.bytes.data()) + label * state.row_bytes;
    const float score = quantized_row_dot(w_row, label, state.graph.width, state.bits, state.group_size,
                                          state.scale_bytes, state.scale_f16, state.zero_bytes, state.zero_f16,
                                          state.has_zero, workspace.q_i8, workspace.q_scale, workspace.q_bsum);
    const auto result = 1.F - score;
    if (!std::isfinite(result)) throw Failure({ErrorCode::RUNTIME, "nonfinite HNSW query distance"});
    return result;
}

auto HnswIndex::node_distance(std::uint32_t left, std::uint32_t right) const -> float {
    const auto& state = *state_;
    const auto stride = state.row_bytes;
    const auto left_row = static_cast<std::size_t>(state.graph.labels[left]) * stride;
    const auto right_row = static_cast<std::size_t>(state.graph.labels[right]) * stride;
    const auto result =
        1.F - state.row_dot(state.bytes.data() + left_row, state.bytes.data() + right_row, state.graph.width);
    if (!std::isfinite(result)) throw Failure({ErrorCode::RUNTIME, "nonfinite HNSW build distance"});
    return result;
}

auto HnswIndex::build(tensor::Tensor vectors, HnswBuildOptions options, HnswProgress progress,
                      HnswParallelFor parallel_for) -> Result<HnswIndex> {
    try {
        if (!vectors.defined() || !vectors.is_host_accessible() || !vectors.is_contiguous() ||
            vectors.dimensions() != 2 || !vectors.size(0) || vectors.size(0) > MAX_NODES || !vectors.size(1) ||
            vectors.size(1) > std::numeric_limits<std::uint32_t>::max() ||
            (vectors.dtype() != tensor::DType::BF16 && vectors.dtype() != tensor::DType::F32) || options.m < 2 ||
            options.m > 65535 || options.ef_construction < options.m || !options.candidates ||
            options.candidates > vectors.size(0) || options.ef_search < options.candidates ||
            options.ef_search > vectors.size(0))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid HNSW vectors or build/search parameters"});
        Graph graph;
        graph.width = static_cast<std::uint32_t>(vectors.size(1));
        graph.m = options.m;
        graph.weight_type = vectors.dtype() == tensor::DType::BF16 ? 1U : 2U;
        graph.weights_crc32 = hnsw_crc32(require(std::as_const(vectors).host_bytes()));
        graph.candidates = options.candidates;
        graph.ef_search = options.ef_search;
        graph.labels.resize(vectors.size(0));
        std::iota(graph.labels.begin(), graph.labels.end(), 0);
        graph.levels.resize(vectors.size(0));
        HnswIndex index(std::make_unique<State>(graph, std::move(vectors)));
        const auto nodes = graph.labels.size();
        std::vector<BuildNode> build(nodes);
        std::mt19937_64 random(options.seed);
        std::uniform_real_distribution<double> uniform(std::nextafter(0.0, 1.0), 1.0);
        const auto multiplier = 1.0 / std::log(static_cast<double>(options.m));
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto sampled = static_cast<std::size_t>(-std::log(uniform(random)) * multiplier);
            graph.levels[node] = static_cast<std::uint8_t>(std::min<std::size_t>(sampled, 255));
            build[node].links.resize(static_cast<std::size_t>(graph.levels[node]) + 1);
        }
        graph.entry = 0;
        auto maximum_level = static_cast<std::size_t>(graph.levels[0]);
        if (progress) progress(1, nodes);
        std::vector<std::uint32_t> visited(nodes), unseen;
        std::vector<float> unseen_distances;
        std::uint32_t epoch = 0;
        using Pair = std::pair<float, std::uint32_t>;
        std::vector<Pair> nearest, candidates;
        constexpr std::size_t wave_pops = 32, dirty_slack = 512;
        nearest.reserve(options.ef_construction + wave_pops * 2 * static_cast<std::size_t>(options.m) + dirty_slack);
        candidates.reserve(options.ef_construction * 2);
        unseen.reserve(wave_pops * 2 * static_cast<std::size_t>(options.m) + 1);
        unseen_distances.reserve(wave_pops * 2 * static_cast<std::size_t>(options.m) + 1);
        const auto compact_nearest = [&](std::size_t ef) -> float {
            std::nth_element(nearest.begin(), nearest.begin() + static_cast<std::ptrdiff_t>(ef - 1), nearest.end());
            nearest.resize(ef);
            return nearest[ef - 1].first;
        };
        const auto search_layer = [&](std::uint32_t query, std::uint32_t entry, std::size_t ef,
                                      std::size_t level) -> std::span<const Pair> {
            if (++epoch == 0) {
                std::ranges::fill(visited, 0);
                epoch = 1;
            }
            nearest.clear();
            candidates.clear();
            const auto initial = index.node_distance(query, entry);
            nearest.emplace_back(initial, entry);
            candidates.emplace_back(initial, entry);
            visited[entry] = epoch;
            float upper_bound = nearest.size() == ef ? initial : std::numeric_limits<float>::infinity();
            const std::size_t max_pops = !level && ef >= 256 ? wave_pops : 1;
            while (!candidates.empty()) {
                unseen.clear();
                for (std::size_t pop_index = 0; pop_index < max_pops && !candidates.empty(); ++pop_index) {
                    if (candidates.front().first > upper_bound) break;
                    std::pop_heap(candidates.begin(), candidates.end(), std::greater<>{});
                    const auto current = candidates.back().second;
                    candidates.pop_back();
                    for (auto neighbor : build[current].links[level]) {
                        if (visited[neighbor] == epoch) continue;
                        visited[neighbor] = epoch;
                        unseen.push_back(neighbor);
                    }
                }
                if (unseen.empty()) break;
                unseen_distances.resize(unseen.size());
                if (parallel_for && unseen.size() >= 64) {
                    std::mutex mutex;
                    std::optional<Error> failure;
                    require(parallel_for(unseen.size(), [&](std::size_t item) {
                        try {
                            unseen_distances[item] = index.node_distance(query, unseen[item]);
                        } catch (const Failure& error) {
                            std::scoped_lock lock(mutex);
                            if (!failure) failure = error.error();
                        }
                    }));
                    if (failure) throw Failure(std::move(*failure));
                } else {
                    for (std::size_t item = 0; item < unseen.size(); ++item) {
                        if (item + 2 < unseen.size()) index.prefetch_node(unseen[item + 2]);
                        unseen_distances[item] = index.node_distance(query, unseen[item]);
                    }
                }
                for (std::size_t item = 0; item < unseen.size(); ++item) {
                    const auto value = unseen_distances[item];
                    if (value < upper_bound) nearest.emplace_back(value, unseen[item]);
                }
                const bool compacted =
                    (!std::isfinite(upper_bound) && nearest.size() >= ef) || nearest.size() >= ef + dirty_slack;
                if (compacted) {
                    upper_bound = compact_nearest(ef);
                    std::erase_if(candidates, [&](const Pair& item) { return item.first > upper_bound; });
                }
                const bool rebuild_candidates = compacted || unseen.size() >= candidates.size() / 4;
                for (std::size_t item = 0; item < unseen.size(); ++item) {
                    const auto value = unseen_distances[item];
                    if (value < upper_bound) {
                        candidates.emplace_back(value, unseen[item]);
                        if (!rebuild_candidates) std::push_heap(candidates.begin(), candidates.end(), std::greater<>{});
                    }
                }
                if (rebuild_candidates) std::make_heap(candidates.begin(), candidates.end(), std::greater<>{});
            }
            if (nearest.size() > ef) compact_nearest(ef);
            const auto sorted_prefix = std::min<std::size_t>(nearest.size(), std::max<std::size_t>(2 * options.m, 66));
            std::partial_sort(nearest.begin(), nearest.begin() + static_cast<std::ptrdiff_t>(sorted_prefix),
                              nearest.end());
            return nearest;
        };
        const auto select_sorted = [&](std::span<const Pair> pool, std::size_t maximum) {
            const auto prefix_limit = std::max<std::size_t>(maximum, 66);
            const auto front = pool.first(std::min(pool.size(), prefix_limit));
            std::vector<std::uint32_t> selected, rejected;
            selected.reserve(maximum);
            for (std::size_t item = 0; item < front.size(); ++item) {
                if (item + 2 < front.size()) index.prefetch_node(front[item + 2].second);
                const auto [distance, candidate] = front[item];
                const bool diverse = std::ranges::all_of(
                    selected, [&](auto other) { return index.node_distance(candidate, other) >= distance; });
                (diverse ? selected : rejected).push_back(candidate);
                if (selected.size() == maximum) return selected;
            }
            if (front.size() < pool.size()) {
                const auto tail = pool.subspan(prefix_limit);
                std::vector<Pair> survivors;
                if (parallel_for && tail.size() >= 64) {
                    std::vector<std::uint8_t> keep(tail.size(), 0);
                    std::mutex mutex;
                    std::optional<Error> failure;
                    require(parallel_for(tail.size(), [&](std::size_t item) {
                        try {
                            const auto [distance, candidate] = tail[item];
                            keep[item] = std::ranges::all_of(selected, [&](auto other) {
                                return index.node_distance(candidate, other) >= distance;
                            });
                        } catch (const Failure& error) {
                            std::scoped_lock lock(mutex);
                            if (!failure) failure = error.error();
                        }
                    }));
                    if (failure) throw Failure(std::move(*failure));
                    for (std::size_t item = 0; item < tail.size(); ++item)
                        if (keep[item]) survivors.push_back(tail[item]);
                } else {
                    for (std::size_t item = 0; item < tail.size(); ++item) {
                        if (item + 2 < tail.size()) index.prefetch_node(tail[item + 2].second);
                        const auto [distance, candidate] = tail[item];
                        if (std::ranges::all_of(selected, [&](auto other) {
                                return index.node_distance(candidate, other) >= distance;
                            }))
                            survivors.push_back(tail[item]);
                    }
                }
                std::ranges::sort(survivors);
                const auto seed_count = selected.size();
                for (const auto [distance, candidate] : survivors) {
                    if (std::ranges::all_of(std::span(selected).subspan(seed_count), [&](auto other) {
                            return index.node_distance(candidate, other) >= distance;
                        })) {
                        selected.push_back(candidate);
                        if (selected.size() == maximum) return selected;
                    }
                }
            }
            for (auto candidate : rejected) {
                if (selected.size() == maximum) break;
                selected.push_back(candidate);
            }
            return selected;
        };
        const auto select = [&](std::uint32_t query, std::vector<Pair> pool, std::size_t maximum) {
            std::ranges::sort(pool);
            pool.erase(std::unique(pool.begin(), pool.end(),
                                   [](const auto& left, const auto& right) { return left.second == right.second; }),
                       pool.end());
            std::erase_if(pool, [&](const auto& candidate) { return candidate.second == query; });
            return select_sorted(pool, maximum);
        };
        for (std::uint32_t node = 1; node < nodes; ++node) {
            auto entry = graph.entry;
            auto best = index.node_distance(node, entry);
            const auto level = static_cast<std::size_t>(graph.levels[node]);
            for (auto current_level = maximum_level; current_level > level; --current_level) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    const auto& neighbors = build[entry].links[current_level];
                    for (std::size_t item = 0; item < neighbors.size(); ++item) {
                        if (item + 2 < neighbors.size()) index.prefetch_node(neighbors[item + 2]);
                        const auto neighbor = neighbors[item];
                        const auto distance = index.node_distance(node, neighbor);
                        if (distance < best) {
                            best = distance;
                            entry = neighbor;
                            changed = true;
                        }
                    }
                }
            }
            const auto connected_level = std::min(level, maximum_level);
            for (std::size_t offset = 0; offset <= connected_level; ++offset) {
                const auto current_level = connected_level - offset;
                const auto layer_candidates = search_layer(node, entry, options.ef_construction, current_level);
                const auto next_entry = layer_candidates.empty() ? entry : layer_candidates.front().second;
                const auto maximum = current_level ? options.m : 2 * options.m;
                auto selected = select_sorted(layer_candidates, maximum);
                build[node].links[current_level] = selected;
                std::vector<std::uint32_t> overflow;
                for (auto neighbor : selected) {
                    auto& links = build[neighbor].links[current_level];
                    if (!std::ranges::contains(links, node)) links.push_back(node);
                    if (links.size() > maximum) {
                        if (parallel_for) {
                            overflow.push_back(neighbor);
                            continue;
                        }
                        std::vector<Pair> alternatives;
                        alternatives.reserve(links.size());
                        for (std::size_t idx = 0; idx < links.size(); ++idx) {
                            if (idx + 2 < links.size()) index.prefetch_node(links[idx + 2]);
                            alternatives.emplace_back(index.node_distance(neighbor, links[idx]), links[idx]);
                        }
                        links = select(neighbor, std::move(alternatives), maximum);
                    }
                }
                if (!overflow.empty()) {
                    std::mutex mutex;
                    std::optional<Error> failure;
                    require(parallel_for(overflow.size(), [&](std::size_t item) {
                        try {
                            const auto neighbor = overflow[item];
                            auto& links = build[neighbor].links[current_level];
                            std::vector<Pair> alternatives;
                            alternatives.reserve(links.size());
                            for (std::size_t idx = 0; idx < links.size(); ++idx) {
                                if (idx + 2 < links.size()) index.prefetch_node(links[idx + 2]);
                                alternatives.emplace_back(index.node_distance(neighbor, links[idx]), links[idx]);
                            }
                            links = select(neighbor, std::move(alternatives), maximum);
                        } catch (const Failure& error) {
                            std::scoped_lock lock(mutex);
                            if (!failure) failure = error.error();
                        }
                    }));
                    if (failure) throw Failure(std::move(*failure));
                }
                entry = next_entry;
            }
            if (level > maximum_level) {
                graph.entry = node;
                maximum_level = level;
            }
            if (progress) progress(node + 1, nodes);
        }
        graph.base_offsets.reserve(nodes + 1);
        graph.base_offsets.push_back(0);
        for (const auto& node : build) {
            graph.base_edges.insert(graph.base_edges.end(), node.links[0].begin(), node.links[0].end());
            if (graph.base_edges.size() > std::numeric_limits<std::uint32_t>::max())
                throw Failure({ErrorCode::UNSUPPORTED, "HNSW base graph exceeds uint32 offset capacity"});
            graph.base_offsets.push_back(static_cast<std::uint32_t>(graph.base_edges.size()));
        }
        graph.upper.resize(maximum_level);
        for (std::size_t level = 1; level <= maximum_level; ++level) {
            auto& destination = graph.upper[level - 1];
            destination.offsets.push_back(0);
            for (std::size_t node = 0; node < nodes; ++node) {
                if (build[node].links.size() <= level) continue;
                destination.nodes.push_back(static_cast<std::uint32_t>(node));
                destination.edges.insert(destination.edges.end(), build[node].links[level].begin(),
                                         build[node].links[level].end());
                if (destination.edges.size() > std::numeric_limits<std::uint32_t>::max())
                    throw Failure({ErrorCode::UNSUPPORTED, "HNSW upper graph exceeds uint32 offset capacity"});
                destination.offsets.push_back(static_cast<std::uint32_t>(destination.edges.size()));
            }
        }
        index.state_->graph = std::move(graph);
        validate_graph(index.state_->graph);
        return index;
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}

auto HnswIndex::search(std::span<const float> query, HnswWorkspace& workspace, std::size_t candidates,
                       std::size_t ef) const -> Result<std::span<const HnswNeighbor>> {
    try {
        const auto& graph = state_->graph;
        if (!candidates) candidates = graph.candidates;
        if (!ef) ef = graph.ef_search;
        if (query.size() != graph.width || !candidates || candidates > graph.labels.size() || ef < candidates ||
            std::ranges::any_of(query, [](float value) { return !std::isfinite(value); }))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid HNSW query or search parameters"});
        prepare_query(query, workspace);
        const auto nodes = graph.labels.size();
        ef = std::min(ef, nodes);
        if (workspace.visited.size() != nodes) {
            workspace.visited.assign(nodes, 0);
            workspace.epoch = 0;
        }
        workspace.frontier.reserve(ef + 1);
        workspace.nearest.reserve(ef + 1);
        if (++workspace.epoch == 0) {
            std::ranges::fill(workspace.visited, 0);
            workspace.epoch = 1;
        }
        workspace.frontier.clear();
        workspace.nearest.clear();
        workspace.result.clear();
        workspace.result.reserve(candidates);
        auto entry = graph.entry;
        auto best = query_distance(query, workspace, entry);
        for (std::size_t level = graph.upper.size(); level; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                const auto neighbors = graph.neighbors(entry, level);
                for (std::size_t idx = 0; idx < neighbors.size(); ++idx) {
                    if (idx + 2 < neighbors.size()) prefetch_node(neighbors[idx + 2]);
                    const auto neighbor = neighbors[idx];
                    const auto value = query_distance(query, workspace, neighbor);
                    if (value < best) {
                        best = value;
                        entry = neighbor;
                        changed = true;
                    }
                }
            }
        }
        push(workspace.frontier, -best, entry);
        push(workspace.nearest, best, entry);
        workspace.visited[entry] = workspace.epoch;
        while (!workspace.frontier.empty()) {
            if (-workspace.frontier.front().first > best) break;
            const auto current = pop(workspace.frontier).second;
            const auto neighbors = graph.neighbors(current, 0);
            for (std::size_t idx = 0; idx < neighbors.size(); ++idx) {
                if (idx + 2 < neighbors.size() && workspace.visited[neighbors[idx + 2]] != workspace.epoch)
                    prefetch_node(neighbors[idx + 2]);
                const auto neighbor = neighbors[idx];
                if (workspace.visited[neighbor] == workspace.epoch) continue;
                workspace.visited[neighbor] = workspace.epoch;
                const auto value = query_distance(query, workspace, neighbor);
                if (workspace.nearest.size() < ef || value < best) {
                    push(workspace.frontier, -value, neighbor);
                    push(workspace.nearest, value, neighbor);
                    if (workspace.nearest.size() > ef) pop(workspace.nearest);
                    best = workspace.nearest.front().first;
                }
            }
        }
        if (workspace.nearest.size() < candidates)
            throw Failure({ErrorCode::RUNTIME, "HNSW search returned too few candidates"});
        std::partial_sort(
            workspace.nearest.begin(), workspace.nearest.begin() + candidates, workspace.nearest.end(),
            [&](const auto& left, const auto& right) {
                return left.first < right.first ||
                       (left.first == right.first && graph.labels[left.second] < graph.labels[right.second]);
            });
        for (std::size_t rank = 0; rank < candidates; ++rank) {
            const auto [distance, node] = workspace.nearest[rank];
            workspace.result.push_back({distance, graph.labels[node]});
        }
        return std::span<const HnswNeighbor>(workspace.result);
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}

} // namespace kidi::search
