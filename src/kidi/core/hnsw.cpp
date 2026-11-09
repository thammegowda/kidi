#include "kidi/core/hnsw.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <tuple>
#include <type_traits>
#include <zlib.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KIDI_HNSW_AVX512 1
#endif

namespace kidi::core {
namespace {
constexpr std::size_t HEADER_BYTES = 128, HEADER_CRC_OFFSET = 52;
constexpr std::uint32_t MAX_NODES = 1U << 24;
constexpr std::uint32_t FORMAT_VERSION = 2;

class Failure {
public:
    explicit Failure(Error error) : error_(std::move(error)) {}
    auto error() const -> const Error& { return error_; }

private:
    Error error_;
};

template <typename T>
auto require(Result<T> result) -> T {
    if (!result) throw Failure(std::move(result.error()));
    if constexpr (!std::is_void_v<T>) return std::move(*result);
}

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

auto hnsw_crc32(std::span<const std::byte> bytes) -> std::uint32_t {
    return static_cast<std::uint32_t>(crc32_z(0, reinterpret_cast<const Bytef*>(bytes.data()), bytes.size()));
}

auto invalid(bool condition, const char* message) -> void {
    if (condition) throw Failure({ErrorCode::INVALID_MANIFEST, message});
}

class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}
    auto take(std::size_t count) -> std::span<const std::byte> {
        invalid(count > bytes_.size() - position_, "truncated HNSW index");
        const auto result = bytes_.subspan(position_, count);
        position_ += count;
        return result;
    }
    auto integer(std::size_t count) -> std::uint64_t {
        std::uint64_t value = 0;
        const auto bytes = take(count);
        for (std::size_t index = 0; index < count; ++index)
            value |= std::to_integer<std::uint64_t>(bytes[index]) << (8 * index);
        return value;
    }
    auto position() const -> std::size_t { return position_; }
    auto remaining() const -> std::size_t { return bytes_.size() - position_; }

private:
    std::span<const std::byte> bytes_;
    std::size_t position_ = 0;
};

auto append(std::vector<std::byte>& bytes, std::uint64_t value, std::size_t count) -> void {
    for (std::size_t index = 0; index < count; ++index)
        bytes.push_back(static_cast<std::byte>((value >> (8 * index)) & 255U));
}
auto write(std::span<std::byte> bytes, std::size_t offset, std::uint64_t value, std::size_t count) -> void {
    for (std::size_t index = 0; index < count; ++index)
        bytes[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255U);
}

class StagingFile {
public:
    std::filesystem::path directory;
    auto file() const -> std::filesystem::path { return directory / "index"; }
    auto cleanup() -> Result<void> {
        if (directory.empty()) return {};
        std::error_code error;
        std::filesystem::remove(file(), error);
        if (!error) std::filesystem::remove(directory, error);
        if (error)
            return std::unexpected(Error{ErrorCode::IO, "cannot clean HNSW staging directory: " + error.message()});
        directory.clear();
        return {};
    }
    ~StagingFile() {
        const auto result = cleanup();
        if (!result) std::cerr << "kidi_hnsw_save|cleanup_error=" << result.error().message << '\n';
    }
};

auto read_file(const std::filesystem::path& path) -> Result<std::vector<std::byte>> {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return std::unexpected(Error{ErrorCode::IO, "cannot open HNSW index: " + path.string()});
    const auto size = stream.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > std::numeric_limits<std::size_t>::max() ||
        static_cast<std::uint64_t>(size) > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
        return std::unexpected(Error{ErrorCode::IO, "cannot determine HNSW index size"});
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        return std::unexpected(Error{ErrorCode::IO, "failed reading HNSW index"});
    return bytes;
}

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
    const auto version = reader.integer(4), header_bytes = reader.integer(4), nodes = reader.integer(4);
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

template <bool bfloat>
auto weight(const std::byte* row, std::size_t column) -> float {
    if constexpr (bfloat)
        return std::bit_cast<float>(static_cast<std::uint32_t>(reinterpret_cast<const std::uint16_t*>(row)[column])
                                    << 16);
    else
        return reinterpret_cast<const float*>(row)[column];
}
template <bool bfloat>
auto dot_portable(const float* query, const std::byte* row, std::size_t width) -> float {
    std::array<float, 16> lanes{};
    std::size_t column = 0;
    for (; column + 16 <= width; column += 16)
        for (std::size_t lane = 0; lane < 16; ++lane)
            lanes[lane] = std::fma(query[column + lane], weight<bfloat>(row, column + lane), lanes[lane]);
    for (std::size_t offset = 8; offset; offset /= 2)
        for (std::size_t lane = 0; lane < offset; ++lane) lanes[lane] += lanes[lane + offset];
    float tail = 0;
    for (; column < width; ++column) tail = std::fma(query[column], weight<bfloat>(row, column), tail);
    return lanes[0] + tail;
}
template <bool bfloat>
auto dot_rows_portable(const std::byte* left, const std::byte* right, std::size_t width) -> float {
    std::array<float, 16> lanes{};
    std::size_t column = 0;
    for (; column + 16 <= width; column += 16)
        for (std::size_t lane = 0; lane < 16; ++lane)
            lanes[lane] =
                std::fma(weight<bfloat>(left, column + lane), weight<bfloat>(right, column + lane), lanes[lane]);
    for (std::size_t offset = 8; offset; offset /= 2)
        for (std::size_t lane = 0; lane < offset; ++lane) lanes[lane] += lanes[lane + offset];
    float tail = 0;
    for (; column < width; ++column) tail = std::fma(weight<bfloat>(left, column), weight<bfloat>(right, column), tail);
    return lanes[0] + tail;
}
#if defined(KIDI_HNSW_AVX512)
template <bool bfloat>
__attribute__((target("avx512f,avx512bw,fma"))) auto dot_avx512(const float* query, const std::byte* row,
                                                                std::size_t width) -> float {
    auto sum = _mm512_setzero_ps();
    std::size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        __m512 values;
        if constexpr (bfloat) {
            const auto packed =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + column * sizeof(std::uint16_t)));
            values = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
        } else {
            values = _mm512_loadu_ps(reinterpret_cast<const float*>(row) + column);
        }
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(query + column), values, sum);
    }
    float tail = 0;
    for (; column < width; ++column) tail = std::fma(query[column], weight<bfloat>(row, column), tail);
    return _mm512_reduce_add_ps(sum) + tail;
}
template <bool bfloat>
__attribute__((target("avx512f,avx512bw,fma"))) auto dot_rows_avx512(const std::byte* left, const std::byte* right,
                                                                     std::size_t width) -> float {
    auto sum = _mm512_setzero_ps();
    std::size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        __m512 first, second;
        if constexpr (bfloat) {
            const auto left_values =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left + column * sizeof(std::uint16_t)));
            const auto right_values =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right + column * sizeof(std::uint16_t)));
            first = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(left_values), 16));
            second = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(right_values), 16));
        } else {
            first = _mm512_loadu_ps(reinterpret_cast<const float*>(left) + column);
            second = _mm512_loadu_ps(reinterpret_cast<const float*>(right) + column);
        }
        sum = _mm512_fmadd_ps(first, second, sum);
    }
    float tail = 0;
    for (; column < width; ++column) tail = std::fma(weight<bfloat>(left, column), weight<bfloat>(right, column), tail);
    return _mm512_reduce_add_ps(sum) + tail;
}
#endif

struct ByDistance {
    auto operator()(const std::pair<float, std::uint32_t>& left, const std::pair<float, std::uint32_t>& right) const
        -> bool {
        return left.first < right.first;
    }
};
auto push(std::vector<std::pair<float, std::uint32_t>>& heap, float distance, std::uint32_t node) -> void {
    heap.emplace_back(distance, node);
    std::push_heap(heap.begin(), heap.end(), ByDistance{});
}
auto pop(std::vector<std::pair<float, std::uint32_t>>& heap) -> std::pair<float, std::uint32_t> {
    std::pop_heap(heap.begin(), heap.end(), ByDistance{});
    const auto result = heap.back();
    heap.pop_back();
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

using Dot = float (*)(const float*, const std::byte*, std::size_t);
using RowDot = float (*)(const std::byte*, const std::byte*, std::size_t);

auto dot_kernels(tensor::DType dtype) -> std::pair<Dot, RowDot> {
    Dot dot = dtype == tensor::DType::BF16 ? dot_portable<true> : dot_portable<false>;
    RowDot row_dot = dtype == tensor::DType::BF16 ? dot_rows_portable<true> : dot_rows_portable<false>;
#if defined(KIDI_HNSW_AVX512)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("fma")) {
        dot = dtype == tensor::DType::BF16 ? dot_avx512<true> : dot_avx512<false>;
        row_dot = dtype == tensor::DType::BF16 ? dot_rows_avx512<true> : dot_rows_avx512<false>;
    }
#endif
    return {dot, row_dot};
}
} // namespace

struct HnswIndex::State {
    Graph graph;
    tensor::Tensor vectors;
    std::span<const std::byte> bytes;
    Dot dot;
    RowDot row_dot;

    State(Graph topology, tensor::Tensor table)
        : graph(std::move(topology)), vectors(std::move(table)), bytes(require(std::as_const(vectors).host_bytes())) {
        std::tie(dot, row_dot) = dot_kernels(vectors.dtype());
    }
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
                const auto expected =
                    vectors.dtype() == tensor::DType::BF16 ? weight<true>(row, column) : weight<false>(row, column);
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
        if (path.empty())
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "HNSW output path must not be empty"});
        const auto destination = std::filesystem::absolute(path);
        std::error_code error;
        const auto status = std::filesystem::symlink_status(destination, error);
        if (error && error != std::errc::no_such_file_or_directory)
            return std::unexpected(Error{ErrorCode::IO, "cannot inspect HNSW output: " + error.message()});
        if (status.type() != std::filesystem::file_type::not_found)
            return std::unexpected(Error{ErrorCode::IO, "HNSW output already exists: " + destination.string()});
        const auto bytes = encode(state_->graph);
        if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            return std::unexpected(Error{ErrorCode::IO, "HNSW output exceeds stream size limits"});
        StagingFile staging;
        std::random_device random;
        for (int attempt = 0; attempt < 16 && staging.directory.empty(); ++attempt) {
            const auto directory = destination.parent_path() / (".kidi-hnsw-" + std::to_string(random()));
            error.clear();
            if (std::filesystem::create_directory(directory, error)) {
                staging.directory = directory;
            } else if (error && error != std::errc::file_exists) {
                return std::unexpected(
                    Error{ErrorCode::IO, "cannot create HNSW staging directory: " + error.message()});
            }
        }
        if (staging.directory.empty())
            return std::unexpected(Error{ErrorCode::IO, "cannot allocate a unique HNSW staging directory"});
        std::ofstream output(staging.file(), std::ios::binary | std::ios::trunc);
        if (!output) return std::unexpected(Error{ErrorCode::IO, "cannot create staged HNSW index"});
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        output.flush();
        if (!output) return std::unexpected(Error{ErrorCode::IO, "failed writing HNSW index"});
        output.close();
        if (!output) return std::unexpected(Error{ErrorCode::IO, "failed closing HNSW index"});
        std::filesystem::create_hard_link(staging.file(), destination, error);
        if (error) return std::unexpected(Error{ErrorCode::IO, "cannot publish HNSW index: " + error.message()});
        const auto cleaned = staging.cleanup();
        if (!cleaned) return cleaned;
        return {};
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::system_error& error) {
        return std::unexpected(Error{ErrorCode::IO, "cannot save HNSW index: " + std::string(error.what())});
    }
}
auto HnswIndex::query_distance(std::span<const float> query, std::uint32_t node) const -> float {
    const auto& state = *state_;
    const auto row = static_cast<std::size_t>(state.graph.labels[node]) * state.graph.width *
                     tensor::element_size(state.vectors.dtype());
    const auto result = 1.F - state.dot(query.data(), state.bytes.data() + row, state.graph.width);
    if (!std::isfinite(result)) throw Failure({ErrorCode::RUNTIME, "nonfinite HNSW query distance"});
    return result;
}
auto HnswIndex::node_distance(std::uint32_t left, std::uint32_t right) const -> float {
    const auto& state = *state_;
    const auto stride = state.graph.width * tensor::element_size(state.vectors.dtype());
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
        const auto distances = [&](std::uint32_t query, std::span<const std::uint32_t> neighbors) {
            std::vector<float> result(neighbors.size());
            if (!parallel_for || neighbors.size() < 8) {
                for (std::size_t item = 0; item < neighbors.size(); ++item)
                    result[item] = index.node_distance(query, neighbors[item]);
                return result;
            }
            std::mutex mutex;
            std::optional<Error> failure;
            require(parallel_for(neighbors.size(), [&](std::size_t item) {
                try {
                    result[item] = index.node_distance(query, neighbors[item]);
                } catch (const Failure& error) {
                    std::scoped_lock lock(mutex);
                    if (!failure) failure = error.error();
                }
            }));
            if (failure) throw Failure(std::move(*failure));
            return result;
        };
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
        std::vector<std::uint32_t> visited(nodes);
        std::uint32_t epoch = 0;
        const auto search_layer = [&](std::uint32_t query, std::uint32_t entry, std::size_t ef, std::size_t level) {
            if (++epoch == 0) {
                std::ranges::fill(visited, 0);
                epoch = 1;
            }
            using Pair = std::pair<float, std::uint32_t>;
            std::priority_queue<Pair> nearest;
            std::priority_queue<Pair, std::vector<Pair>, std::greater<>> candidates;
            const auto initial = index.node_distance(query, entry);
            nearest.emplace(initial, entry);
            candidates.emplace(initial, entry);
            visited[entry] = epoch;
            while (!candidates.empty()) {
                const auto [distance, current] = candidates.top();
                if (nearest.size() == ef && distance > nearest.top().first) break;
                candidates.pop();
                std::vector<std::uint32_t> unseen;
                for (auto neighbor : build[current].links[level]) {
                    if (visited[neighbor] == epoch) continue;
                    visited[neighbor] = epoch;
                    unseen.push_back(neighbor);
                }
                const auto values = distances(query, unseen);
                for (std::size_t item = 0; item < unseen.size(); ++item) {
                    const auto neighbor = unseen[item];
                    const auto value = values[item];
                    if (nearest.size() < ef || value < nearest.top().first) {
                        candidates.emplace(value, neighbor);
                        nearest.emplace(value, neighbor);
                        if (nearest.size() > ef) nearest.pop();
                    }
                }
            }
            std::vector<Pair> result;
            result.reserve(nearest.size());
            while (!nearest.empty()) {
                result.push_back(nearest.top());
                nearest.pop();
            }
            std::ranges::sort(result, [](const Pair& left, const Pair& right) {
                return left.first < right.first || (left.first == right.first && left.second < right.second);
            });
            return result;
        };
        const auto select = [&](std::uint32_t query, std::vector<std::pair<float, std::uint32_t>> candidates,
                                std::size_t maximum) {
            std::ranges::sort(candidates, [](const auto& left, const auto& right) {
                return left.first < right.first || (left.first == right.first && left.second < right.second);
            });
            candidates.erase(
                std::unique(candidates.begin(), candidates.end(),
                            [](const auto& left, const auto& right) { return left.second == right.second; }),
                candidates.end());
            std::vector<std::uint32_t> selected, rejected;
            selected.reserve(maximum);
            for (const auto [distance, candidate] : candidates) {
                if (candidate == query) continue;
                bool diverse;
                if (parallel_for && selected.size() >= 8) {
                    const auto values = distances(candidate, selected);
                    diverse = std::ranges::all_of(values, [&](auto value) { return value >= distance; });
                } else {
                    diverse = std::ranges::all_of(
                        selected, [&](auto other) { return index.node_distance(candidate, other) >= distance; });
                }
                (diverse ? selected : rejected).push_back(candidate);
                if (selected.size() == maximum) break;
            }
            for (auto candidate : rejected) {
                if (selected.size() == maximum) break;
                selected.push_back(candidate);
            }
            return selected;
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
                    const auto values = distances(node, neighbors);
                    for (std::size_t item = 0; item < neighbors.size(); ++item) {
                        const auto neighbor = neighbors[item];
                        const auto distance = values[item];
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
                auto candidates = search_layer(node, entry, options.ef_construction, current_level);
                const auto maximum = current_level ? options.m : 2 * options.m;
                auto selected = select(node, candidates, maximum);
                build[node].links[current_level] = selected;
                for (auto neighbor : selected) {
                    auto& links = build[neighbor].links[current_level];
                    if (!std::ranges::contains(links, node)) links.push_back(node);
                    if (links.size() > maximum) {
                        std::vector<std::pair<float, std::uint32_t>> alternatives;
                        alternatives.reserve(links.size());
                        const auto values = distances(neighbor, links);
                        for (std::size_t item = 0; item < links.size(); ++item)
                            alternatives.emplace_back(values[item], links[item]);
                        links = select(neighbor, std::move(alternatives), maximum);
                    }
                }
                if (!candidates.empty()) entry = candidates.front().second;
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
        const auto nodes = graph.labels.size();
        if (workspace.visited.size() != nodes) {
            workspace.visited.assign(nodes, 0);
            workspace.frontier.reserve(ef + 1);
            workspace.nearest.reserve(ef + 1);
            workspace.epoch = 0;
        }
        if (++workspace.epoch == 0) {
            std::ranges::fill(workspace.visited, 0);
            workspace.epoch = 1;
        }
        workspace.frontier.clear();
        workspace.nearest.clear();
        workspace.result.clear();
        ef = std::min(ef, nodes);
        workspace.result.reserve(candidates);
        auto entry = graph.entry;
        auto best = query_distance(query, entry);
        for (std::size_t level = graph.upper.size(); level; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (auto neighbor : graph.neighbors(entry, level)) {
                    const auto value = query_distance(query, neighbor);
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
            for (auto neighbor : graph.neighbors(current, 0)) {
                if (workspace.visited[neighbor] == workspace.epoch) continue;
                workspace.visited[neighbor] = workspace.epoch;
                const auto value = query_distance(query, neighbor);
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
        while (workspace.nearest.size() > candidates) pop(workspace.nearest);
        while (!workspace.nearest.empty()) {
            const auto [distance, node] = pop(workspace.nearest);
            workspace.result.push_back({distance, graph.labels[node]});
        }
        std::ranges::sort(workspace.result, [](const HnswNeighbor& left, const HnswNeighbor& right) {
            return left.distance < right.distance || (left.distance == right.distance && left.id < right.id);
        });
        return std::span<const HnswNeighbor>(workspace.result);
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}
} // namespace kidi::core
