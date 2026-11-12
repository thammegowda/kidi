#include "kidi/search/hnsw.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <zlib.h>

namespace kidi::search {
namespace {
constexpr std::size_t HEADER_BYTES = 128, HEADER_CRC_OFFSET = 52;
constexpr std::uint32_t MAX_NODES = ynn::kHnswMaxNodes;
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

inline auto invalid(bool condition, const char* message) -> void {
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

inline auto append(std::vector<std::byte>& bytes, std::uint64_t value, std::size_t count) -> void {
    for (std::size_t index = 0; index < count; ++index)
        bytes.push_back(static_cast<std::byte>((value >> (8 * index)) & 255U));
}

inline auto write(std::span<std::byte> bytes, std::size_t offset, std::uint64_t value, std::size_t count) -> void {
    for (std::size_t index = 0; index < count; ++index)
        bytes[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255U);
}

auto hnsw_crc32(std::span<const std::byte> bytes) -> std::uint32_t {
    return static_cast<std::uint32_t>(crc32_z(0, reinterpret_cast<const Bytef*>(bytes.data()), bytes.size()));
}

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

auto save_atomic_file(const std::filesystem::path& path, std::span<const std::byte> bytes) -> Result<void> {
    if (path.empty()) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "HNSW output path must not be empty"});
    const auto destination = std::filesystem::absolute(path);
    std::error_code error;
    const auto status = std::filesystem::symlink_status(destination, error);
    if (error && error != std::errc::no_such_file_or_directory)
        return std::unexpected(Error{ErrorCode::IO, "cannot inspect HNSW output: " + error.message()});
    if (status.type() != std::filesystem::file_type::not_found)
        return std::unexpected(Error{ErrorCode::IO, "HNSW output already exists: " + destination.string()});
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
            return std::unexpected(Error{ErrorCode::IO, "cannot create HNSW staging directory: " + error.message()});
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
    return staging.cleanup();
}

struct SerializedGraph {
    ynn::hnsw_graph topology;
    std::uint32_t weight_type = 0, weights_crc32 = 0;
    std::size_t candidates = 8, ef_search = 800;
    std::uint64_t disk_bytes = 0;
};

auto validate_serialized(const SerializedGraph& graph) -> void {
    const auto nodes = graph.topology.num_nodes();
    invalid(!ynn::validate_hnsw_graph(graph.topology) || !graph.candidates || graph.candidates > nodes ||
                graph.ef_search < graph.candidates || graph.ef_search > nodes ||
                (graph.weight_type != 1 && graph.weight_type != 2),
            "invalid HNSW dimensions, search defaults, or weight type");
}

auto decode(std::span<const std::byte> bytes) -> SerializedGraph {
    invalid(bytes.size() < HEADER_BYTES, "truncated HNSW header");
    invalid(std::memcmp(bytes.data(), "KIDIHNSW", 8) != 0, "invalid HNSW magic");
    std::array<std::byte, HEADER_BYTES> header;
    std::ranges::copy(bytes.first(HEADER_BYTES), header.begin());
    std::fill_n(header.begin() + HEADER_CRC_OFFSET, 4, std::byte{0});
    Reader reader(bytes);
    reader.take(8);
    const auto version = reader.integer(4), header_bytes = reader.integer(4);
    const auto nodes = reader.integer(4);
    SerializedGraph graph;
    auto& topo = graph.topology;
    topo.width = static_cast<std::uint32_t>(reader.integer(4));
    reader.integer(4);
    topo.m = static_cast<std::uint32_t>(reader.integer(4));
    const auto maximum_level = reader.integer(4);
    topo.entry = static_cast<std::uint32_t>(reader.integer(4));
    graph.weight_type = static_cast<std::uint32_t>(reader.integer(4));
    graph.weights_crc32 = static_cast<std::uint32_t>(reader.integer(4));
    const auto payload_crc32 = reader.integer(4), header_crc32 = reader.integer(4);
    const auto map_offset = reader.integer(8), base_offset = reader.integer(8), upper_offset = reader.integer(8);
    graph.disk_bytes = reader.integer(8);
    const auto metadata = reader.take(32);
    reader.integer(8);
    invalid(version != FORMAT_VERSION || header_bytes != HEADER_BYTES || hnsw_crc32(header) != header_crc32,
            "unsupported HNSW version or invalid header checksum");
    invalid(!nodes || nodes > MAX_NODES || !topo.width || topo.m < 2 || topo.m > 65535 || maximum_level > 255 ||
                topo.entry >= nodes || map_offset != HEADER_BYTES || base_offset != map_offset + 4 * nodes ||
                upper_offset < base_offset + 4 * (nodes + 1) || upper_offset > bytes.size() ||
                graph.disk_bytes != bytes.size(),
            "invalid HNSW dimensions, offsets, or file size");
    invalid(hnsw_crc32(bytes.subspan(HEADER_BYTES)) != payload_crc32, "HNSW payload checksum mismatch");
    Reader settings(metadata);
    graph.candidates = settings.integer(4);
    graph.ef_search = settings.integer(4);
    topo.labels.resize(nodes);
    topo.levels.resize(nodes);
    for (auto& label : topo.labels) label = static_cast<std::uint32_t>(reader.integer(4));
    invalid(reader.position() != base_offset, "HNSW label section length mismatch");
    invalid(upper_offset < base_offset + 4 * (nodes + 1) || (upper_offset - base_offset - 4 * (nodes + 1)) % 3,
            "invalid sparse HNSW base section");
    topo.base_offsets.resize(nodes + 1);
    for (auto& offset : topo.base_offsets) offset = static_cast<std::uint32_t>(reader.integer(4));
    topo.base_edges.resize((upper_offset - reader.position()) / 3);
    for (auto& edge : topo.base_edges) edge = static_cast<std::uint32_t>(reader.integer(3));
    invalid(reader.position() != upper_offset, "HNSW base section length mismatch");
    topo.upper.reserve(maximum_level);
    for (std::size_t level = 1; level <= maximum_level; ++level) {
        const auto count = reader.integer(4);
        invalid(!count || count > nodes || count > reader.remaining() / 5,
                "invalid HNSW upper-layer active-node count");
        ynn::hnsw_layer layer;
        layer.nodes.reserve(count);
        layer.offsets.reserve(count + 1);
        layer.offsets.push_back(0);
        for (std::size_t index = 0; index < count; ++index) {
            const auto node = static_cast<std::uint32_t>(reader.integer(3));
            const auto length = reader.integer(2);
            invalid(node >= nodes || (!layer.nodes.empty() && node <= layer.nodes.back()) ||
                        topo.levels[node] != level - 1 || length > topo.m || length > reader.remaining() / 3 ||
                        length > std::numeric_limits<std::uint32_t>::max() - layer.edges.size(),
                    "invalid HNSW upper-layer membership or list");
            layer.nodes.push_back(node);
            topo.levels[node] = static_cast<std::uint8_t>(level);
            for (std::size_t neighbor = 0; neighbor < length; ++neighbor)
                layer.edges.push_back(static_cast<std::uint32_t>(reader.integer(3)));
            layer.offsets.push_back(static_cast<std::uint32_t>(layer.edges.size()));
        }
        topo.upper.push_back(std::move(layer));
    }
    invalid(reader.remaining(), "unexpected HNSW trailing bytes");
    validate_serialized(graph);
    return graph;
}

auto encode(const SerializedGraph& graph) -> std::vector<std::byte> {
    validate_serialized(graph);
    const auto& topo = graph.topology;
    std::vector<std::byte> payload;
    payload.reserve(topo.labels.size() * 4 + topo.base_offsets.size() * 4 + topo.base_edges.size() * 3);
    for (auto label : topo.labels) append(payload, label, 4);
    const auto base_offset = HEADER_BYTES + payload.size();
    for (auto offset : topo.base_offsets) append(payload, offset, 4);
    for (auto edge : topo.base_edges) append(payload, edge, 3);
    const auto upper_offset = HEADER_BYTES + payload.size();
    for (const auto& layer : topo.upper) {
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
    write(result, 16, topo.labels.size(), 4);
    write(result, 20, topo.width, 4);
    const auto maximum_degree = [&] {
        std::uint32_t value = 0;
        for (std::size_t node = 0; node < topo.labels.size(); ++node)
            value = std::max(value, topo.base_offsets[node + 1] - topo.base_offsets[node]);
        return value;
    }();
    write(result, 24, maximum_degree, 4);
    write(result, 28, topo.m, 4);
    write(result, 32, topo.upper.size(), 4);
    write(result, 36, topo.entry, 4);
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

auto wrap_parallel_for(const HnswParallelFor& parallel_for, std::mutex& mutex, std::optional<Error>& failure)
    -> ynn::hnsw_parallel_for_fn {
    if (!parallel_for) return {};
    return [&](std::size_t tasks, std::function<void(std::size_t)> body) -> bool {
        auto result = parallel_for(tasks, std::move(body));
        if (!result) {
            std::scoped_lock lock(mutex);
            if (!failure) failure = std::move(result.error());
            return false;
        }
        return true;
    };
}

auto status_error(ynn_status status, std::optional<Error>& failure, const char* invalid_msg, const char* runtime_msg)
    -> Error {
    if (failure) return std::move(*failure);
    if (status == ynn_status_invalid_parameter) return {ErrorCode::INVALID_ARGUMENT, invalid_msg};
    if (status == ynn_status_unsupported_parameter)
        return {ErrorCode::UNSUPPORTED, "HNSW graph exceeds uint32 offset capacity"};
    return {ErrorCode::RUNTIME, runtime_msg};
}

} // namespace

struct HnswIndex::State {
    SerializedGraph graph;
    tensor::Tensor vectors;
    ynn::hnsw_search_index search_index;

    State(SerializedGraph topology, tensor::Tensor table) : graph(std::move(topology)), vectors(std::move(table)) {
        const auto bytes = require(std::as_const(vectors).host_bytes());
        const auto status = ynn::prepare_hnsw_dense_search(graph.topology, bytes.data(),
                                                           vectors.dtype() == tensor::DType::BF16, &search_index);
        if (status != ynn_status_success)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "failed preparing dense HNSW search layout"});
    }

    State(SerializedGraph topology, tensor::Tensor weight_table, const tensor::Tensor& scale_table,
          const tensor::Tensor& zero_table, std::int32_t weight_bits, std::int32_t group, float act_scale)
        : graph(std::move(topology)), vectors(std::move(weight_table)) {
        const auto w_bytes = require(std::as_const(vectors).host_bytes());
        const auto s_bytes = require(scale_table.host_bytes());
        const auto z_bytes = zero_table.defined() ? require(zero_table.host_bytes()) : std::span<const std::byte>{};
        const auto status = ynn::prepare_hnsw_quantized_search(
            graph.topology, w_bytes.data(), s_bytes.data(), scale_table.dtype() == tensor::DType::F16,
            zero_table.defined() ? z_bytes.data() : nullptr,
            zero_table.defined() && zero_table.dtype() == tensor::DType::F16, weight_bits, group, act_scale,
            &search_index);
        if (status != ynn_status_success)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "failed preparing quantized HNSW search layout"});
        vectors.release_pages(false);
    }
};

HnswIndex::HnswIndex(std::unique_ptr<State> state) : state_(std::move(state)) {}
HnswIndex::HnswIndex(HnswIndex&&) noexcept = default;
auto HnswIndex::operator=(HnswIndex&&) noexcept -> HnswIndex& = default;
HnswIndex::~HnswIndex() = default;
auto HnswIndex::size() const noexcept -> std::size_t { return state_->graph.topology.num_nodes(); }
auto HnswIndex::width() const noexcept -> std::size_t { return state_->graph.topology.width; }
auto HnswIndex::candidates() const noexcept -> std::size_t { return state_->graph.candidates; }
auto HnswIndex::ef_search() const noexcept -> std::size_t { return state_->graph.ef_search; }
auto HnswIndex::runtime_bytes() const noexcept -> std::size_t { return state_->graph.topology.runtime_bytes(); }
auto HnswIndex::vectors() const noexcept -> const tensor::Tensor& { return state_->vectors; }

auto HnswIndex::load(const std::filesystem::path& path, tensor::Tensor vectors) -> Result<HnswIndex> {
    auto bytes = read_file(path);
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    try {
        auto graph = decode(*bytes);
        const auto& topo = graph.topology;
        if (!vectors.defined() || !vectors.is_host_accessible() || !vectors.is_contiguous() ||
            vectors.dimensions() != 2 || vectors.size(0) != topo.num_nodes() || vectors.size(1) != topo.width ||
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
        const auto& topo = graph.topology;
        if ((bits != 2 && bits != 4 && bits != 8) || !std::isfinite(input_scale) || input_scale < 0 ||
            group_size <= 0 || topo.width % group_size != 0 || group_size % 32 != 0 || !weights.defined() ||
            !weights.is_host_accessible() || !weights.is_contiguous() || weights.dtype() != tensor::DType::U8 ||
            weights.dimensions() != 2 || weights.size(0) != topo.num_nodes() ||
            weights.size(1) != topo.width / (8 / bits) || !scales.defined() || !scales.is_host_accessible() ||
            !scales.is_contiguous() || scales.dimensions() != 2 || scales.size(0) != topo.num_nodes() ||
            scales.size(1) != topo.width / group_size ||
            (scales.dtype() != tensor::DType::F32 && scales.dtype() != tensor::DType::F16) ||
            (zeros.defined() &&
             (bits != 4 || !zeros.is_host_accessible() || !zeros.is_contiguous() || zeros.dimensions() != 2 ||
              zeros.size(0) != scales.size(0) || zeros.size(1) != scales.size(1) ||
              (zeros.dtype() != tensor::DType::F32 && zeros.dtype() != tensor::DType::F16))))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "HNSW index does not match quantized vector table"});
        return HnswIndex(std::make_unique<State>(std::move(graph), std::move(weights), scales, zeros, bits, group_size,
                                                 input_scale));
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
        SerializedGraph graph;
        auto& topo = graph.topology;
        topo.width = static_cast<std::uint32_t>(vectors.size(1));
        topo.m = static_cast<std::uint32_t>(m);
        topo.entry = static_cast<std::uint32_t>(entry);
        graph.weight_type = vectors.dtype() == tensor::DType::BF16 ? 1U : 2U;
        graph.candidates = candidates;
        graph.ef_search = ef_search;
        const auto table = require(std::as_const(vectors).host_bytes());
        graph.weights_crc32 = hnsw_crc32(table);
        topo.labels.resize(nodes);
        topo.levels.resize(nodes);
        topo.base_offsets.reserve(nodes + 1);
        topo.base_offsets.push_back(0);
        const auto row_bytes = vectors.size(1) * tensor::element_size(vectors.dtype());
        const bool bfloat = vectors.dtype() == tensor::DType::BF16;
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto record = source.take(record_bytes);
            Reader labels(record.subspan(label_offset));
            const auto label = labels.integer(8);
            invalid(label >= nodes, "hnswlib label is outside the supplied vector table");
            topo.labels[node] = static_cast<std::uint32_t>(label);
            Reader neighbors(record.first(vector_offset));
            const auto count = neighbors.integer(4);
            invalid(count > base_capacity, "invalid or deleted hnswlib base list");
            for (std::size_t edge = 0; edge < count; ++edge)
                topo.base_edges.push_back(static_cast<std::uint32_t>(neighbors.integer(4)));
            if (topo.base_edges.size() > std::numeric_limits<std::uint32_t>::max())
                throw Failure({ErrorCode::UNSUPPORTED, "imported HNSW graph exceeds uint32 offset capacity"});
            topo.base_offsets.push_back(static_cast<std::uint32_t>(topo.base_edges.size()));
            Reader values(record.subspan(vector_offset, 4 * vectors.size(1)));
            const auto* row = table.data() + label * row_bytes;
            for (std::size_t column = 0; column < vectors.size(1); ++column) {
                const auto actual = static_cast<std::uint32_t>(values.integer(4));
                const auto expected = ynn::hnsw_weight_element_f32(row, column, bfloat);
                invalid(!std::isfinite(expected) || actual != std::bit_cast<std::uint32_t>(expected),
                        "hnswlib vectors differ from the original unnormalized table");
            }
        }
        topo.upper.resize(maximum_level);
        for (auto& layer : topo.upper) layer.offsets.push_back(0);
        const auto upper_record_bytes = 4 + 4 * m;
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto length = source.integer(4);
            invalid(length % upper_record_bytes || length > source.remaining() ||
                        length / upper_record_bytes > maximum_level,
                    "invalid hnswlib upper-list length");
            const auto levels = length / upper_record_bytes;
            topo.levels[node] = static_cast<std::uint8_t>(levels);
            for (std::size_t level = 0; level < levels; ++level) {
                Reader neighbors(source.take(upper_record_bytes));
                const auto count = neighbors.integer(4);
                invalid(count > m, "invalid hnswlib upper-neighbor count");
                auto& layer = topo.upper[level];
                layer.nodes.push_back(static_cast<std::uint32_t>(node));
                for (std::size_t edge = 0; edge < count; ++edge)
                    layer.edges.push_back(static_cast<std::uint32_t>(neighbors.integer(4)));
                if (layer.edges.size() > std::numeric_limits<std::uint32_t>::max())
                    throw Failure({ErrorCode::UNSUPPORTED, "imported HNSW layer exceeds uint32 offset capacity"});
                layer.offsets.push_back(static_cast<std::uint32_t>(layer.edges.size()));
            }
        }
        invalid(source.remaining(), "unexpected hnswlib trailing data");
        validate_serialized(graph);
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
        const auto table = require(std::as_const(vectors).host_bytes());
        const bool bfloat = vectors.dtype() == tensor::DType::BF16;
        SerializedGraph graph;
        graph.weight_type = bfloat ? 1U : 2U;
        graph.weights_crc32 = hnsw_crc32(table);
        graph.candidates = options.candidates;
        graph.ef_search = options.ef_search;
        std::mutex mutex;
        std::optional<Error> failure;
        const ynn::hnsw_build_options build_opts{
            .m = options.m,
            .ef_construction = options.ef_construction,
            .seed = options.seed,
        };
        const auto status =
            ynn::build_hnsw_graph(table.data(), bfloat, vectors.size(0), vectors.size(1), build_opts, &graph.topology,
                                  std::move(progress), wrap_parallel_for(parallel_for, mutex, failure));
        if (status != ynn_status_success)
            return std::unexpected(status_error(status, failure, "invalid HNSW vectors or build/search parameters",
                                                "nonfinite HNSW build distance"));
        validate_serialized(graph);
        return HnswIndex(std::make_unique<State>(std::move(graph), std::move(vectors)));
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}

auto HnswIndex::search(std::span<const float> query, HnswWorkspace& workspace, std::size_t candidates, std::size_t ef,
                       HnswParallelFor parallel_for) const -> Result<std::span<const HnswNeighbor>> {
    const auto& state = *state_;
    if (!candidates) candidates = state.graph.candidates;
    if (!ef) ef = state.graph.ef_search;
    std::mutex mutex;
    std::optional<Error> failure;
    const HnswNeighbor* neighbors = nullptr;
    std::size_t count = 0;
    const auto status = ynn::search_hnsw(state.search_index, query.data(), query.size(), candidates, ef, &workspace,
                                         &neighbors, &count, wrap_parallel_for(parallel_for, mutex, failure));
    if (status != ynn_status_success)
        return std::unexpected(
            status_error(status, failure, "invalid HNSW query or search parameters", "nonfinite HNSW query distance"));
    return std::span<const HnswNeighbor>(neighbors, count);
}

} // namespace kidi::search
