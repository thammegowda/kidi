#include "kidi/core/hnsw.h"
#include "kidi/ops/context.h"
#include "kidi/runtime/ynn/graph.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

namespace {
using kidi::ops::require;

auto check(bool valid, const char* message) -> void {
    if (!valid) throw std::runtime_error(message);
}

auto bytes(const std::filesystem::path& path) -> std::vector<std::byte> {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open HNSW test file");
    const auto size = input.tellg();
    std::vector<std::byte> result(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()));
    if (!input) throw std::runtime_error("cannot read HNSW test file");
    return result;
}

auto save(const std::filesystem::path& path, std::span<const std::byte> data) -> void {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!output) throw std::runtime_error("cannot write HNSW test file");
}

auto integer(std::span<const std::byte> data, std::size_t offset, std::size_t count) -> std::uint64_t {
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < count; ++index)
        result |= std::to_integer<std::uint64_t>(data[offset + index]) << (8 * index);
    return result;
}

auto integer(std::span<std::byte> data, std::size_t offset, std::size_t count, std::uint64_t value) -> void {
    for (std::size_t index = 0; index < count; ++index)
        data[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255U);
}

auto crc32(std::span<const std::byte> data) -> std::uint32_t {
    std::uint32_t result = ~0U;
    for (auto value : data) {
        result ^= std::to_integer<std::uint32_t>(value);
        for (int bit = 0; bit < 8; ++bit) result = (result >> 1) ^ (0xEDB88320U & (0U - (result & 1U)));
    }
    return ~result;
}

auto repair_checksums(std::span<std::byte> data) -> void {
    integer(data, 48, 4, crc32(data.subspan(128)));
    integer(data, 52, 4, 0);
    integer(data, 52, 4, crc32(data.first(128)));
}

auto malformed_indexes(const std::filesystem::path& root, const std::filesystem::path& valid,
                       const kidi::tensor::Tensor& vectors) -> void {
    const auto original = bytes(valid);
    const auto path = root / "malformed.hnsw";
    const auto reject = [&](std::span<const std::byte> data) {
        save(path, data);
        const auto loaded = kidi::core::HnswIndex::load(path, vectors);
        check(!loaded && loaded.error().code == kidi::ErrorCode::INVALID_MANIFEST,
              "malformed index was not rejected as an invalid manifest");
    };
    for (const auto count : {std::size_t{0}, std::size_t{127}, original.size() - 1})
        reject(std::span(original).first(count));
    auto corrupted = original;
    corrupted.back() ^= std::byte{1};
    reject(corrupted);
    corrupted = original;
    corrupted.push_back(std::byte{0});
    reject(corrupted);
    const auto base = static_cast<std::size_t>(integer(original, 64, 8));
    const auto upper = static_cast<std::size_t>(integer(original, 72, 8));
    const auto edges = base + 4 * (vectors.size(0) + 1);
    check(integer(original, 32, 4) > 0, "fixture did not exercise upper levels");
    const std::array<std::array<std::uint64_t, 3>, 17> mutations{{
        {0, 1, 0},
        {8, 4, 99},
        {16, 4, 0},
        {16, 4, (1U << 24) + 1U},
        {36, 4, vectors.size(0)},
        {40, 4, 3},
        {56, 8, 129},
        {64, 8, base + 1},
        {72, 8, base},
        {88, 4, 0},
        {92, 4, 1},
        {128 + 4, 4, 0},
        {base, 4, 1},
        {base + 4, 4, std::numeric_limits<std::uint32_t>::max()},
        {edges, 3, 0xFFFFFFU},
        {upper + 4, 3, vectors.size(0)},
        {upper + 7, 2, 65535},
    }};
    for (const auto& mutation : mutations) {
        auto data = original;
        integer(data, static_cast<std::size_t>(mutation[0]), static_cast<std::size_t>(mutation[1]), mutation[2]);
        repair_checksums(data);
        reject(data);
    }
}

auto saving(const std::filesystem::path& root, const kidi::core::HnswIndex& index) -> void {
    const auto path = root / "saved.hnsw";
    require(index.save(path));
    const auto original = bytes(path);
    check(!index.save(path), "saving overwrote an existing index");
    check(bytes(path) == original, "failed save changed the existing index");
    check(!index.save({}), "empty save path was accepted");
    check(!index.save(root / "absent" / "index"), "save with a missing parent succeeded");
    check(!index.save(root), "saving replaced a directory");
    const auto concurrent = root / "concurrent.hnsw";
    auto first = std::async(std::launch::async, [&] { return index.save(concurrent); });
    auto second = std::async(std::launch::async, [&] { return index.save(concurrent); });
    const auto left = first.get(), right = second.get();
    check(left.has_value() != right.has_value(), "concurrent saves did not publish exactly once");
    check(bytes(concurrent) == original, "concurrent save published an incomplete file");
#if !defined(_WIN32)
    const auto dangling = root / "dangling";
    std::filesystem::create_symlink(root / "missing-target", dangling);
    check(!index.save(dangling) && std::filesystem::is_symlink(dangling), "saving replaced a dangling symbolic link");
#endif
    for (const auto& entry : std::filesystem::directory_iterator(root))
        check(!entry.path().filename().string().starts_with(".kidi-hnsw-"), "save left a staging directory behind");
}

auto vector_table(std::size_t rows, std::size_t width, bool bfloat, std::vector<float>& values)
    -> kidi::tensor::Tensor {
    values.resize(rows * width);
    std::mt19937 random(17);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < width; ++column)
            values[row * width + column] =
                (static_cast<float>(random() % 33) - 16.F) / 16.F * (1.F + static_cast<float>(row % 5) / 4.F);
    if (!bfloat)
        return require(kidi::tensor::Tensor::from_host(
            {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(width)}, std::span<const float>(values)));
    std::vector<std::uint16_t> encoded(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        encoded[index] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(values[index]) >> 16);
        values[index] = std::bit_cast<float>(static_cast<std::uint32_t>(encoded[index]) << 16);
    }
    return require(kidi::tensor::Tensor::from_bytes({static_cast<std::int64_t>(rows), static_cast<std::int64_t>(width)},
                                                    kidi::tensor::DType::BF16, std::as_bytes(std::span(encoded))));
}

auto brute_force_checks(const std::filesystem::path& root, bool bfloat, std::size_t width) -> void {
    using namespace kidi;
    constexpr std::size_t rows = 256, candidates = 8;
    std::vector<float> values;
    const auto vectors = vector_table(rows, width, bfloat, values);
    auto index = require(core::HnswIndex::build(
        vectors, {.m = 16, .ef_construction = 128, .seed = 42, .candidates = candidates, .ef_search = rows}));
    check(index.size() == rows && index.width() == width && index.candidates() == candidates &&
              index.ef_search() == rows && index.runtime_bytes() > rows * sizeof(float),
          "incorrect index metadata");
    const auto path = root / (std::string(bfloat ? "bf16-" : "fp32-") + std::to_string(width) + ".hnsw");
    require(index.save(path));
    runtime::ynn::set_thread_count(4);
    require(runtime::ynn::reserve_thread_pool(4));
    std::size_t parallel_calls = 0;
    auto threaded = require(core::HnswIndex::build(
        vectors, {.m = 16, .ef_construction = 128, .seed = 42, .candidates = candidates, .ef_search = rows}, {},
        [&](std::size_t count, std::function<void(std::size_t)> body) {
            ++parallel_calls;
            return runtime::ynn::parallel_for(count, std::move(body));
        }));
    const auto parallel_path = path.string() + ".parallel";
    require(threaded.save(parallel_path));
    check(parallel_calls > 0 && bytes(parallel_path) == bytes(path),
          "threaded construction changed graph order or values");
    const auto failed_executor = core::HnswIndex::build(
        vectors, {.m = 16, .ef_construction = 128, .seed = 42, .candidates = candidates, .ef_search = rows}, {},
        [](std::size_t, std::function<void(std::size_t)>) -> Result<void> {
            return std::unexpected(Error{ErrorCode::RUNTIME, "test executor failure"});
        });
    check(!failed_executor && failed_executor.error().message == "test executor failure",
          "executor errors were not propagated");
    const auto readonly =
        require(tensor::Tensor::from_blob({rows, static_cast<std::int64_t>(width)}, vectors.dtype(),
                                          require(vectors.host_bytes()), std::make_shared<tensor::Tensor>(vectors)));
    auto loaded = require(core::HnswIndex::load(path, readonly));
    auto moved = std::move(loaded);
    core::HnswWorkspace workspace;
    std::mt19937 random(23);
    std::vector<float> query(width);
    for (std::size_t attempt = 0; attempt < 20; ++attempt) {
        for (auto& value : query) value = (static_cast<float>(random() % 10001) - 5000.F) / 4096.F;
        std::vector<std::pair<float, std::uint32_t>> expected;
        for (std::size_t row = 0; row < rows; ++row) {
            double dot = 0;
            for (std::size_t column = 0; column < width; ++column)
                dot += static_cast<double>(query[column]) * values[row * width + column];
            expected.emplace_back(1.F - static_cast<float>(dot), static_cast<std::uint32_t>(row));
        }
        std::ranges::sort(expected);
        const auto actual = require(moved.search(query, workspace));
        check(actual.size() == candidates, "search did not return the configured candidate subset");
        std::vector<std::uint32_t> ids;
        for (std::size_t rank = 0; rank < candidates; ++rank) {
            ids.push_back(actual[rank].id);
            check(std::abs(actual[rank].distance - expected[rank].first) < 1e-4F,
                  "search disagrees with brute-force inner products");
            if (rank == 0 || expected[rank].first != expected[rank - 1].first)
                check(actual[rank].id == expected[rank].second, "search returned the wrong brute-force neighbor");
        }
        std::ranges::sort(ids);
        check(std::ranges::adjacent_find(ids) == ids.end(), "search returned repeated row IDs");
        const auto before_wrap = actual.front().id;
        workspace.epoch = std::numeric_limits<std::uint32_t>::max();
        check(require(moved.search(query, workspace)).front().id == before_wrap, "visited epoch wrap changed results");
    }
    query[0] = std::numeric_limits<float>::quiet_NaN();
    check(!moved.search(query, workspace), "nonfinite query was accepted");
    check(!moved.search({}, workspace), "wrong query dimension was accepted");
    query[0] = 0;
    check(!moved.search(query, workspace, 4, 3), "search effort smaller than candidate count was accepted");
    if (bfloat && width == 67) malformed_indexes(root, path, vectors);
}

auto import_checks(const std::filesystem::path& root, bool bfloat) -> void {
    using namespace kidi;
    constexpr std::size_t rows = 8, width = 4;
    std::vector<float> values;
    const auto vectors = vector_table(rows, width, bfloat, values);
    std::vector<std::byte> source;
    const auto append = [&](std::uint64_t value, std::size_t count) {
        const auto offset = source.size();
        source.resize(offset + count);
        integer(source, offset, count, value);
    };
    for (auto value : std::array<std::uint64_t, 6>{0, rows, rows, 44, 36, 20}) append(value, 8);
    append(1, 4);
    append(0, 4);
    for (auto value : {2U, 4U, 2U}) append(value, 8);
    append(std::bit_cast<std::uint64_t>(1.), 8);
    append(8, 8);
    for (std::size_t node = 0; node < rows; ++node) {
        const auto label = rows - 1 - node;
        append(2, 4);
        append((node + 1) % rows, 4);
        append((node + 2) % rows, 4);
        append(0, 8);
        for (std::size_t column = 0; column < width; ++column)
            append(std::bit_cast<std::uint32_t>(values[label * width + column]), 4);
        append(label, 8);
    }
    for (std::size_t node = 0; node < rows; ++node) {
        const auto upper = node == 0 || node == 2;
        append(upper ? 12 : 0, 4);
        if (upper) {
            append(1, 4);
            append(node == 0 ? 2 : 0, 4);
            append(0, 4);
        }
    }
    const auto source_path = root / (bfloat ? "import-bf16.bin" : "import-fp32.bin");
    save(source_path, source);
    auto imported = require(core::HnswIndex::from_hnswlib(source_path, vectors, 3, 8));
    const auto packed_path = source_path.string() + ".hnsw";
    require(imported.save(packed_path));
    const auto packed = bytes(packed_path);
    const auto base = static_cast<std::size_t>(integer(packed, 64, 8));
    const auto edges = base + 4 * (rows + 1);
    for (std::size_t node = 0; node < rows; ++node) {
        check(integer(packed, 128 + 4 * node, 4) == rows - 1 - node, "import changed internal-to-row mapping");
        check(integer(packed, edges + 6 * node, 3) == (node + 1) % rows &&
                  integer(packed, edges + 6 * node + 3, 3) == (node + 2) % rows,
              "import changed ordered live neighbors");
    }
    core::HnswWorkspace workspace;
    const std::array<float, 4> query{1, 0, 0, 0};
    auto best = std::uint32_t{0};
    for (std::size_t row = 1; row < rows; ++row)
        if (values[row * width] > values[best * width]) best = static_cast<std::uint32_t>(row);
    check(require(imported.search(query, workspace)).front().id == best, "import changed raw inner-product ranking");
    integer(source, 96 + 20, 4, std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN()));
    save(source_path, source);
    check(!core::HnswIndex::from_hnswlib(source_path, vectors, 3, 8), "import accepted altered vectors");
    save(source_path, std::span(source).first(90));
    check(!core::HnswIndex::from_hnswlib(source_path, vectors, 3, 8), "import accepted a truncated source index");
}
} // namespace

auto main() -> int {
    using namespace kidi;
    const auto root = std::filesystem::temp_directory_path() /
                      ("kidi-hnsw-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto first_path = root / "first.hnsw";
    const auto second_path = root / "second.hnsw";
    try {
        check(std::filesystem::create_directory(root), "cannot create unique test directory");
        const std::array<float, 32> values{
            1, 0, 0, 0, 10, 1, 0, 0, -2, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, -1, 1, 0, 0,
        };
        const auto vectors = require(tensor::Tensor::from_host({8, 4}, std::span<const float>(values)));
        std::size_t progress = 0;
        const core::HnswBuildOptions options{.m = 2, .ef_construction = 8, .seed = 42, .candidates = 3, .ef_search = 8};
        auto first = require(core::HnswIndex::build(vectors, options, [&](std::size_t completed, std::size_t total) {
            if (completed > total || completed <= progress) throw std::runtime_error("invalid HNSW progress");
            progress = completed;
        }));
        auto second = require(core::HnswIndex::build(vectors, options));
        check(progress == vectors.size(0), "HNSW build progress is incomplete");
        require(first.save(first_path));
        require(second.save(second_path));
        if (bytes(first_path) != bytes(second_path)) throw std::runtime_error("deterministic HNSW files differ");
        auto loaded = require(core::HnswIndex::load(first_path, vectors));
        require(loaded.save(root / "roundtrip.hnsw"));
        check(bytes(root / "roundtrip.hnsw") == bytes(first_path), "graph order changed during round trip");
        core::HnswWorkspace workspace;
        const std::array<float, 4> query{1, 0, 0, 0};
        const auto found = require(loaded.search(query, workspace));
        if (found.size() != 3 || found[0].id != 1 || found[0].distance != -9.F)
            throw std::runtime_error("native HNSW build/search returned the wrong nearest neighbor");
        auto invalid_query = query;
        invalid_query[0] = std::numeric_limits<float>::quiet_NaN();
        if (loaded.search(invalid_query, workspace) || loaded.search(query, workspace, 4, 3))
            throw std::runtime_error("invalid HNSW search was accepted");
        auto corrupt = bytes(first_path);
        corrupt.back() ^= std::byte{1};
        save(second_path, corrupt);
        if (core::HnswIndex::load(second_path, vectors)) throw std::runtime_error("corrupt HNSW index was accepted");
        const auto wrong = require(tensor::Tensor::zeros({8, 4}, tensor::DType::F32));
        if (core::HnswIndex::load(first_path, wrong)) throw std::runtime_error("mismatched HNSW vectors were accepted");
        saving(root, first);
        for (bool bfloat : {false, true})
            for (const std::size_t width : {64, 67}) brute_force_checks(root, bfloat, width);
        for (bool bfloat : {false, true}) import_checks(root, bfloat);
        const std::array<float, 32> zeros{};
        const auto tied_vectors = require(tensor::Tensor::from_host({8, 4}, std::span<const float>(zeros)));
        auto tied = require(core::HnswIndex::build(tied_vectors, options));
        for (const auto& neighbor : require(tied.search(query, workspace)))
            check(neighbor.distance == 1.F, "tied distances changed");
        std::filesystem::remove_all(root);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::error_code cleanup;
        std::filesystem::remove_all(root, cleanup);
        if (cleanup) std::cerr << "HNSW test cleanup: " << cleanup.message() << '\n';
        return 1;
    }
}
