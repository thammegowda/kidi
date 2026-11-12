#include "kidi/search/hnsw.h"
#include "kidi/layers/linear.h"
#include "kidi/ops/context.h"
#include "kidi/runtime/ynn/graph.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
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
        const auto loaded = kidi::search::HnswIndex::load(path, vectors);
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

auto saving(const std::filesystem::path& root, const kidi::search::HnswIndex& index) -> void {
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

auto check_ground_truth_graph(std::span<const std::byte> data) -> void {
    check(integer(data, 16, 4) == 8 && integer(data, 20, 4) == 4 && integer(data, 24, 4) == 4 &&
              integer(data, 28, 4) == 2 && integer(data, 32, 4) == 3 && integer(data, 36, 4) == 5 &&
              integer(data, 44, 4) == 1332379836U && integer(data, 48, 4) == 828371281U &&
              integer(data, 52, 4) == 1310553304U,
          "8x4 ground-truth HNSW header metadata or checksums changed");
    const auto base = static_cast<std::size_t>(integer(data, 64, 8));
    const auto upper = static_cast<std::size_t>(integer(data, 72, 8));
    const std::array<std::uint32_t, 9> expected_offsets{0, 2, 6, 8, 12, 14, 18, 20, 23};
    for (std::size_t node = 0; node <= 8; ++node)
        check(integer(data, base + 4 * node, 4) == expected_offsets[node], "8x4 ground-truth base offsets changed");
    const std::array<std::uint32_t, 23> expected_base{
        1, 5, 3, 5, 0, 4, 7, 5, 1, 7, 5, 6, 1, 3, 0, 2, 3, 6, 3, 5, 3, 5, 2,
    };
    const auto edges = base + 4 * 9;
    for (std::size_t idx = 0; idx < expected_base.size(); ++idx)
        check(integer(data, edges + 3 * idx, 3) == expected_base[idx], "8x4 ground-truth base edges changed");
    const std::array<std::uint8_t, 66> expected_upper{
        3, 0, 0, 0, 3, 0, 0, 2, 0, 5, 0, 0, 7, 0, 0, 5, 0, 0, 2, 0, 3, 0, 0, 7, 0, 0, 7, 0, 0, 2, 0, 3, 0,
        0, 5, 0, 0, 2, 0, 0, 0, 3, 0, 0, 1, 0, 5, 0, 0, 5, 0, 0, 1, 0, 3, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0,
    };
    check(data.size() - upper == expected_upper.size(), "8x4 ground-truth upper-layer byte length changed");
    for (std::size_t idx = 0; idx < expected_upper.size(); ++idx)
        check(std::to_integer<std::uint8_t>(data[upper + idx]) == expected_upper[idx],
              "8x4 ground-truth upper-layer topology changed");
}

auto brute_force_checks(const std::filesystem::path& root, bool bfloat, std::size_t width) -> void {
    using namespace kidi;
    constexpr std::size_t rows = 320, candidates = 8;
    std::vector<float> values;
    const auto vectors = vector_table(rows, width, bfloat, values);
    auto index = require(search::HnswIndex::build(
        vectors, {.m = 16, .ef_construction = 256, .seed = 42, .candidates = candidates, .ef_search = rows}));
    check(index.size() == rows && index.width() == width && index.candidates() == candidates &&
              index.ef_search() == rows && index.runtime_bytes() > rows * sizeof(float),
          "incorrect index metadata");
    const auto path = root / (std::string(bfloat ? "bf16-" : "fp32-") + std::to_string(width) + ".hnsw");
    require(index.save(path));
    const auto serialized = bytes(path);
    const auto expected_size = width == 64 ? std::size_t{26256} : std::size_t{26244};
    const auto expected_crc = width == 64 ? 0x04C11AC2U : 0x769AA146U;
    check(serialized.size() == expected_size && integer(serialized, 24, 4) == 32 && integer(serialized, 32, 4) == 2 &&
              integer(serialized, 36, 4) == 183 && integer(serialized, 48, 4) == expected_crc,
          "320-row ground-truth HNSW topology or payload checksum changed");
    const auto base = static_cast<std::size_t>(integer(serialized, 64, 8));
    const auto edges = base + 4 * (rows + 1);
    const std::array<std::uint32_t, 8> expected_first8 =
        width == 64 ? std::array<std::uint32_t, 8>{129, 248, 213, 192, 155, 295, 282, 73}
                    : std::array<std::uint32_t, 8>{229, 319, 89, 284, 255, 55, 127, 6};
    for (std::size_t idx = 0; idx < expected_first8.size(); ++idx)
        check(integer(serialized, edges + 3 * idx, 3) == expected_first8[idx],
              "320-row ground-truth node-0 neighbor list changed");
    runtime::ynn::set_thread_count(4);
    require(runtime::ynn::reserve_thread_pool(4));
    std::size_t parallel_calls = 0;
    auto threaded = require(search::HnswIndex::build(
        vectors, {.m = 16, .ef_construction = 256, .seed = 42, .candidates = candidates, .ef_search = rows}, {},
        [&](std::size_t count, std::function<void(std::size_t)> body) {
            ++parallel_calls;
            return runtime::ynn::parallel_for(count, std::move(body));
        }));
    const auto parallel_path = path.string() + ".parallel";
    require(threaded.save(parallel_path));
    check(parallel_calls > 0 && bytes(parallel_path) == serialized,
          "threaded construction changed graph order or values");
    const auto failed_executor = search::HnswIndex::build(
        vectors, {.m = 16, .ef_construction = 256, .seed = 42, .candidates = candidates, .ef_search = rows}, {},
        [](std::size_t, std::function<void(std::size_t)>) -> Result<void> {
            return std::unexpected(Error{ErrorCode::RUNTIME, "test executor failure"});
        });
    check(!failed_executor && failed_executor.error().message == "test executor failure",
          "executor errors were not propagated");
    const auto readonly =
        require(tensor::Tensor::from_blob({rows, static_cast<std::int64_t>(width)}, vectors.dtype(),
                                          require(vectors.host_bytes()), std::make_shared<tensor::Tensor>(vectors)));
    auto loaded = require(search::HnswIndex::load(path, readonly));
    auto moved = std::move(loaded);
    search::HnswWorkspace workspace;
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
    auto imported = require(search::HnswIndex::from_hnswlib(source_path, vectors, 3, 8));
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
    search::HnswWorkspace workspace;
    const std::array<float, 4> query{1, 0, 0, 0};
    auto best = std::uint32_t{0};
    for (std::size_t row = 1; row < rows; ++row)
        if (values[row * width] > values[best * width]) best = static_cast<std::uint32_t>(row);
    check(require(imported.search(query, workspace)).front().id == best, "import changed raw inner-product ranking");
    integer(source, 96 + 20, 4, std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN()));
    save(source_path, source);
    check(!search::HnswIndex::from_hnswlib(source_path, vectors, 3, 8), "import accepted altered vectors");
    save(source_path, std::span(source).first(90));
    check(!search::HnswIndex::from_hnswlib(source_path, vectors, 3, 8), "import accepted a truncated source index");
}

auto quantized_linear_checks(const std::filesystem::path& root) -> void {
    using namespace kidi;
    constexpr std::size_t rows = 16, width = 128;
    std::vector<float> values(rows * width, 0.F);
    for (std::size_t r = 0; r < rows; ++r) values[r * width + r * 4] = 8.F;
    const auto vectors = require(tensor::Tensor::from_host({rows, width}, std::span<const float>(values)));
    auto index = require(search::HnswIndex::build(
        vectors, {.m = 4, .ef_construction = 16, .seed = 42, .candidates = 4, .ef_search = 16}));
    const auto path = root / "quantized-linear.hnsw";
    require(index.save(path));

    ops::Context context(tensor::Device::cpu());
    for (const std::int32_t bits : {8, 4, 2}) {
        const std::int32_t group_size = bits == 4 ? 32 : static_cast<std::int32_t>(width);
        const std::size_t B = width / static_cast<std::size_t>(group_size);
        const std::size_t row_bytes = width / (8 / static_cast<std::size_t>(bits));
        std::vector<std::uint8_t> packed(rows * row_bytes, 0);
        for (std::size_t r = 0; r < rows; ++r) {
            const std::size_t col = r * 4;
            if (bits == 8) {
                packed[r * row_bytes + col] = static_cast<std::uint8_t>(64);
            } else if (bits == 4) {
                packed[r * row_bytes + col / 2] = static_cast<std::uint8_t>(6);
            } else {
                packed[r * row_bytes + col / 4] = static_cast<std::uint8_t>(1);
            }
        }
        std::vector<float> scales(rows * B, 1.F);
        const ModuleScope host(tensor::Device::cpu());
        layers::Linear linear(width, rows, true, false, bits, false, group_size);
        StateDict state;
        state.emplace("weight",
                      require(tensor::Tensor::from_bytes({rows, static_cast<std::int64_t>(row_bytes)},
                                                         tensor::DType::U8, std::as_bytes(std::span(packed)))));
        state.emplace("weight_scale", require(tensor::Tensor::from_host({rows, static_cast<std::int64_t>(B)},
                                                                        std::span<const float>(scales))));
        const float zero_f = 0.F;
        state.emplace("input_activation_scale",
                      require(tensor::Tensor::from_host({}, std::span<const float>(&zero_f, 1))));
        state.emplace("output_activation_scale",
                      require(tensor::Tensor::from_host({}, std::span<const float>(&zero_f, 1))));
        require(linear->load_state_dict(state));
        require(linear->bind_hnsw(path));
        check(linear->has_hnsw(), "LinearImpl::bind_hnsw did not enable HNSW");
        std::vector<float> q(width, 0.F);
        q[7 * 4] = 4.F;
        const auto q_tensor = require(tensor::Tensor::from_host({1, 1, width}, std::span<const float>(q)));
        const auto selected = linear->select(context, q_tensor);
        check(require(selected.data<std::int32_t>())[0] == 7, "quantized LinearImpl::select returned wrong token");
        q.resize(2 * width, 0.F);
        q[width + 3 * 4] = 4.F;
        const auto batch = require(tensor::Tensor::from_host({1, 2, width}, std::span<const float>(q)));
        const auto chosen = linear->select(context, batch);
        const auto ids = require(chosen.data<std::int32_t>());
        check(ids.size() == 2 && ids[0] == 7 && ids[1] == 3, "batched HNSW selection mixed query workspaces");
        linear->release_workspaces();
        check(require(linear->select(context, q_tensor).data<std::int32_t>())[0] == 7,
              "HNSW workspaces did not rebuild after release");
        bool rejected = false;
        try {
            linear->select(context, require(tensor::Tensor::zeros({1, width / 2}, tensor::DType::F32)));
        } catch (const ops::Failure&) {
            rejected = true;
        }
        check(rejected, "HNSW linear accepted a query with the wrong width");
    }
}

auto quantized_distance_checks(const std::filesystem::path& root, std::int32_t bits, std::size_t group,
                               std::size_t width, bool affine, bool half) -> void {
    using namespace kidi;
    constexpr std::size_t rows = 32;
    const auto blocks = width / group;
    const auto values_per_byte = static_cast<std::size_t>(8 / bits);
    std::mt19937 random(17);
    std::vector<std::uint8_t> packed(rows * width / values_per_byte);
    std::vector<float> decoded(rows * width), scales(rows * blocks), zeros(rows * blocks);
    std::vector<std::uint16_t> half_scales(rows * blocks), half_zeros(rows * blocks);
    const std::array<float, 4> scale_values{0.125F, 0.25F, 0.5F, 1.F};
    const std::array<std::uint16_t, 4> encoded_scales{0x3000, 0x3400, 0x3800, 0x3c00};
    const std::array<float, 4> zero_values{-0.5F, 0.F, 0.25F, 0.5F};
    const std::array<std::uint16_t, 4> encoded_zeros{0xb800, 0, 0x3400, 0x3800};
    for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = scale_values[i % 4];
        half_scales[i] = encoded_scales[i % 4];
        zeros[i] = affine ? zero_values[(i / 4) % 4] : 0.F;
        half_zeros[i] = encoded_zeros[(i / 4) % 4];
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            const auto encoded = random() % (1U << bits);
            const auto value = static_cast<std::int32_t>(encoded) - (encoded >= (1U << (bits - 1)) ? (1 << bits) : 0);
            packed[row * width / values_per_byte + column / values_per_byte] |=
                static_cast<std::uint8_t>(encoded << (bits * (column % values_per_byte)));
            decoded[row * width + column] =
                value * scales[row * blocks + column / group] + zeros[row * blocks + column / group];
        }
    }
    const auto vectors =
        require(tensor::Tensor::from_host({rows, static_cast<std::int64_t>(width)}, std::span<const float>(decoded)));
    auto topology = require(search::HnswIndex::build(
        vectors, {.m = 32, .ef_construction = 64, .seed = 42, .candidates = rows, .ef_search = rows}));
    const auto path = root / ("packed-" + std::to_string(bits) + "-" + std::to_string(group) + "-" +
                              std::to_string(width) + "-" + std::to_string(affine) + "-" + std::to_string(half));
    require(topology.save(path));
    const auto weights = require(tensor::Tensor::from_bytes({rows, static_cast<std::int64_t>(width / values_per_byte)},
                                                            tensor::DType::U8, std::as_bytes(std::span(packed))));
    const auto metadata = [&](const std::vector<float>& values, const std::vector<std::uint16_t>& encoded) {
        const std::vector<std::int64_t> shape{rows, static_cast<std::int64_t>(blocks)};
        return half ? require(tensor::Tensor::from_bytes(shape, tensor::DType::F16, std::as_bytes(std::span(encoded))))
                    : require(tensor::Tensor::from_host(shape, std::span<const float>(values)));
    };
    const auto act_scale = bits == 2 ? 0.125F : 0.F;
    auto index = require(search::HnswIndex::load_quantized(path, weights, metadata(scales, half_scales),
                                                           affine ? metadata(zeros, half_zeros) : tensor::Tensor{},
                                                           bits, static_cast<std::int32_t>(group), act_scale));
    if (bits == 8 && !half) {
        for (const auto invalid_scale : {-1.F, std::numeric_limits<float>::quiet_NaN()})
            check(!search::HnswIndex::load_quantized(path, weights, metadata(scales, half_scales), {}, bits,
                                                     static_cast<std::int32_t>(group), invalid_scale),
                  "packed HNSW accepted an invalid activation scale");
        check(!search::HnswIndex::load_quantized(path, weights, metadata(scales, half_scales),
                                                 metadata(zeros, half_zeros), bits, static_cast<std::int32_t>(group)),
              "INT8 HNSW silently ignored affine offsets");
    }
    search::HnswWorkspace workspace;
    std::vector<float> query(width), rounded(width);
    for (std::size_t attempt = 0; attempt < 5; ++attempt) {
        for (auto& value : query) value = (static_cast<float>(random() % 101) - 50.F) / 16.F;
        for (std::size_t b = 0; b < blocks; ++b) {
            if (bits == 8 && act_scale <= 0.F) {
                float rmin = 0.F, rmax = 0.F;
                for (std::size_t c = 0; c < group; ++c) {
                    rmin = std::min(rmin, query[b * group + c]);
                    rmax = std::max(rmax, query[b * group + c]);
                }
                const float rdiff = rmax - rmin;
                const float inverse = rdiff > 0.F ? 255.F / rdiff : 0.F;
                const float scale = rdiff > 0.F ? rdiff / 255.F : 1.F;
                const float min_error = -128.F + rmin * inverse;
                const float max_error = 127.F + rmax * inverse;
                const float fzp =
                    min_error + max_error > 0.F ? -128.F - rmin * inverse : 127.F - rmax * inverse;
                const auto zp = rdiff > 0.F
                                    ? static_cast<std::int32_t>(std::rint(std::clamp(fzp, -128.F, 127.F)))
                                    : -128;
                for (std::size_t c = 0; c < group; ++c) {
                    const auto q = std::clamp<std::int32_t>(
                        static_cast<std::int32_t>(std::round(query[b * group + c] * inverse)) + zp, -128, 127);
                    rounded[b * group + c] = static_cast<float>(q - zp) * scale;
                }
                continue;
            }
            float maximum = 0.F;
            for (std::size_t c = 0; c < group; ++c) maximum = std::max(maximum, std::abs(query[b * group + c]));
            const auto scale = act_scale > 0.F ? act_scale : maximum / 127.F;
            const auto inverse = act_scale > 0.F ? 1.F / act_scale : 127.F / maximum;
            for (std::size_t c = 0; c < group; ++c)
                rounded[b * group + c] = std::clamp(std::nearbyint(query[b * group + c] * inverse),
                                                    act_scale > 0.F ? -128.F : -127.F, 127.F) *
                                         scale;
        }
        const auto found = require(index.search(query, workspace));
        check(found.size() == rows, "exhaustive packed search did not return every row");
        for (const auto& neighbor : found) {
            double dot = 0.;
            for (std::size_t c = 0; c < width; ++c)
                dot += static_cast<double>(rounded[c]) * decoded[neighbor.id * width + c];
            const auto expected = 1.F - static_cast<float>(dot);
            check(std::abs(neighbor.distance - expected) < 0.00005F * (1.F + std::abs(expected)),
                  "packed search differs from scalar reference scores");
        }
    }
}

#if !defined(_WIN32)
auto search_settings_checks(const std::filesystem::path& root) -> void {
    using namespace kidi;
    struct Environment {
        std::optional<std::string> ef;
        Environment() {
            if (const auto* value = std::getenv("KIDI_HNSW_EF")) ef = value;
            unsetenv("KIDI_HNSW_EF");
        }
        ~Environment() {
            if (ef)
                setenv("KIDI_HNSW_EF", ef->c_str(), 1);
            else
                unsetenv("KIDI_HNSW_EF");
        }
    } restore;
    std::vector<float> values;
    const auto vectors = vector_table(16, 64, false, values);
    auto topology =
        require(search::HnswIndex::build(vectors, {.m = 4, .ef_construction = 16, .candidates = 4, .ef_search = 16}));
    const auto path = root / "search-settings.hnsw";
    require(topology.save(path));
    const ModuleScope host(tensor::DType::F32, false, tensor::Device::cpu());
    layers::Linear linear(64, 16, true, false);
    require(linear->set_state({{"weight", vectors}}));
    require(linear->bind_hnsw(path));
    ops::Context context(tensor::Device::cpu());
    std::vector<float> query(64, 1.F);
    const auto input = require(tensor::Tensor::from_host({1, 64}, std::span<const float>(query)));
    const auto expected = require(linear->select(context, input).data<std::int32_t>())[0];
    for (const auto* value : {"", "0", "-1", "abc", "8junk", "17", "18446744073709551616"}) {
        setenv("KIDI_HNSW_EF", value, 1);
        const auto bound = linear->bind_hnsw(path);
        check(!bound && bound.error().code == ErrorCode::INVALID_ARGUMENT,
              "invalid HNSW setting was silently accepted");
        check(require(linear->select(context, input).data<std::int32_t>())[0] == expected,
              "failed HNSW binding changed the previous search configuration");
    }
    setenv("KIDI_HNSW_EF", "3", 1);
    check(!linear->bind_hnsw(path), "search effort smaller than candidate count was accepted");
    setenv("KIDI_HNSW_EF", "16", 1);
    require(linear->bind_hnsw(path));
    setenv("KIDI_HNSW_EF", "invalid-after-binding", 1);
    check(require(linear->select(context, input).data<std::int32_t>())[0] == expected,
          "HNSW hot path reread its environment settings");
}
#endif
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
        const search::HnswBuildOptions options{
            .m = 2, .ef_construction = 8, .seed = 42, .candidates = 3, .ef_search = 8};
        auto first = require(search::HnswIndex::build(vectors, options, [&](std::size_t completed, std::size_t total) {
            if (completed > total || completed <= progress) throw std::runtime_error("invalid HNSW progress");
            progress = completed;
        }));
        auto second = require(search::HnswIndex::build(vectors, options));
        check(progress == vectors.size(0), "HNSW build progress is incomplete");
        require(first.save(first_path));
        require(second.save(second_path));
        if (bytes(first_path) != bytes(second_path)) throw std::runtime_error("deterministic HNSW files differ");
        check_ground_truth_graph(bytes(first_path));
        auto loaded = require(search::HnswIndex::load(first_path, vectors));
        require(loaded.save(root / "roundtrip.hnsw"));
        check(bytes(root / "roundtrip.hnsw") == bytes(first_path), "graph order changed during round trip");
        search::HnswWorkspace workspace;
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
        if (search::HnswIndex::load(second_path, vectors)) throw std::runtime_error("corrupt HNSW index was accepted");
        const auto wrong = require(tensor::Tensor::zeros({8, 4}, tensor::DType::F32));
        if (search::HnswIndex::load(first_path, wrong))
            throw std::runtime_error("mismatched HNSW vectors were accepted");
        saving(root, first);
        for (bool bfloat : {false, true})
            for (const std::size_t width : {64, 67}) brute_force_checks(root, bfloat, width);
        for (bool bfloat : {false, true}) import_checks(root, bfloat);
        quantized_linear_checks(root);
        for (const bool half : {false, true}) {
            quantized_distance_checks(root, 8, 1536, 1536, false, half);
            quantized_distance_checks(root, 2, 1536, 1536, false, half);
            quantized_distance_checks(root, 2, 32, 128, false, half);
            for (const bool affine : {false, true}) {
                quantized_distance_checks(root, 4, 32, 384, affine, half);
                quantized_distance_checks(root, 4, 128, 1536, affine, half);
                quantized_distance_checks(root, 4, 32, 96, affine, half);
                quantized_distance_checks(root, 4, 64, 384, affine, half);
            }
        }
#if !defined(_WIN32)
        search_settings_checks(root);
#endif
        const std::array<float, 32> zeros{};
        const auto tied_vectors = require(tensor::Tensor::from_host({8, 4}, std::span<const float>(zeros)));
        auto tied = require(search::HnswIndex::build(tied_vectors, options));
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
