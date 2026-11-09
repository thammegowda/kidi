#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cuvs/neighbors/hnsw.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
auto check(cuvsError_t result) -> void {
    if (result == CUVS_SUCCESS) return;
    const auto* message = cuvsGetLastErrorText();
    throw std::runtime_error(std::string("cuVS: ") + (message ? message : "unknown failure"));
}

template <typename T, cuvsError_t (*destroy)(T)>
class CuvsHandle {
public:
    explicit CuvsHandle(cuvsError_t (*create)(T*)) { check(create(&value_)); }
    ~CuvsHandle() {
        if (destroy(value_) != CUVS_SUCCESS)
            std::cerr << "kidi_index_cuda|cleanup_error="
                      << (cuvsGetLastErrorText() ? cuvsGetLastErrorText() : "unknown") << '\n';
    }
    CuvsHandle(const CuvsHandle&) = delete;
    auto operator=(const CuvsHandle&) -> CuvsHandle& = delete;
    auto get() const -> T { return value_; }

private:
    T value_{};
};

auto integer(std::string_view text) -> std::size_t {
    std::size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !value)
        throw std::invalid_argument("expected positive integer cuVS helper arguments");
    return value;
}
} // namespace

auto main(int argc, char** argv) -> int {
    try {
        if (argc != 8) {
            std::cerr << "usage: kidi-index-cuda VECTORS.f32 OUTPUT ROWS WIDTH M EF THREADS\n";
            return 2;
        }
        const auto rows = integer(argv[3]), width = integer(argv[4]), m = integer(argv[5]);
        const auto ef = integer(argv[6]), threads = integer(argv[7]);
        if (rows > (1U << 24) || m < 2 || m > 65535 || ef < m ||
            ef > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            threads > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            width > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float) / rows)
            throw std::invalid_argument("cuVS helper dimensions or parameters exceed supported bounds");
        std::vector<float> values(rows * width);
        std::ifstream input(argv[1], std::ios::binary);
        input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!input || input.peek() != std::char_traits<char>::eof() ||
            std::ranges::any_of(values, [](float value) { return !std::isfinite(value); }))
            throw std::runtime_error("invalid staged cuVS vector data");
        std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(rows), static_cast<std::int64_t>(width)};
        DLManagedTensor dataset{};
        dataset.dl_tensor.data = values.data();
        dataset.dl_tensor.device = {kDLCPU, 0};
        dataset.dl_tensor.ndim = 2;
        dataset.dl_tensor.dtype = {kDLFloat, 32, 1};
        dataset.dl_tensor.shape = shape.data();
        const CuvsHandle<cuvsResources_t, cuvsResourcesDestroy> resources(cuvsResourcesCreate);
        const CuvsHandle<cuvsHnswIndexParams_t, cuvsHnswIndexParamsDestroy> parameters(cuvsHnswIndexParamsCreate);
        const CuvsHandle<cuvsHnswIndex_t, cuvsHnswIndexDestroy> index(cuvsHnswIndexCreate);
        parameters.get()->hierarchy = CPU;
        parameters.get()->M = m;
        parameters.get()->ef_construction = static_cast<int>(ef);
        parameters.get()->num_threads = static_cast<int>(threads);
        parameters.get()->metric = InnerProduct;
        const auto started = std::chrono::steady_clock::now();
        std::cerr << "GPU graph construction: 0% | ETA unavailable (cuVS provides no incremental callback)\n";
        {
            std::mutex mutex;
            std::condition_variable_any wake;
            std::jthread heartbeat([&](std::stop_token stop) {
                std::unique_lock lock(mutex);
                while (!stop.stop_requested()) {
                    wake.wait_for(lock, stop, std::chrono::seconds(5), [] { return false; });
                    if (!stop.stop_requested())
                        std::cerr << "GPU graph construction: progress/ETA unavailable | elapsed "
                                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                                  << "s\n";
                }
            });
            check(cuvsHnswBuild(resources.get(), parameters.get(), &dataset, index.get()));
            check(cuvsStreamSync(resources.get()));
        }
        std::cerr << "GPU graph construction: 100% | ETA 0s | exporting HNSW graph\n";
        check(cuvsHnswSerialize(resources.get(), argv[2], index.get()));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kidi_index_cuda: " << error.what() << '\n';
        return 1;
    }
}
