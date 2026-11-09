#include "kidi/cli/index.h"
#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/weights.h"
#include "kidi/core/hnsw.h"
#include "kidi/model/gemma4.h"
#include "kidi/runtime/ynn/graph.h"
#if defined(KIDI_HAS_CUDA)
#include "kidi/ops/context.h"
#include "kidi/tensor/backend.h"
#include <cerrno>
#include <cmath>
#include <fstream>
#include <random>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace kidi::cli {
namespace {
#if defined(KIDI_HAS_CUDA)
class TemporaryDirectory {
public:
    TemporaryDirectory() {
        std::random_device random;
        for (int attempt = 0; attempt < 16; ++attempt) {
            const auto candidate = std::filesystem::temp_directory_path() / (".kidi-cuvs-" + std::to_string(random()));
            if (std::filesystem::create_directory(candidate)) {
                path_ = candidate;
                return;
            }
        }
        throw ops::Failure({ErrorCode::IO, "cannot create a unique cuVS export directory"});
    }
    ~TemporaryDirectory() {
        std::error_code error;
        for (const auto& file : {input(), output(), path_}) {
            std::filesystem::remove(file, error);
            if (error) std::cerr << "kidi_index_cuda|cleanup_error=" << error.message() << '\n';
        }
    }
    auto input() const -> std::filesystem::path { return path_ / "vectors.f32"; }
    auto output() const -> std::filesystem::path { return path_ / "index.bin"; }

private:
    std::filesystem::path path_;
};

auto run_helper(std::vector<std::string> arguments) -> Result<void> {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    auto status = posix_spawn_file_actions_init(&actions);
    if (status)
        return std::unexpected(
            Error{ErrorCode::IO, "cannot prepare cuVS helper: " + std::system_category().message(status)});
    struct Cleanup {
        posix_spawn_file_actions_t& actions;
        ~Cleanup() {
            const auto status = posix_spawn_file_actions_destroy(&actions);
            if (status) std::cerr << "kidi_index_cuda|cleanup_error=" << std::system_category().message(status) << '\n';
        }
    } cleanup{actions};
    status = posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
    if (status)
        return std::unexpected(
            Error{ErrorCode::IO, "cannot redirect cuVS helper output: " + std::system_category().message(status)});
    pid_t pid;
    status = posix_spawnp(&pid, argv.front(), &actions, nullptr, argv.data(), environ);
    if (status)
        return std::unexpected(
            Error{ErrorCode::IO, "cannot launch kidi-index-cuda; install it beside kidi or on PATH: " +
                                     std::system_category().message(status)});
    int result = 0;
    while (waitpid(pid, &result, 0) < 0) {
        if (errno != EINTR)
            return std::unexpected(
                Error{ErrorCode::IO, "cannot wait for cuVS helper: " + std::system_category().message(errno)});
    }
    if (!WIFEXITED(result) || WEXITSTATUS(result) != 0)
        return std::unexpected(Error{ErrorCode::RUNTIME, "cuVS helper failed; see its diagnostic output above"});
    return {};
}

auto cuda_index_available() -> Result<bool> {
    auto backend = tensor::BackendRegistry::instance().backend(tensor::Device::cuda());
    if (!backend) return std::unexpected(std::move(backend.error()));
    return (*backend)->is_available(tensor::Device::cuda());
}

auto build_cuda_index(const tensor::Tensor& vectors, core::HnswBuildOptions options, std::size_t threads)
    -> Result<core::HnswIndex> {
    try {
        if (!vectors.defined() || !vectors.is_host_accessible() || !vectors.is_contiguous() ||
            vectors.dimensions() != 2 || !vectors.size(0) || vectors.size(0) > (1U << 24) ||
            (vectors.dtype() != tensor::DType::BF16 && vectors.dtype() != tensor::DType::F32) || options.m < 2 ||
            options.m > 65535 || options.ef_construction < options.m || !options.candidates ||
            options.candidates > vectors.size(0) || options.ef_search < options.candidates ||
            options.ef_search > vectors.size(0))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid CUDA HNSW construction parameters"});
        const TemporaryDirectory temporary;
        {
            ops::Context context(tensor::Device::cpu());
            const auto expanded = context.cast(vectors, tensor::DType::F32);
            context.synchronize();
            const auto values = ops::require(expanded.data<float>());
            if (std::ranges::any_of(values, [](float value) { return !std::isfinite(value); }))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "nonfinite CUDA index vectors"});
            std::ofstream output(temporary.input(), std::ios::binary);
            output.write(reinterpret_cast<const char*>(values.data()),
                         static_cast<std::streamsize>(values.size_bytes()));
            output.flush();
            if (!output) throw ops::Failure({ErrorCode::IO, "failed staging cuVS vectors"});
            output.close();
            if (!output) throw ops::Failure({ErrorCode::IO, "failed closing staged cuVS vectors"});
        }
        std::string helper = "kidi-index-cuda";
        std::error_code executable_error;
        const auto executable = std::filesystem::read_symlink("/proc/self/exe", executable_error);
        if (!executable_error) {
            const auto sibling = executable.parent_path() / helper;
            if (std::filesystem::is_regular_file(sibling)) helper = sibling.string();
        } else {
            std::cerr << "kidi_index_cuda|helper_lookup=PATH|reason=" << executable_error.message() << '\n';
        }
        ops::require(
            run_helper({helper, temporary.input().string(), temporary.output().string(),
                        std::to_string(vectors.size(0)), std::to_string(vectors.size(1)), std::to_string(options.m),
                        std::to_string(options.ef_construction), std::to_string(threads)}));
        std::cerr << "Validating every exported vector and packing ordered 24-bit graph references\n";
        return core::HnswIndex::from_hnswlib(temporary.output(), vectors, options.candidates, options.ef_search);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::filesystem::filesystem_error& error) {
        return std::unexpected(Error{ErrorCode::IO, error.what()});
    }
}
#else
auto cuda_index_available() -> Result<bool> { return false; }

auto build_cuda_index(const tensor::Tensor&, core::HnswBuildOptions, std::size_t) -> Result<core::HnswIndex> {
    return std::unexpected(Error{ErrorCode::UNSUPPORTED, "CUDA index construction is disabled in this build"});
}
#endif
} // namespace

auto configure_index_parser(ArgumentParser& parser) -> void {
    parser.description("Learn and save MODEL/lm_head.hnsw from original Gemma FP32/BF16 output weights.");
    parser.add_argument("-m", "--model")
        .type<std::filesystem::path>()
        .required()
        .metavar("MODEL")
        .help("local directory or @owner/model (Python launcher with kidi[hf])");
    parser.add_argument("-c", "--cache")
        .type<std::filesystem::path>()
        .default_value(std::filesystem::path("~/.cache/kidi/model-hub"))
        .help("Hugging Face model cache directory");
    parser.add_argument("-o", "--output")
        .type<std::filesystem::path>()
        .default_value(std::filesystem::path{})
        .metavar("FILE")
        .help("new index path (default: MODEL/lm_head.hnsw)");
    parser.add_argument("--m").default_value<std::size_t>(32).help("maximum upper-layer degree");
    parser.add_argument("--ef-construction")
        .dest("ef_construction")
        .default_value<std::size_t>(200)
        .help("construction candidate-list size");
    parser.add_argument("--candidates").default_value<std::size_t>(8).help("default returned candidates");
    parser.add_argument("--ef-search")
        .dest("ef_search")
        .default_value<std::size_t>(800)
        .help("default search candidate-list size");
    parser.add_argument("--seed").default_value<std::uint64_t>(42);
    parser.add_argument("-j", "--threads")
        .default_value<std::size_t>(1)
        .help("CPU workers for distance calculations; insertion order remains deterministic");
    parser.add_argument("--builder")
        .default_value(std::string("auto"))
        .choices({"auto", "cpu", "cuda"})
        .help("offline construction backend; CUDA requires a CUDA-enabled build and device");
}

auto index_command(const Namespace& arguments, const std::filesystem::path& directory) -> int {
    const auto threads = arguments.get<std::size_t>("threads");
    if (!threads || threads > std::max(1U, std::thread::hardware_concurrency())) {
        spdlog::error("HNSW thread count must be between 1 and the available logical CPU count");
        return 2;
    }
    const auto& requested_builder = arguments.get<std::string>("builder");
    bool use_cuda = false;
    if (requested_builder != "cpu") {
        const auto available = cuda_index_available();
        if (!available) {
            spdlog::error("{}", available.error().message);
            return 1;
        }
        use_cuda = *available;
    }
    if (requested_builder == "cuda" && !use_cuda) {
        spdlog::error("CUDA index construction is unavailable; enable KIDI_ENABLE_CUDA and provide a CUDA device");
        return 2;
    }
    runtime::ynn::set_thread_count(threads);
    core::HnswParallelFor parallel_for;
    if (threads > 1) {
        const auto reserved = runtime::ynn::reserve_thread_pool(threads);
        if (!reserved) {
            spdlog::error("{}", reserved.error().message);
            return 1;
        }
        parallel_for = runtime::ynn::parallel_for;
    }
    auto config = std::filesystem::is_regular_file(directory / "model.yaml")
                      ? checkpoint::load_config(directory / "model.yaml")
                      : checkpoint::load_config(directory, model::Gemma4Impl::checkpoint_config());
    if (!config) {
        spdlog::error("{}", config.error().message);
        return 1;
    }
    auto valid = model::Gemma4Impl::validate_config((*config)["model"]);
    if (!valid) {
        spdlog::error("{}", valid.error().message);
        return 1;
    }
    if ((*config)["model"]["quantization_config"]) {
        spdlog::error("HNSW learning requires original FP32/BF16 Gemma weights");
        return 2;
    }
    auto weights = checkpoint::Weights::load((*config)["weights_file"].as<std::string>());
    if (!weights) {
        spdlog::error("{}", weights.error().message);
        return 1;
    }
    auto vectors = weights->tensor("model.language_model.embed_tokens.weight");
    if (!vectors) {
        spdlog::error("{}", vectors.error().message);
        return 1;
    }
    auto output = arguments.get<std::filesystem::path>("output");
    if (output.empty()) output = directory / "lm_head.hnsw";
    output = std::filesystem::absolute(output);
    if (std::filesystem::exists(output)) {
        spdlog::error("HNSW index already exists; choose a fresh --output or remove it explicitly: {}",
                      output.string());
        return 2;
    }
    const auto m = arguments.get<std::size_t>("m");
    if (m > std::numeric_limits<std::uint32_t>::max()) {
        spdlog::error("HNSW M exceeds uint32 bounds");
        return 2;
    }
    const core::HnswBuildOptions options{
        .m = static_cast<std::uint32_t>(m),
        .ef_construction = arguments.get<std::size_t>("ef_construction"),
        .seed = arguments.get<std::uint64_t>("seed"),
        .candidates = arguments.get<std::size_t>("candidates"),
        .ef_search = arguments.get<std::size_t>("ef_search"),
    };
    const auto started = std::chrono::steady_clock::now();
    auto last = started;
    std::size_t last_completed = 0;
    std::cerr << "Building HNSW over " << vectors->size(0) << " original output rows with "
              << (use_cuda ? "cuVS on CUDA device 0" : "native CPU") << " and " << threads << " CPU threads\n";
    if (!use_cuda) std::cerr << "Indexed 0/" << vectors->size(0) << " (0.00%) | ETA calculating\n";
    const auto progress = [&](std::size_t completed, std::size_t total) {
        const auto now = std::chrono::steady_clock::now();
        if (completed != total && now - last < std::chrono::seconds(5)) return;
        const auto seconds = std::chrono::duration<double>(now - started).count();
        const auto interval = std::chrono::duration<double>(now - last).count();
        const auto rate = interval > 0 ? static_cast<double>(completed - last_completed) / interval : 0.;
        const auto eta = rate > 0 ? static_cast<double>(total - completed) / rate : 0.;
        std::ostringstream line;
        line << std::fixed << std::setprecision(2) << "Indexed " << completed << '/' << total << " ("
             << 100. * static_cast<double>(completed) / static_cast<double>(total) << "%)" << std::setprecision(1)
             << " | elapsed " << seconds << "s | " << rate << " rows/s | ETA " << eta << "s";
        std::cerr << line.str() << '\n';
        last = now;
        last_completed = completed;
    };
    auto index = [&]() -> Result<core::HnswIndex> {
        if (use_cuda) return build_cuda_index(*vectors, options, threads);
        return core::HnswIndex::build(*vectors, options, progress, std::move(parallel_for));
    }();
    if (!index) {
        spdlog::error("{}", index.error().message);
        return index.error().code == ErrorCode::INVALID_ARGUMENT ? 2 : 1;
    }
    auto saved = index->save(output);
    if (!saved) {
        spdlog::error("{}", saved.error().message);
        return 1;
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << nlohmann::json{{"index", output.string()},
                                {"nodes", vectors->size(0)},
                                {"dimensions", vectors->size(1)},
                                {"m", options.m},
                                {"ef_construction", options.ef_construction},
                                {"candidates", options.candidates},
                                {"ef_search", options.ef_search},
                                {"seed", options.seed},
                                {"seed_applied", !use_cuda},
                                {"builder", use_cuda ? "cuvs" : "cpu"},
                                {"threads", threads},
                                {"bytes", std::filesystem::file_size(output)},
                                {"build_seconds", elapsed}}
                     .dump()
              << '\n';
    return 0;
}
} // namespace kidi::cli
