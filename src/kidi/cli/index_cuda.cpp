#include "kidi/cli/index_cuda.h"
#include "kidi/ops/context.h"
#include "kidi/tensor/backend.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

namespace kidi::cli {
namespace {
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
} // namespace

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
} // namespace kidi::cli
