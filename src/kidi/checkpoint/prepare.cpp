#include "kidi/checkpoint/prepare.h"

#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <nlohmann/json.hpp>

#include "kidi/ops/context.h"

namespace kidi::checkpoint {
using ops::require;

auto prepare(const std::filesystem::path& path, const ConfigAdapter& adapter, PreparationHook customize)
    -> Result<std::filesystem::path> {
    static std::mutex conversion_mutex;
    std::scoped_lock lock(conversion_mutex);
    std::filesystem::path temporary;
    try {
        if (!customize) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "missing checkpoint preparation hook"});
        const auto config = require(load_config(path, adapter));
        const auto policy = require(customize(config));
        const auto cache_name = std::filesystem::path(policy.cache_directory);
        if (cache_name.empty() || cache_name.has_parent_path() || cache_name == "." || cache_name == ".." ||
            policy.format.empty() || !policy.convert)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid checkpoint preparation policy"});
        const auto source_path = std::filesystem::absolute(policy.source).lexically_normal();
        const auto source_size = std::filesystem::file_size(source_path);
        const auto source_time = std::filesystem::last_write_time(source_path).time_since_epoch().count();
        const auto root = source_path.parent_path();
        const auto destination = root / cache_name;
        const ConfigSource source{root, {}, {}};
        for (const auto& name : policy.files) {
            if (std::filesystem::path(name).has_parent_path())
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "checkpoint sidecars must be filenames"});
            require(source.file(name));
        }
        if (std::filesystem::exists(destination)) {
            std::ifstream stream(destination / "quantization.json");
            const auto metadata = nlohmann::json::parse(stream);
            if (metadata.value("format", "") != policy.format || metadata.at("source_bytes") != source_size ||
                metadata.at("source_mtime") != source_time ||
                metadata.at("model_bytes") != std::filesystem::file_size(destination / "model.safetensors"))
                throw ops::Failure(
                    {ErrorCode::INVALID_ARGUMENT, "stale or incomplete checkpoint cache: " + destination.string()});
            for (const auto& name : policy.files)
                if (!std::filesystem::is_regular_file(destination / name))
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "incomplete checkpoint cache"});
            return destination;
        }
        const auto weights = require(Weights::load(source_path));
        const auto checkpoint = require(policy.convert(config["model"], weights));
        const auto bytes =
            std::accumulate(checkpoint.begin(), checkpoint.end(), std::uint64_t{0},
                            [](std::uint64_t total, const auto& item) { return total + item.second.nbytes(); });
        if (std::filesystem::space(root).available < bytes + 16 * 1024 * 1024)
            throw ops::Failure({ErrorCode::RUNTIME, "not enough storage for the checkpoint cache"});
        std::random_device random;
        for (int attempt = 0; attempt < 8 && temporary.empty(); ++attempt) {
            auto candidate = root / (".kidi-checkpoint-" + std::to_string(random()));
            if (std::filesystem::create_directory(candidate)) temporary = std::move(candidate);
        }
        if (temporary.empty()) throw ops::Failure({ErrorCode::RUNTIME, "cannot create checkpoint staging directory"});
        require(Weights::save(temporary / "model.safetensors", checkpoint));
        for (const auto& name : policy.files) std::filesystem::copy_file(root / name, temporary / name);
        auto metadata = nlohmann::json{{"format", policy.format},
                                       {"source_file", source_path.filename().string()},
                                       {"source_bytes", source_size},
                                       {"source_mtime", source_time},
                                       {"model_bytes", std::filesystem::file_size(temporary / "model.safetensors")}};
        for (const auto& [key, value] : policy.metadata) {
            if (metadata.contains(key))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "reserved checkpoint metadata key: " + key});
            metadata[key] = value;
        }
        std::ofstream stream(temporary / "quantization.json");
        stream << metadata.dump(2) << '\n';
        stream.close();
        if (!stream) throw ops::Failure({ErrorCode::RUNTIME, "failed to write checkpoint metadata"});
        std::filesystem::rename(temporary, destination);
        temporary.clear();
        return destination;
    } catch (const ops::Failure& error) {
        if (!temporary.empty()) {
            std::error_code ignored;
            std::filesystem::remove_all(temporary, ignored);
        }
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        if (!temporary.empty()) {
            std::error_code ignored;
            std::filesystem::remove_all(temporary, ignored);
        }
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

} // namespace kidi::checkpoint