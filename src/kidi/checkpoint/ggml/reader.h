#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "kidi/tensor/tensor.h"

namespace kidi::checkpoint::ggml {

inline constexpr std::uint32_t GGUF_MAGIC = 0x46554747;
inline constexpr std::uint32_t GGML_MAGIC = 0x67676d6c;

class GgmlFile {
public:
    struct Entry {
        std::vector<std::int64_t> shape;
        std::uint32_t type;
        std::uint64_t offset, bytes;
    };

    static auto open(const std::filesystem::path& path) -> Result<GgmlFile>;
    auto tensors() const -> const std::map<std::string, Entry>& { return entries_; }
    auto whisper_header() const -> const std::optional<std::array<std::int32_t, 11>>& { return whisper_; }
    auto tensor(std::string_view name) const -> Result<tensor::Tensor>;

private:
    explicit GgmlFile(const std::filesystem::path& path);
    std::filesystem::path path_;
    std::map<std::string, Entry> entries_;
    std::optional<std::array<std::int32_t, 11>> whisper_;
};

} // namespace kidi::checkpoint::ggml