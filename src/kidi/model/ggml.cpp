#include "kidi/model/ggml.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

#include "kidi/model/ggml_dequantize.h"

namespace kidi::model {
namespace {

class Reader {
public:
    explicit Reader(const std::filesystem::path& path)
        : stream_(path, std::ios::binary), size_(std::filesystem::file_size(path)) {
        if (!stream_ || std::endian::native != std::endian::little)
            throw std::runtime_error("GGML reading requires a readable little-endian file");
    }
    auto remaining() const -> std::uint64_t { return size_ - position_; }
    auto position() const -> std::uint64_t { return position_; }
    auto read(void* destination, std::uint64_t count) -> void {
        if (count > remaining() || count > std::numeric_limits<std::streamsize>::max())
            throw std::runtime_error("truncated GGML data");
        stream_.read(static_cast<char*>(destination), static_cast<std::streamsize>(count));
        if (!stream_) throw std::runtime_error("cannot read GGML data");
        position_ += count;
    }
    template <typename Value>
    auto value() -> Value {
        Value result;
        read(&result, sizeof(result));
        return result;
    }
    auto skip(std::uint64_t count) -> void {
        if (count > remaining()) throw std::runtime_error("truncated GGML data");
        stream_.seekg(static_cast<std::streamoff>(count), std::ios::cur);
        if (!stream_) throw std::runtime_error("cannot seek GGML data");
        position_ += count;
    }
    auto string(std::uint64_t length) -> std::string {
        if (length > remaining() || length > 1024 * 1024) throw std::runtime_error("invalid GGML string length");
        std::string result(static_cast<std::size_t>(length), '\0');
        read(result.data(), length);
        return result;
    }
    auto string() -> std::string { return string(value<std::uint64_t>()); }
    auto skip_value(std::uint32_t type, bool array = false) -> void {
        constexpr std::array<std::uint32_t, 13> SIZES{1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (type >= SIZES.size()) throw std::runtime_error("unsupported GGUF metadata type");
        if (type == 8) {
            skip(value<std::uint64_t>());
        } else if (type == 9) {
            if (array) throw std::runtime_error("nested GGUF metadata arrays are invalid");
            const auto element = value<std::uint32_t>();
            const auto count = value<std::uint64_t>();
            if (element >= SIZES.size() || element == 9 || count > remaining())
                throw std::runtime_error("invalid GGUF array type or length");
            if (element == 8) {
                for (std::uint64_t index = 0; index < count; ++index) skip_value(element, true);
            } else {
                if (count > remaining() / SIZES[element]) throw std::runtime_error("truncated GGUF array");
                skip(count * SIZES[element]);
            }
        } else {
            skip(SIZES[type]);
        }
    }

private:
    std::ifstream stream_;
    std::uint64_t size_, position_ = 0;
};

auto tensor_bytes(const GgmlFile::Entry& entry) -> std::uint64_t {
    std::uint64_t elements = 1;
    for (const auto dimension : entry.shape) {
        if (dimension <= 0 || elements > std::numeric_limits<std::int64_t>::max() / sizeof(float) / dimension)
            throw std::runtime_error("invalid GGML tensor dimensions");
        elements *= dimension;
    }
    if (entry.type == 0) return elements * 4;
    if (entry.type == 1 || entry.type == 30) return elements * 2;
    const auto block_bytes = entry.type == 2   ? 18
                             : entry.type == 3 ? 20
                             : entry.type == 6 ? 22
                             : entry.type == 7 ? 24
                             : entry.type == 8 ? 34
                                               : 0;
    if (!block_bytes) throw std::runtime_error("unsupported GGML tensor encoding: " + std::to_string(entry.type));
    if (entry.shape.back() % 32) throw std::runtime_error("GGML quantized row is not a multiple of 32");
    return elements / 32 * block_bytes;
}

auto whisper_name(std::string name) -> std::string {
    if (!name.starts_with("encoder.") && !name.starts_with("decoder."))
        throw std::runtime_error("unknown Whisper GGML tensor name: " + name);
    const std::pair<std::string_view, std::string_view> replacements[]{
        {".positional_embedding", ".embed_positions.weight"},
        {".token_embedding.", ".embed_tokens."},
        {".ln_post.", ".layer_norm."},
        {".ln.", ".layer_norm."},
        {".blocks.", ".layers."},
        {".cross_attn_ln.", ".encoder_attn_layer_norm."},
        {".attn_ln.", ".self_attn_layer_norm."},
        {".mlp_ln.", ".final_layer_norm."},
        {".cross_attn.", ".encoder_attn."},
        {".attn.", ".self_attn."},
        {".query.", ".q_proj."},
        {".key.", ".k_proj."},
        {".value.", ".v_proj."},
        {".out.", ".out_proj."},
        {".mlp.0.", ".fc1."},
        {".mlp.2.", ".fc2."}};
    for (const auto& [source, target] : replacements)
        if (const auto offset = name.find(source); offset != std::string::npos)
            name.replace(offset, source.size(), target);
    return "model." + name;
}

template <typename Block>
auto decode_blocks(Reader& reader, std::span<float> output) -> void {
    std::array<Block, 256> blocks;
    for (std::size_t offset = 0; offset < output.size();) {
        const auto count = std::min(blocks.size(), (output.size() - offset) / 32);
        reader.read(blocks.data(), count * sizeof(Block));
        for (std::size_t index = 0; index < count; ++index)
            ggml_read::dequantize(blocks[index], output.data() + offset + index * 32);
        offset += count * 32;
    }
}

} // namespace

GgmlFile::GgmlFile(const std::filesystem::path& path) : path_(path) {
    Reader reader(path);
    const auto magic = reader.value<std::uint32_t>();
    const auto append = [&](std::string name, Entry entry) {
        if (name.empty() || name.size() > 1024 || name.find('\0') != std::string::npos ||
            !entries_.emplace(std::move(name), std::move(entry)).second)
            throw std::runtime_error("invalid or duplicate GGML tensor name");
    };
    if (magic == GGUF_MAGIC) {
        const auto version = reader.value<std::uint32_t>();
        if (version != 2 && version != 3) throw std::runtime_error("only little-endian GGUF v2/v3 is supported");
        const auto count = reader.value<std::uint64_t>(), metadata = reader.value<std::uint64_t>();
        if (!count || count > 1000000 || metadata > 1000000 || count > reader.remaining() / 32 ||
            metadata > reader.remaining() / 13)
            throw std::runtime_error("invalid GGUF entry counts");
        std::uint32_t alignment = 32;
        std::set<std::string> keys;
        for (std::uint64_t index = 0; index < metadata; ++index) {
            const auto key = reader.string();
            if (key.empty() || !keys.insert(key).second) throw std::runtime_error("invalid GGUF metadata key");
            const auto type = reader.value<std::uint32_t>();
            if (key == "general.alignment") {
                if (type != 4) throw std::runtime_error("GGUF alignment must be uint32");
                alignment = reader.value<std::uint32_t>();
                if (!std::has_single_bit(alignment) || alignment > 65536)
                    throw std::runtime_error("invalid GGUF alignment");
            } else
                reader.skip_value(type);
        }
        for (std::uint64_t index = 0; index < count; ++index) {
            auto name = reader.string();
            const auto rank = reader.value<std::uint32_t>();
            if (!rank || rank > 4) throw std::runtime_error("invalid GGUF tensor rank");
            Entry entry;
            for (std::uint32_t axis = 0; axis < rank; ++axis) entry.shape.push_back(reader.value<std::int64_t>());
            std::ranges::reverse(entry.shape);
            entry.type = reader.value<std::uint32_t>();
            entry.offset = reader.value<std::uint64_t>();
            entry.bytes = tensor_bytes(entry);
            if (entry.offset % alignment) throw std::runtime_error("unaligned GGUF tensor offset");
            append(std::move(name), std::move(entry));
        }
        reader.skip((alignment - reader.position() % alignment) % alignment);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        for (auto& [name, entry] : entries_) {
            if (entry.offset > reader.remaining() || entry.bytes > reader.remaining() - entry.offset)
                throw std::runtime_error("GGUF tensor extends past end of file");
            ranges.emplace_back(entry.offset, entry.offset + entry.bytes);
            entry.offset += reader.position();
        }
        std::ranges::sort(ranges);
        for (std::size_t index = 1; index < ranges.size(); ++index)
            if (ranges[index].first < ranges[index - 1].second) throw std::runtime_error("overlapping GGUF tensors");
    } else if (magic == GGML_MAGIC) {
        whisper_.emplace();
        for (auto& value : *whisper_) value = reader.value<std::int32_t>();
        for (std::size_t index = 0; index < 10; ++index)
            if ((*whisper_)[index] <= 0) throw std::runtime_error("invalid Whisper GGML dimensions");
        const auto version = (*whisper_)[10] / 1000;
        if ((*whisper_)[10] < 0 || (version != 0 && version != 2))
            throw std::runtime_error("unsupported Whisper GGML quantization version");
        const auto mel_bins = reader.value<std::uint32_t>(), mel_width = reader.value<std::uint32_t>();
        if (mel_bins != static_cast<std::uint32_t>((*whisper_)[9]) || mel_bins > 256 || mel_width != 201)
            throw std::runtime_error("invalid Whisper GGML mel filters");
        reader.skip(static_cast<std::uint64_t>(mel_bins) * mel_width * sizeof(float));
        const auto vocabulary = reader.value<std::uint32_t>();
        if (vocabulary > static_cast<std::uint32_t>((*whisper_)[0]) || vocabulary > reader.remaining() / 4)
            throw std::runtime_error("invalid Whisper GGML vocabulary");
        for (std::uint32_t index = 0; index < vocabulary; ++index) reader.skip(reader.value<std::uint32_t>());
        while (reader.remaining()) {
            const auto rank = reader.value<std::uint32_t>(), length = reader.value<std::uint32_t>();
            Entry entry;
            entry.type = reader.value<std::uint32_t>();
            if (!rank || rank > 4 || length > 1024) throw std::runtime_error("invalid Whisper GGML tensor header");
            for (std::uint32_t axis = 0; axis < rank; ++axis) entry.shape.push_back(reader.value<std::int32_t>());
            std::ranges::reverse(entry.shape);
            auto name = whisper_name(reader.string(length));
            if ((name == "model.encoder.conv1.bias" || name == "model.encoder.conv2.bias") && entry.shape.size() == 2 &&
                entry.shape.back() == 1)
                entry.shape.pop_back();
            if (entry.type > 1 && version != 2)
                throw std::runtime_error("legacy GGML quantization layout is unsupported");
            entry.bytes = tensor_bytes(entry);
            entry.offset = reader.position();
            reader.skip(entry.bytes);
            append(std::move(name), std::move(entry));
        }
        if (entries_.empty()) throw std::runtime_error("Whisper GGML has no tensors");
    } else
        throw std::runtime_error("not a GGUF or Whisper GGML checkpoint");
}

auto GgmlFile::open(const std::filesystem::path& path) -> Result<GgmlFile> {
    try {
        return GgmlFile(path);
    } catch (const std::runtime_error& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    }
}

auto GgmlFile::tensor(std::string_view name) const -> Result<tensor::Tensor> {
    try {
        const auto found = entries_.find(std::string(name));
        if (found == entries_.end()) throw std::runtime_error("GGML tensor not found: " + std::string(name));
        const auto& entry = found->second;
        Reader reader(path_);
        reader.skip(entry.offset);
        if (entry.bytes > reader.remaining()) throw std::runtime_error("GGML tensor was truncated");
        auto result = tensor::Tensor::empty(entry.shape, tensor::DType::F32, tensor::Device::cpu());
        if (!result) return result;
        auto values = result->data<float>();
        if (!values) return std::unexpected(values.error());
        if (entry.type == 0)
            reader.read(values->data(), entry.bytes);
        else if (entry.type == 1 || entry.type == 30) {
            std::array<std::uint16_t, 4096> buffer;
            for (std::size_t offset = 0; offset < values->size();) {
                const auto count = std::min(buffer.size(), values->size() - offset);
                reader.read(buffer.data(), count * sizeof(std::uint16_t));
                for (std::size_t index = 0; index < count; ++index)
                    (*values)[offset + index] =
                        entry.type == 1 ? ggml_read::fp16_to_fp32(buffer[index])
                                        : std::bit_cast<float>(static_cast<std::uint32_t>(buffer[index]) << 16);
                offset += count;
            }
        } else if (entry.type == 2)
            decode_blocks<ggml_read::block_q4_0>(reader, *values);
        else if (entry.type == 3)
            decode_blocks<ggml_read::block_q4_1>(reader, *values);
        else if (entry.type == 6)
            decode_blocks<ggml_read::block_q5_0>(reader, *values);
        else if (entry.type == 7)
            decode_blocks<ggml_read::block_q5_1>(reader, *values);
        else if (entry.type == 8)
            decode_blocks<ggml_read::block_q8_0>(reader, *values);
        if (!std::ranges::all_of(*values, [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("non-finite GGML tensor: " + std::string(name));
        return result;
    } catch (const std::runtime_error& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    }
}

} // namespace kidi::model