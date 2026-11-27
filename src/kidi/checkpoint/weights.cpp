#include "kidi/checkpoint/weights.h"

#include <algorithm>
#include <bit>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <cctype>
#include <cstring>
#include <optional>
#include <numeric>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(KIDI_HAS_METAL)
#include <unistd.h>
#endif

#include "kidi/checkpoint/safetensors/mapped.h"
#include "kidi/checkpoint/ggml/reader.h"
#include "kidi/checkpoint/ggml/dequantize.h"
#include "kidi/ops/context.h"

namespace kidi::checkpoint {
namespace {

auto parse_data_type(std::string_view value) -> std::optional<DataType> {
    if (value == "BOOL") return DataType::BOOL;
    if (value == "U8") return DataType::U8;
    if (value == "I8") return DataType::I8;
    if (value == "U16") return DataType::U16;
    if (value == "I16") return DataType::I16;
    if (value == "U32") return DataType::U32;
    if (value == "I32") return DataType::I32;
    if (value == "U64") return DataType::U64;
    if (value == "I64") return DataType::I64;
    if (value == "F16") return DataType::F16;
    if (value == "BF16") return DataType::BF16;
    if (value == "F32") return DataType::F32;
    if (value == "F64") return DataType::F64;
    if (value == "F8_E4M3") return DataType::E4M3;
    if (value == "F8_E5M2") return DataType::E5M2;
    return std::nullopt;
}

auto data_type_size(DataType data_type) -> std::size_t { return tensor::element_size(data_type); }

struct FileStorage {
    explicit FileStorage(const std::filesystem::path& path) {
        std::ifstream stream(path, std::ios::binary);
        std::uint32_t magic = 0;
        stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic == ggml::GGUF_MAGIC || magic == ggml::GGML_MAGIC)
            imported = std::make_unique<ggml::GgmlFile>(ops::require(ggml::GgmlFile::open(path)));
        else
            checkpoint = std::make_unique<safetensors::MappedCheckpoint>(path.string());
    }

    std::unique_ptr<safetensors::MappedCheckpoint> checkpoint;
    std::unique_ptr<ggml::GgmlFile> imported;
};

struct Entry {
    enum class Kind { VIEW, GGML, TENSOR } kind = Kind::VIEW;
    std::optional<DataType> dtype;
    std::string raw_dtype;
    std::vector<std::int64_t> shape;
    std::span<const std::byte> bytes;
    std::string ggml_name;
    tensor::Tensor tensor;
    std::shared_ptr<const void> owner;
};

struct RawTensorView {
    DataType data_type;
    std::vector<std::int64_t> shape;
    std::span<const std::byte> bytes;
    bool mapped_view = false;
    std::shared_ptr<const void> owner;

    auto element_size() const noexcept -> std::size_t { return tensor::element_size(data_type); }
};

auto escape_regex(std::string_view text) -> std::string {
    std::string escaped;
    escaped.reserve(text.size() * 2);
    for (const char ch : text) {
        switch (ch) {
            case '.':
            case '^':
            case '$':
            case '|':
            case '(':
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
            case '+':
            case '*':
            case '?':
            case '\\':
                escaped.push_back('\\');
                break;
            default:
                break;
        }
        escaped.push_back(ch);
    }
    return escaped;
}

auto is_zero_reference(std::string_view source) noexcept -> bool { return source.starts_with("0:"); }

auto find_anchor_index(std::span<const std::string> sources) -> std::size_t {
    for (std::size_t index = 0; index < sources.size(); ++index) {
        if (!is_zero_reference(sources[index])) return index;
    }
    throw std::invalid_argument("weight mapping requires at least one concrete source pattern");
}

} // namespace

auto prefix_state_mappings(std::string_view source_prefix, std::string_view destination_prefix,
                           std::span<const StateMappingSpec> mappings) -> std::vector<StateMappingSpec> {
    const auto escaped_prefix = escape_regex(source_prefix);
    std::vector<StateMappingSpec> prefixed;
    prefixed.reserve(mappings.size());
    for (auto spec : mappings) {
        const auto anchor_index = find_anchor_index(spec.sources);
        for (std::size_t index = 0; index < spec.sources.size(); ++index) {
            auto& source = spec.sources[index];
            if (index == anchor_index) {
                source = "^" + escaped_prefix + (source.starts_with('^') ? source.substr(1) : source);
            } else if (is_zero_reference(source)) {
                source = "0:" + std::string(source_prefix) + source.substr(2);
            } else {
                source = std::string(source_prefix) + source;
            }
        }
        if (spec.op != StateMappingOp::IGNORE && !spec.destination.empty())
            spec.destination = std::string(destination_prefix) + spec.destination;
        prefixed.push_back(std::move(spec));
    }
    return prefixed;
}

auto Weights::save(const std::filesystem::path& path, const StateDict& state) -> Result<void> {
    if (std::endian::native != std::endian::little || state.empty())
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "checkpoint writing requires nonempty little-endian tensors"});
    try {
        nlohmann::json header = nlohmann::json::object();
        std::uint64_t offset = 0;
        std::vector<const StateDict::value_type*> ordered;
        ordered.reserve(state.size());
        for (const auto& item : state) ordered.push_back(&item);
        std::stable_sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
            return tensor::element_size(left->second.dtype()) > tensor::element_size(right->second.dtype());
        });
        constexpr std::array names{"BOOL", "U8",  "I8",   "U16", "I16", "U32",     "I32",    "U64",
                                   "I64",  "F16", "BF16", "F32", "F64", "F8_E4M3", "F8_E5M2"};
        for (const auto* item : ordered) {
            const auto& [name, tensor] = *item;
            const auto dtype = std::ranges::find_if(
                names, [&](const char* value) { return parse_data_type(value) == tensor.dtype(); });
            if (name.empty() || name == "__metadata__" || !tensor.defined() || !tensor.is_contiguous() ||
                dtype == names.end() || tensor.nbytes() > std::numeric_limits<std::uint64_t>::max() - offset ||
                tensor.nbytes() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
                return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid checkpoint tensor: " + name});
            const auto bytes = tensor.host_bytes();
            if (!bytes) return std::unexpected(bytes.error());
            header[name] = {{"dtype", *dtype},
                            {"shape", std::vector<std::int64_t>(tensor.shape().begin(), tensor.shape().end())},
                            {"data_offsets", {offset, offset + tensor.nbytes()}}};
            offset += tensor.nbytes();
        }
        auto serialized = header.dump();
        serialized.append((8 - serialized.size() % 8) % 8, ' ');
        const std::uint64_t length = serialized.size();
        std::ofstream stream(path, std::ios::binary | std::ios::out | std::ios::noreplace);
        if (!stream)
            return std::unexpected(Error{ErrorCode::RUNTIME, "cannot exclusively create checkpoint: " + path.string()});
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
        stream.write(serialized.data(), serialized.size());
        for (const auto* item : ordered) {
            const auto& [name, tensor] = *item;
            const auto bytes = tensor.host_bytes();
            if (!bytes) return std::unexpected(bytes.error());
            stream.write(reinterpret_cast<const char*>(bytes->data()), bytes->size());
        }
        stream.close();
        if (!stream) return std::unexpected(Error{ErrorCode::RUNTIME, "failed to write checkpoint: " + path.string()});
        return {};
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

struct Weights::Impl {
    explicit Impl(const std::filesystem::path& path) : storage(std::make_shared<const FileStorage>(path)) {
        if (storage->imported) {
            for (const auto& [name, unused] : storage->imported->tensors())
                entries.emplace(name, Entry{.kind = Entry::Kind::GGML, .ggml_name = name});
        } else {
            for (const auto& [name, tensor] : storage->checkpoint->tensors()) {
                entries.emplace(name, Entry{
                                          .kind = Entry::Kind::VIEW,
                                          .dtype = parse_data_type(tensor.dtype),
                                          .raw_dtype = tensor.dtype,
                                          .shape = tensor.shape,
                                          .bytes = tensor.bytes(),
                                          .owner = storage,
                                      });
            }
        }
    }

    Impl(std::shared_ptr<const FileStorage> shared_storage, std::unordered_map<std::string, Entry> copied_entries)
        : storage(std::move(shared_storage)), entries(std::move(copied_entries)) {}

    std::shared_ptr<const FileStorage> storage;
    std::unordered_map<std::string, Entry> entries;
};

namespace {

auto render_template(std::string_view value, const std::smatch& match) -> std::string {
    std::string result;
    for (std::size_t index = 0; index < value.size();) {
        if (value[index] == '$' && index + 1 < value.size() &&
            std::isdigit(static_cast<unsigned char>(value[index + 1]))) {
            std::size_t end = index + 1;
            while (end < value.size() && std::isdigit(static_cast<unsigned char>(value[end]))) ++end;
            const auto capture =
                static_cast<std::size_t>(std::stoul(std::string(value.substr(index + 1, end - index - 1))));
            if (capture >= match.size()) throw std::invalid_argument("weight mapping capture is out of range");
            result += match[capture].str();
            index = end;
        } else {
            result.push_back(value[index++]);
        }
    }
    return result;
}

auto materialize_view(const FileStorage& storage, const Entry& entry, std::vector<tensor::Tensor>& scratch)
    -> RawTensorView {
    switch (entry.kind) {
        case Entry::Kind::VIEW:
            if (!entry.dtype) throw std::invalid_argument("unsupported mapped tensor dtype: " + entry.raw_dtype);
            return RawTensorView{
                .data_type = *entry.dtype,
                .shape = entry.shape,
                .bytes = entry.bytes,
                .mapped_view = true,
                .owner = entry.owner,
            };
        case Entry::Kind::GGML:
            scratch.push_back(ops::require(storage.imported->tensor(entry.ggml_name)));
            return RawTensorView{
                .data_type = scratch.back().dtype(),
                .shape = {scratch.back().shape().begin(), scratch.back().shape().end()},
                .bytes = ops::require(scratch.back().host_bytes()),
            };
        case Entry::Kind::TENSOR:
            return RawTensorView{
                .data_type = entry.tensor.dtype(),
                .shape = {entry.tensor.shape().begin(), entry.tensor.shape().end()},
                .bytes = ops::require(entry.tensor.host_bytes()),
            };
    }
    throw std::invalid_argument("invalid weight entry");
}

auto concatenate_entries(const std::vector<RawTensorView>& inputs, std::int64_t axis) -> Entry {
    if (inputs.empty()) throw std::invalid_argument("weight mapping has no inputs");
    const auto rank = static_cast<std::int64_t>(inputs.front().shape.size());
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) throw std::invalid_argument("weight mapping axis is out of range");

    std::vector<std::int64_t> shape(inputs.front().shape.begin(), inputs.front().shape.end());
    bool contiguous_views = axis == 0 && inputs.front().mapped_view && inputs.front().owner != nullptr;
    std::size_t total_bytes = inputs.front().bytes.size();
    for (std::size_t input = 1; input < inputs.size(); ++input) {
        if (inputs[input].data_type != inputs.front().data_type ||
            inputs[input].shape.size() != static_cast<std::size_t>(rank)) {
            throw std::invalid_argument("weight mapping dtype or rank differs");
        }
        for (std::int64_t dimension = 0; dimension < rank; ++dimension) {
            if (dimension != axis && inputs[input].shape[static_cast<std::size_t>(dimension)] !=
                                         shape[static_cast<std::size_t>(dimension)]) {
                throw std::invalid_argument("weight mapping geometry differs");
            }
        }
        shape[static_cast<std::size_t>(axis)] += inputs[input].shape[static_cast<std::size_t>(axis)];
        contiguous_views =
            contiguous_views && inputs[input].mapped_view && inputs[input].owner == inputs.front().owner &&
            inputs[input - 1].bytes.data() + inputs[input - 1].bytes.size() == inputs[input].bytes.data();
        total_bytes += inputs[input].bytes.size();
    }

    if (contiguous_views) {
        return Entry{
            .kind = Entry::Kind::VIEW,
            .dtype = inputs.front().data_type,
            .shape = std::move(shape),
            .bytes = {inputs.front().bytes.data(), total_bytes},
            .owner = inputs.front().owner,
        };
    }

    const auto product = [](std::span<const std::int64_t> dimensions) {
        return std::accumulate(dimensions.begin(), dimensions.end(), std::size_t{1}, std::multiplies<>{});
    };
    const auto inner = product(std::span<const std::int64_t>(shape).subspan(static_cast<std::size_t>(axis + 1)));
    const auto outer = product(std::span<const std::int64_t>(shape).first(static_cast<std::size_t>(axis)));
    const auto element_size = inputs.front().element_size();
    auto output = ops::require(tensor::Tensor::empty(shape, inputs.front().data_type));
    auto* destination = ops::require(output.host_bytes()).data();
    for (std::size_t row = 0; row < outer; ++row) {
        std::size_t axis_offset = 0;
        for (const auto& input : inputs) {
            const auto count = static_cast<std::size_t>(input.shape[static_cast<std::size_t>(axis)]) * inner;
            std::memcpy(
                destination + (row * static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]) + axis_offset) *
                                  element_size,
                input.bytes.data() + row * count * element_size, count * element_size);
            axis_offset += count;
        }
    }
    for (const auto& input : inputs) {
        if (input.mapped_view) (void)safetensors::MappedFile::release_any_pages(input.bytes.data(), input.bytes.size());
    }
    return Entry{
        .kind = Entry::Kind::TENSOR,
        .dtype = output.dtype(),
        .shape = std::move(shape),
        .tensor = std::move(output),
    };
}

auto current_keys(const std::unordered_map<std::string, Entry>& entries) -> std::vector<std::string> {
    std::vector<std::string> keys;
    keys.reserve(entries.size());
    for (const auto& [name, unused] : entries) keys.push_back(name);
    std::ranges::sort(keys);
    return keys;
}

template <typename Storage>
auto apply_state_mappings(Storage& impl, std::span<const StateMappingSpec> specs) -> void {
    for (const auto& spec : specs) {
        const auto anchor_index = find_anchor_index(spec.sources);
        const std::regex anchor(spec.sources[anchor_index]);
        const auto keys = current_keys(impl.entries);
        for (const auto& key : keys) {
            if (!impl.entries.contains(key)) continue;
            std::smatch match;
            if (!std::regex_match(key, match, anchor)) continue;

            std::vector<std::string> sources;
            sources.reserve(spec.sources.size());
            bool complete = true;
            for (std::size_t index = 0; index < spec.sources.size(); ++index) {
                auto source = index == anchor_index ? key : render_template(spec.sources[index], match);
                const auto lookup =
                    is_zero_reference(source) ? std::string_view(source).substr(2) : std::string_view(source);
                complete = complete && impl.entries.contains(std::string(lookup));
                sources.push_back(std::move(source));
            }
            if (!complete) continue;

            if (spec.op == StateMappingOp::IGNORE || spec.destination.empty()) {
                for (const auto& source : sources) {
                    if (!is_zero_reference(source)) impl.entries.erase(source);
                }
                continue;
            }

            const auto destination = render_template(spec.destination, match);
            const bool destination_in_sources = std::ranges::find(sources, destination) != sources.end();
            if (!destination_in_sources && impl.entries.contains(destination)) continue;

            if (spec.op == StateMappingOp::IDENTICAL) {
                std::vector<tensor::Tensor> scratch;
                scratch.reserve(sources.size());
                std::vector<RawTensorView> views;
                views.reserve(sources.size());
                for (const auto& source : sources)
                    views.push_back(materialize_view(*impl.storage, impl.entries.at(source), scratch));
                for (std::size_t index = 1; index < views.size(); ++index) {
                    if (views[index].data_type != views.front().data_type ||
                        !std::ranges::equal(views[index].shape, views.front().shape) ||
                        !std::ranges::equal(views[index].bytes, views.front().bytes))
                        throw ops::Failure({ErrorCode::UNSUPPORTED,
                                            "paired parameters must match for fused projection: " + destination});
                }
                auto retained = std::move(impl.entries.at(sources.front()));
                for (const auto& source : sources) impl.entries.erase(source);
                impl.entries.emplace(destination, std::move(retained));
                continue;
            }

            if (sources.size() == 1 && !is_zero_reference(sources.front())) {
                auto moved = std::move(impl.entries.at(sources.front()));
                impl.entries.erase(sources.front());
                impl.entries.emplace(destination, std::move(moved));
                continue;
            }

            std::vector<tensor::Tensor> scratch;
            scratch.reserve(sources.size() * 2);
            std::vector<std::vector<std::byte>> zero_buffers;
            zero_buffers.reserve(sources.size());
            std::vector<RawTensorView> tensors;
            tensors.reserve(sources.size());
            for (const auto& source : sources) {
                if (is_zero_reference(source)) {
                    const auto reference = materialize_view(*impl.storage, impl.entries.at(source.substr(2)), scratch);
                    zero_buffers.emplace_back(reference.bytes.size(), std::byte{0});
                    tensors.push_back(RawTensorView{
                        .data_type = reference.data_type,
                        .shape = reference.shape,
                        .bytes = zero_buffers.back(),
                    });
                } else {
                    tensors.push_back(materialize_view(*impl.storage, impl.entries.at(source), scratch));
                }
            }
            auto combined = concatenate_entries(tensors, spec.concat_axis);
            for (const auto& source : sources) {
                if (!is_zero_reference(source)) impl.entries.erase(source);
            }
            impl.entries.emplace(destination, std::move(combined));
        }
    }
}

} // namespace

Weights::Weights(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
Weights::Weights(Weights&&) noexcept = default;
auto Weights::operator=(Weights&&) noexcept -> Weights& = default;
Weights::~Weights() = default;

auto Weights::load(const std::filesystem::path& path, std::span<const StateMappingSpec> mappings) -> Result<Weights> {
    try {
        auto impl = std::make_shared<Impl>(path);
        apply_state_mappings(*impl, mappings);
        return Weights(std::move(impl));
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::runtime_error& error) {
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "cannot load weights " + path.string() + ": " + error.what()});
    } catch (const std::exception& error) {
        return std::unexpected(Error{
            ErrorCode::INVALID_ARGUMENT,
            "cannot load weights " + path.string() + ": " + error.what(),
        });
    }
}

auto Weights::mapped(std::span<const StateMappingSpec> mappings) const -> Result<Weights> {
    try {
        auto impl = std::make_shared<Impl>(impl_->storage, impl_->entries);
        apply_state_mappings(*impl, mappings);
        return Weights(std::move(impl));
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, std::string("cannot map weights: ") + error.what()});
    }
}

auto Weights::contains(std::string_view name) const -> bool { return impl_->entries.contains(std::string(name)); }

auto Weights::state_dict() const -> Result<StateDict> {
    StateDict result;
    for (const auto& name : names()) {
        auto value = tensor(name);
        if (!value) return std::unexpected(std::move(value.error()));
        result.emplace(name, std::move(*value));
    }
    return result;
}

auto Weights::names() const -> std::vector<std::string> { return current_keys(impl_->entries); }

auto Weights::size() const noexcept -> std::size_t { return impl_->entries.size(); }

auto Weights::add(std::string name, tensor::Tensor value) -> Result<void> {
    if (!value.defined() || contains(name))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "duplicate or undefined weight tensor: " + name});
    std::vector<std::int64_t> shape(value.shape().begin(), value.shape().end());
    impl_->entries.emplace(std::move(name), Entry{
                                                .kind = Entry::Kind::TENSOR,
                                                .dtype = value.dtype(),
                                                .shape = std::move(shape),
                                                .tensor = std::move(value),
                                            });
    return {};
}

auto Weights::tensor(std::string_view name) const -> Result<tensor::Tensor> {
    const auto found = impl_->entries.find(std::string(name));
    if (found == impl_->entries.end()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "weight tensor not found: " + std::string(name)});
    }
    const auto& entry = found->second;
    switch (entry.kind) {
        case Entry::Kind::TENSOR:
            return entry.tensor;
        case Entry::Kind::GGML:
            return impl_->storage->imported->tensor(entry.ggml_name);
        case Entry::Kind::VIEW: {
            if (!entry.dtype)
                return std::unexpected(
                    Error{ErrorCode::INVALID_ARGUMENT, "unsupported mapped tensor dtype: " + entry.raw_dtype});
            if (reinterpret_cast<std::uintptr_t>(entry.bytes.data()) % data_type_size(*entry.dtype) != 0) {
                return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "weight tensor " + std::string(name) +
                                                                              " is not aligned for zero-copy access"});
            }
            return tensor::Tensor::from_blob(entry.shape, *entry.dtype, entry.bytes, entry.owner);
        }
    }
    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid weight tensor: " + std::string(name)});
}

auto Weights::tensor(std::string_view name, tensor::Device device) const -> Result<tensor::Tensor> {
    auto value = tensor(name);
    if (!value) return value;
#if defined(KIDI_HAS_METAL)
    if (device == tensor::Device::apple_gpu()) {
        const auto& entry = impl_->entries.at(std::string(name));
        const auto address = reinterpret_cast<std::uintptr_t>(entry.bytes.data());
        if (entry.kind == Entry::Kind::VIEW && entry.owner == impl_->storage && !entry.bytes.empty() &&
            address % 4 == 0) {
            const auto page = static_cast<std::size_t>(::getpagesize());
            const auto offset = address % page;
            const auto length = (offset + entry.bytes.size() + page - 1) / page * page;
            const auto element = data_type_size(value->dtype());
            // Metal requires whole pages. Expose only the validated tensor within the mapped range.
            const auto file = impl_->storage->checkpoint->owner();
            std::shared_ptr<const std::byte> mapping;
            try {
                const auto file_base = reinterpret_cast<std::uintptr_t>(file->data());
                mapping = file->map_pages(address - file_base - offset, length);
            } catch (const safetensors::SafetensorsException& error) {
                return std::unexpected(Error{ErrorCode::IO, error.what()});
            }
            auto storage = tensor::Tensor::from_blob({static_cast<std::int64_t>(length / element)}, value->dtype(),
                                                     std::span(mapping.get(), length), mapping, device);
            if (!storage) return std::unexpected(std::move(storage.error()));
            auto view = storage->narrow(0, static_cast<std::int64_t>(offset / element),
                                        static_cast<std::int64_t>(value->numel()));
            if (!view) return view;
            return view->reshape(entry.shape);
        }
    }
#endif
    auto transferred = value->to(device);
    if (transferred && value->device() != device) value->release_pages(false);
    return transferred;
}

auto Weights::float32_tensor(std::string_view name) const -> Result<tensor::Tensor> {
    const auto value = tensor(name);
    if (!value) return std::unexpected(value.error());
    if (value->dtype() == DataType::F32) return value;
    if (value->dtype() != DataType::F16 && value->dtype() != DataType::BF16)
        return std::unexpected(
            Error{ErrorCode::UNSUPPORTED, "weight is not FP16, BF16, or FP32: " + std::string(name)});
    const auto bytes = value->host_bytes();
    if (!bytes) return std::unexpected(bytes.error());
    std::vector<float> floating(value->numel());
    for (std::size_t index = 0; index < floating.size(); ++index) {
        std::uint16_t bits;
        std::memcpy(&bits, bytes->data() + index * sizeof(bits), sizeof(bits));
        floating[index] = value->dtype() == DataType::F16
                              ? ggml::detail::fp16_to_fp32(bits)
                              : std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
    }
    return tensor::Tensor::from_host({value->shape().begin(), value->shape().end()}, std::span<const float>(floating));
}

} // namespace kidi::checkpoint