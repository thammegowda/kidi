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

#include "kidi/checkpoint/safetensors/mapped.h"
#include "kidi/checkpoint/ggml/reader.h"
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

struct OwnedTensor {
    DataType data_type;
    std::vector<std::int64_t> shape;
    std::vector<std::byte> bytes;
};

struct RawTensorView {
    DataType data_type;
    std::span<const std::int64_t> shape;
    std::span<const std::byte> bytes;

    auto element_size() const noexcept -> std::size_t { return tensor::element_size(data_type); }
};

} // namespace

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
    explicit Impl(const std::filesystem::path& path) {
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
    std::unordered_map<std::string, OwnedTensor> transformed;
    std::unordered_map<std::string, std::string> aliases;
    std::unordered_set<std::string> hidden;
    std::unordered_map<std::string, tensor::Tensor> provided;
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

auto mapped_tensor_view(const safetensors::MappedTensor& tensor) -> RawTensorView {
    const auto data_type = parse_data_type(tensor.dtype);
    if (!data_type) throw std::invalid_argument("unsupported mapped tensor dtype: " + tensor.dtype);
    return RawTensorView{.data_type = *data_type, .shape = tensor.shape, .bytes = tensor.bytes()};
}

template <typename Storage>
auto tensor_view(const Storage& impl, std::string_view name) -> RawTensorView {
    if (const auto transformed = impl.transformed.find(std::string(name)); transformed != impl.transformed.end()) {
        return RawTensorView{
            .data_type = transformed->second.data_type,
            .shape = transformed->second.shape,
            .bytes = transformed->second.bytes,
        };
    }
    if (const auto alias = impl.aliases.find(std::string(name)); alias != impl.aliases.end()) {
        return mapped_tensor_view(impl.checkpoint->at(alias->second));
    }
    return mapped_tensor_view(impl.checkpoint->at(std::string(name)));
}

template <typename Storage>
auto contains_tensor(const Storage& impl, std::string_view name) -> bool {
    return impl.provided.contains(std::string(name)) || impl.transformed.contains(std::string(name)) ||
           impl.aliases.contains(std::string(name)) ||
           (!impl.hidden.contains(std::string(name)) &&
            (impl.imported ? impl.imported->tensors().contains(std::string(name))
                           : impl.checkpoint->contains(std::string(name))));
}

template <typename Storage>
auto base_names(const Storage& impl) -> std::vector<std::string> {
    std::vector<std::string> keys;
    if (impl.imported) {
        for (const auto& [name, unused] : impl.imported->tensors()) keys.push_back(name);
    } else {
        for (const auto& [name, unused] : impl.checkpoint->tensors()) keys.push_back(name);
    }
    return keys;
}

auto concatenate_tensors(const std::vector<RawTensorView>& inputs, std::int64_t axis) -> OwnedTensor {
    if (inputs.empty()) throw std::invalid_argument("weight mapping has no inputs");
    const auto rank = static_cast<std::int64_t>(inputs.front().shape.size());
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) throw std::invalid_argument("weight mapping axis is out of range");

    std::vector<std::int64_t> shape(inputs.front().shape.begin(), inputs.front().shape.end());
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
    }

    const auto product = [](std::span<const std::int64_t> dimensions) {
        return std::accumulate(dimensions.begin(), dimensions.end(), std::size_t{1}, std::multiplies<>{});
    };
    const auto inner = product(std::span<const std::int64_t>(shape).subspan(static_cast<std::size_t>(axis + 1)));
    const auto outer = product(std::span<const std::int64_t>(shape).first(static_cast<std::size_t>(axis)));
    const auto element_size = inputs.front().element_size();
    std::vector<std::byte> bytes(product(shape) * element_size);
    for (std::size_t row = 0; row < outer; ++row) {
        std::size_t axis_offset = 0;
        for (const auto& input : inputs) {
            const auto count = static_cast<std::size_t>(input.shape[static_cast<std::size_t>(axis)]) * inner;
            std::memcpy(
                bytes.data() + (row * static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]) + axis_offset) *
                                   element_size,
                input.bytes.data() + row * count * element_size, count * element_size);
            axis_offset += count;
        }
    }
    return OwnedTensor{
        .data_type = inputs.front().data_type,
        .shape = std::move(shape),
        .bytes = std::move(bytes),
    };
}

template <typename Storage>
auto apply_state_mappings(Storage& impl, const std::vector<StateMappingSpec>& specs) -> void {
    for (const auto& spec : specs) {
        const std::regex anchor(spec.sources.front());
        const auto keys = base_names(impl);
        for (const auto& key : keys) {
            std::smatch match;
            if (!std::regex_match(key, match, anchor)) continue;
            const auto destination = render_template(spec.destination, match);
            if (contains_tensor(impl, destination)) continue;

            std::vector<std::string> sources;
            sources.reserve(spec.sources.size());
            sources.push_back(key);
            bool complete = true;
            for (std::size_t index = 1; index < spec.sources.size(); ++index) {
                auto source = render_template(spec.sources[index], match);
                complete = complete && contains_tensor(impl, source);
                sources.push_back(std::move(source));
            }
            if (!complete) continue;
            if (sources.size() == 1) {
                impl.aliases.emplace(destination, sources.front());
                impl.hidden.insert(sources.front());
                continue;
            }

            std::vector<RawTensorView> tensors;
            tensors.reserve(sources.size());
            std::vector<tensor::Tensor> imported;
            imported.reserve(sources.size());
            for (const auto& source : sources) {
                if (impl.imported && !impl.transformed.contains(source)) {
                    const auto alias = impl.aliases.find(source);
                    imported.push_back(
                        ops::require(impl.imported->tensor(alias == impl.aliases.end() ? source : alias->second)));
                    const auto& value = imported.back();
                    tensors.push_back({value.dtype(), value.shape(), ops::require(value.host_bytes())});
                } else
                    tensors.push_back(tensor_view(impl, source));
            }
            impl.transformed.emplace(destination, concatenate_tensors(tensors, spec.concat_axis));
            impl.hidden.insert(sources.begin(), sources.end());
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
        apply_state_mappings(*impl, std::vector<StateMappingSpec>(mappings.begin(), mappings.end()));
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

auto Weights::contains(std::string_view name) const -> bool { return contains_tensor(*impl_, name); }

auto Weights::state_dict() const -> Result<StateDict> {
    StateDict result;
    for (const auto& name : names()) {
        auto value = tensor(name);
        if (!value) return std::unexpected(std::move(value.error()));
        result.emplace(name, std::move(*value));
    }
    return result;
}

auto Weights::names() const -> std::vector<std::string> {
    auto result = base_names(*impl_);
    std::erase_if(result, [&](const auto& name) {
        return impl_->hidden.contains(name) || impl_->transformed.contains(name) || impl_->aliases.contains(name);
    });
    for (const auto& [name, unused] : impl_->aliases) result.push_back(name);
    for (const auto& [name, unused] : impl_->transformed) result.push_back(name);
    for (const auto& [name, unused] : impl_->provided) result.push_back(name);
    std::ranges::sort(result);
    return result;
}

auto Weights::size() const noexcept -> std::size_t {
    return (impl_->imported ? impl_->imported->tensors().size() : impl_->checkpoint->tensors().size()) -
           impl_->hidden.size() + impl_->transformed.size() + impl_->aliases.size() + impl_->provided.size();
}

auto Weights::add(std::string name, tensor::Tensor value) -> Result<void> {
    if (!value.defined() || contains(name))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "duplicate or undefined weight tensor: " + name});
    impl_->provided.emplace(std::move(name), std::move(value));
    return {};
}

auto Weights::tensor(std::string_view name) const -> Result<tensor::Tensor> {
    if (!contains(name)) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "weight tensor not found: " + std::string(name)});
    }
    if (const auto found = impl_->provided.find(std::string(name)); found != impl_->provided.end())
        return found->second;
    if (impl_->imported && !impl_->transformed.contains(std::string(name))) {
        const auto alias = impl_->aliases.find(std::string(name));
        return impl_->imported->tensor(alias == impl_->aliases.end() ? name : std::string_view(alias->second));
    }
    const auto raw = tensor_view(*impl_, name);
    if (reinterpret_cast<std::uintptr_t>(raw.bytes.data()) % data_type_size(raw.data_type) != 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT,
                                     "weight tensor " + std::string(name) + " is not aligned for zero-copy access"});
    }
    std::shared_ptr<const void> owner = impl_;
    return tensor::Tensor::from_blob(std::vector<std::int64_t>(raw.shape.begin(), raw.shape.end()), raw.data_type,
                                     raw.bytes, std::move(owner));
}

} // namespace kidi::checkpoint