#include "kidi/model/weights.h"

#include <algorithm>
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

#include "kidi/model/safetensors/mapped.h"

namespace kidi::model {
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

struct Weights::Impl {
    explicit Impl(const std::filesystem::path& path) : checkpoint(path.string()) {}

    safetensors::MappedCheckpoint checkpoint;
    std::unordered_map<std::string, OwnedTensor> transformed;
    std::unordered_map<std::string, std::string> aliases;
    std::unordered_set<std::string> hidden;
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
        return mapped_tensor_view(impl.checkpoint.at(alias->second));
    }
    return mapped_tensor_view(impl.checkpoint.at(std::string(name)));
}

template <typename Storage>
auto contains_tensor(const Storage& impl, std::string_view name) -> bool {
    return impl.transformed.contains(std::string(name)) || impl.aliases.contains(std::string(name)) ||
           (!impl.hidden.contains(std::string(name)) && impl.checkpoint.contains(std::string(name)));
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
        std::vector<std::string> keys;
        keys.reserve(impl.checkpoint.tensors().size());
        for (const auto& [name, tensor] : impl.checkpoint.tensors()) {
            static_cast<void>(tensor);
            keys.push_back(name);
        }
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
            for (const auto& source : sources) tensors.push_back(tensor_view(impl, source));
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
    } catch (const std::exception& error) {
        return std::unexpected(Error{
            ErrorCode::INVALID_ARGUMENT,
            "cannot load Safetensors weights " + path.string() + ": " + error.what(),
        });
    }
}

auto Weights::contains(std::string_view name) const -> bool { return contains_tensor(*impl_, name); }

auto Weights::size() const noexcept -> std::size_t {
    return impl_->checkpoint.tensors().size() - impl_->hidden.size() + impl_->transformed.size() +
           impl_->aliases.size();
}

auto Weights::tensor(std::string_view name) const -> Result<tensor::Tensor> {
    if (!contains(name)) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "weight tensor not found: " + std::string(name)});
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

} // namespace kidi::model