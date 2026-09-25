#include "kidi/core/module.h"
#include "kidi/model/weights.h"
#include <algorithm>
#include <stdexcept>

namespace kidi::core {
auto default_module_device() -> tensor::Device {
    for (auto device : {tensor::Device::apple_gpu(), tensor::Device::cuda()}) {
        auto backend = tensor::BackendRegistry::instance().backend(device);
        if (backend && (*backend)->is_available(device) && (*backend)->supports_execution()) return device;
    }
    return tensor::Device::cpu();
}

thread_local tensor::Device module_device = default_module_device();

auto Module::check_name(std::string_view name) const -> void {
    if (name.empty() || name.find('.') != std::string_view::npos || parameters_.contains(name) ||
        modules_.contains(name))
        throw std::invalid_argument("invalid or duplicate module member name: " + std::string(name));
}
auto Module::register_parameter(std::string name, tensor::Tensor& parameter) -> void {
    check_name(name);
    if (!parameter.defined()) throw std::invalid_argument("cannot register undefined parameter: " + name);
    parameters_.emplace(
        std::move(name),
        Parameter{
            &parameter, {parameter.shape().begin(), parameter.shape().end()}, parameter.dtype(), parameter.device()});
}
auto Module::register_parameter(std::string name, tensor::Tensor& parameter, std::vector<std::int64_t> shape,
                                tensor::DType dtype, bool allocate, std::optional<tensor::Device> storage_device)
    -> void {
    check_name(name);
    if (std::ranges::any_of(shape, [](auto extent) { return extent <= 0; }))
        throw std::invalid_argument("parameter dimensions must be positive: " + name);
    if (allocate) {
        auto value = tensor::Tensor::zeros(shape, dtype, storage_device.value_or(device_));
        if (!value) throw std::runtime_error(value.error().message);
        parameter = std::move(*value);
    }
    parameters_.emplace(std::move(name),
                        Parameter{&parameter, std::move(shape), dtype, storage_device.value_or(device_)});
}
auto Module::contains(const Module* module) const -> bool {
    if (this == module) return true;
    for (const auto& [name, child] : modules_)
        if (child->contains(module)) return true;
    return false;
}
auto Module::register_module(std::string name, std::shared_ptr<Module> module) -> void {
    check_name(name);
    if (!module || module->contains(this)) throw std::invalid_argument("null or cyclic child module: " + name);
    modules_.emplace(std::move(name), std::move(module));
}
auto Module::collect(Slots& result, const std::string& prefix) const -> void {
    for (const auto& [name, parameter] : parameters_) result.emplace(prefix + name, parameter);
    for (const auto& [name, module] : modules_) module->collect(result, prefix + name + '.');
    for (const auto& [alias, target] : aliases_) result.at(prefix + alias).alias = prefix + target;
}
auto Module::tie_parameter(std::string alias, std::string target) -> void {
    Slots slots;
    collect(slots, "");
    if (alias == target || !slots.contains(alias) || !slots.contains(target) || !slots.at(alias).alias.empty() ||
        !slots.at(target).alias.empty() || slots.at(alias).shape != slots.at(target).shape ||
        slots.at(alias).dtype != slots.at(target).dtype || slots.at(alias).device != slots.at(target).device)
        throw std::invalid_argument("invalid tied parameters: " + alias + ", " + target);
    *slots.at(alias).value = *slots.at(target).value;
    aliases_.emplace(std::move(alias), std::move(target));
}
auto Module::state_dict() const -> StateDict {
    Slots slots;
    collect(slots, "");
    StateDict result;
    for (const auto& [name, parameter] : slots) result.emplace(name, *parameter.value);
    return result;
}
auto Module::set_state(const StateDict& state, bool strict) -> Result<void> {
    return assign_state(state, strict, false);
}
auto Module::set_state(const model::Weights& weights, bool strict) -> Result<void> {
    auto state = weights.state_dict();
    if (!state) return std::unexpected(std::move(state.error()));
    return set_state(*state, strict);
}
auto Module::load_state_dict(const StateDict& state, bool strict) -> Result<void> {
    return assign_state(state, strict, true);
}
auto Module::assign_state(const StateDict& state, bool strict, bool copy) -> Result<void> {
    Slots slots;
    collect(slots, "");
    if (strict) {
        for (const auto& [name, value] : state)
            if (!slots.contains(name))
                return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "unexpected parameter: " + name});
        for (const auto& [name, parameter] : slots)
            if (parameter.alias.empty() && !state.contains(name))
                return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "missing parameter: " + name});
    }
    for (const auto& [name, parameter] : slots) {
        auto found = state.find(name);
        if (found == state.end()) continue;
        const auto& value = found->second;
        if (!value.defined() || !value.is_contiguous() || value.dtype() != parameter.dtype ||
            !std::ranges::equal(value.shape(), parameter.shape))
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "parameter shape or dtype mismatch: " + name});
    }
    StateDict replacements;
    for (const auto& [name, parameter] : slots) {
        if (!parameter.alias.empty()) {
            if (state.contains(name) && !state.contains(parameter.alias))
                return std::unexpected(
                    Error{ErrorCode::INVALID_ARGUMENT, "tied parameter requires: " + parameter.alias});
            continue;
        }
        auto found = state.find(name);
        if (found == state.end()) continue;
        const auto& value = found->second;
        if (!copy) {
            auto bound = value.to(parameter.device);
            if (!bound) return std::unexpected(std::move(bound.error()));
            replacements.emplace(name, std::move(*bound));
            continue;
        }
        auto bytes = value.copy_to_host();
        if (!bytes) return std::unexpected(std::move(bytes.error()));
        auto copy = tensor::Tensor::from_bytes({value.shape().begin(), value.shape().end()}, value.dtype(), *bytes,
                                               parameter.device);
        if (!copy) return std::unexpected(std::move(copy.error()));
        replacements.emplace(name, std::move(*copy));
    }
    for (const auto& [name, parameter] : slots) {
        if (parameter.alias.empty() || !replacements.contains(parameter.alias)) continue;
        if (state.contains(name)) {
            const auto& alias = state.at(name);
            const auto& canonical = state.at(parameter.alias);
            const auto alias_bytes = alias.host_bytes(), canonical_bytes = canonical.host_bytes();
            if (!alias_bytes || !canonical_bytes || alias_bytes->data() != canonical_bytes->data()) {
                auto left = alias.copy_to_host(), right = canonical.copy_to_host();
                if (!left) return std::unexpected(std::move(left.error()));
                if (!right) return std::unexpected(std::move(right.error()));
                if (*left != *right)
                    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "conflicting tied parameter: " + name});
            }
        }
        replacements.emplace(name, replacements.at(parameter.alias));
    }
    for (auto& [name, value] : replacements) *slots.at(name).value = std::move(value);
    return {};
}
} // namespace kidi::core