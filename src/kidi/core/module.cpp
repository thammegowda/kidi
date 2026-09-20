#include "kidi/core/module.h"
#include <algorithm>
#include <stdexcept>

namespace kidi::core {
auto Module::check_name(std::string_view name) const -> void {
    if (name.empty() || name.find('.') != std::string_view::npos || parameters_.contains(name) ||
        modules_.contains(name))
        throw std::invalid_argument("invalid or duplicate module member name: " + std::string(name));
}
auto Module::register_parameter(std::string name, tensor::Tensor& parameter) -> void {
    check_name(name);
    if (!parameter.defined()) throw std::invalid_argument("cannot register undefined parameter: " + name);
    parameters_.emplace(std::move(name), &parameter);
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
}
auto Module::state_dict() const -> StateDict {
    Slots slots;
    collect(slots, "");
    StateDict result;
    for (const auto& [name, parameter] : slots) result.emplace(name, *parameter);
    return result;
}
auto Module::load_state_dict(const StateDict& state, bool strict) -> Result<void> {
    Slots slots;
    collect(slots, "");
    if (strict) {
        for (const auto& [name, value] : state)
            if (!slots.contains(name))
                return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "unexpected parameter: " + name});
        for (const auto& [name, parameter] : slots)
            if (!state.contains(name))
                return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "missing parameter: " + name});
    }
    for (const auto& [name, parameter] : slots) {
        auto found = state.find(name);
        if (found == state.end()) continue;
        const auto& value = found->second;
        if (!value.defined() || !value.is_contiguous() || value.dtype() != parameter->dtype() ||
            !std::ranges::equal(value.shape(), parameter->shape()))
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "parameter shape or dtype mismatch: " + name});
    }
    std::vector<std::pair<tensor::Tensor*, tensor::Tensor>> replacements;
    for (const auto& [name, parameter] : slots) {
        auto found = state.find(name);
        if (found == state.end()) continue;
        const auto& value = found->second;
        auto bytes = value.copy_to_host();
        if (!bytes) return std::unexpected(std::move(bytes.error()));
        auto copy = tensor::Tensor::from_bytes({value.shape().begin(), value.shape().end()}, value.dtype(), *bytes,
                                               parameter->device());
        if (!copy) return std::unexpected(std::move(copy.error()));
        replacements.emplace_back(parameter, std::move(*copy));
    }
    for (auto& [parameter, value] : replacements) *parameter = std::move(value);
    return {};
}
} // namespace kidi::core