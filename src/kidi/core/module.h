#pragma once

#include <memory>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <utility>

#include "kidi/tensor/tensor.h"

#define KIDI_MODULE(NAME) \
    class NAME##Impl;     \
    using NAME = ::std::shared_ptr<NAME##Impl>

namespace kidi::core {

using StateDict = std::map<std::string, tensor::Tensor, std::less<>>;

class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    Module(const Module&) = delete;
    auto operator=(const Module&) -> Module& = delete;
    Module(Module&&) = delete;
    auto operator=(Module&&) -> Module& = delete;

    auto state_dict() const -> StateDict;
    auto load_state_dict(const StateDict& state, bool strict = true) -> Result<void>;

protected:
    auto register_parameter(std::string name, tensor::Tensor& parameter) -> void;
    auto register_module(std::string name, std::shared_ptr<Module> module) -> void;

private:
    using Slots = std::map<std::string, tensor::Tensor*, std::less<>>;
    auto collect(Slots& result, const std::string& prefix) const -> void;
    auto contains(const Module* module) const -> bool;
    auto check_name(std::string_view name) const -> void;
    Slots parameters_;
    std::map<std::string, std::shared_ptr<Module>, std::less<>> modules_;
};

template <typename Implementation = Module>
class ModuleListImpl : public Module {
    static_assert(std::is_base_of_v<Module, Implementation>);

public:
    using Pointer = std::shared_ptr<Implementation>;
    auto push_back(Pointer module) -> void {
        register_module(std::to_string(modules_.size()), module);
        modules_.push_back(std::move(module));
    }
    auto at(std::size_t index) const -> const Pointer& { return modules_.at(index); }
    auto operator[](std::size_t index) const -> const Pointer& { return at(index); }
    auto size() const noexcept -> std::size_t { return modules_.size(); }
    auto empty() const noexcept -> bool { return modules_.empty(); }
    auto begin() const noexcept { return modules_.begin(); }
    auto end() const noexcept { return modules_.end(); }

private:
    std::vector<Pointer> modules_;
};

template <typename Implementation = Module>
using ModuleList = std::shared_ptr<ModuleListImpl<Implementation>>;

template <typename Implementation = Module>
class ModuleMapImpl : public Module {
    static_assert(std::is_base_of_v<Module, Implementation>);

public:
    using Pointer = std::shared_ptr<Implementation>;
    auto insert(std::string name, Pointer module) -> void {
        register_module(name, module);
        modules_.emplace(std::move(name), std::move(module));
    }
    auto at(std::string_view name) const -> const Pointer& {
        auto found = modules_.find(name);
        if (found == modules_.end()) throw std::out_of_range("unknown module: " + std::string(name));
        return found->second;
    }
    auto size() const noexcept -> std::size_t { return modules_.size(); }
    auto empty() const noexcept -> bool { return modules_.empty(); }
    auto begin() const noexcept { return modules_.begin(); }
    auto end() const noexcept { return modules_.end(); }

private:
    std::map<std::string, Pointer, std::less<>> modules_;
};

template <typename Implementation = Module>
using ModuleMap = std::shared_ptr<ModuleMapImpl<Implementation>>;
} // namespace kidi::core

namespace kidi {
using core::Module;
using core::ModuleList;
using core::ModuleListImpl;
using core::ModuleMap;
using core::ModuleMapImpl;
using core::StateDict;
} // namespace kidi