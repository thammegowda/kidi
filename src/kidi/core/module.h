#pragma once

#include <memory>
#include <optional>
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
    using NAME = ::kidi::core::ModuleHolder<NAME##Impl>

namespace kidi::model {
class Weights;
}

namespace kidi::core {

using StateDict = std::map<std::string, tensor::Tensor, std::less<>>;

template <typename Implementation>
class ModuleHolder {
public:
    ModuleHolder() : impl_(std::make_shared<Implementation>()) {}
    ModuleHolder(std::nullptr_t) noexcept {}
    ModuleHolder(std::shared_ptr<Implementation> impl) noexcept : impl_(std::move(impl)) {}

    template <typename... Args>
        requires(sizeof...(Args) > 0 && std::is_constructible_v<Implementation, Args...>)
    explicit ModuleHolder(Args&&... args) : impl_(std::make_shared<Implementation>(std::forward<Args>(args)...)) {}

    auto get() const noexcept -> Implementation* { return impl_.get(); }
    auto ptr() const noexcept -> const std::shared_ptr<Implementation>& { return impl_; }
    auto operator->() const noexcept -> Implementation* { return get(); }
    auto operator*() const noexcept -> Implementation& { return *impl_; }
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    std::shared_ptr<Implementation> impl_;
};

inline thread_local tensor::DType module_dtype = tensor::DType::F32;
inline thread_local bool allocate_parameters = true;
auto default_module_device() -> tensor::Device;
extern thread_local tensor::Device module_device;

class ModuleScope {
public:
    explicit ModuleScope(tensor::DType dtype = module_dtype, bool allocate = allocate_parameters,
                         tensor::Device device = module_device) noexcept
        : previous_dtype_(std::exchange(module_dtype, dtype)),
          previous_allocate_(std::exchange(allocate_parameters, allocate)),
          previous_device_(std::exchange(module_device, device)) {}
    explicit ModuleScope(tensor::Device device) noexcept : ModuleScope(module_dtype, allocate_parameters, device) {}
    ~ModuleScope() {
        module_dtype = previous_dtype_;
        allocate_parameters = previous_allocate_;
        module_device = previous_device_;
    }
    ModuleScope(const ModuleScope&) = delete;
    auto operator=(const ModuleScope&) -> ModuleScope& = delete;

private:
    tensor::DType previous_dtype_;
    bool previous_allocate_;
    tensor::Device previous_device_;
};

class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    Module(const Module&) = delete;
    auto operator=(const Module&) -> Module& = delete;
    Module(Module&&) = delete;
    auto operator=(Module&&) -> Module& = delete;

    auto device() const noexcept -> tensor::Device { return device_; }

    auto state_dict() const -> StateDict;
    auto set_state(const StateDict& state, bool strict = true) -> Result<void>;
    auto set_state(const model::Weights& weights, bool strict = true) -> Result<void>;
    auto load_state_dict(const StateDict& state, bool strict = true) -> Result<void>;

protected:
    auto register_parameter(std::string name, tensor::Tensor& parameter) -> void;
    auto register_parameter(std::string name, tensor::Tensor& parameter, std::vector<std::int64_t> shape,
                            tensor::DType dtype = module_dtype, bool allocate = allocate_parameters,
                            std::optional<tensor::Device> storage_device = {}) -> void;
    auto register_module(std::string name, std::shared_ptr<Module> module) -> void;
    template <typename Implementation>
    auto register_module(std::string name, const ModuleHolder<Implementation>& module) -> void {
        register_module(std::move(name), module.ptr());
    }
    auto tie_parameter(std::string alias, std::string target) -> void;

private:
    struct Parameter {
        tensor::Tensor* value;
        std::vector<std::int64_t> shape;
        tensor::DType dtype;
        tensor::Device device;
        std::string alias;
    };
    using Slots = std::map<std::string, Parameter, std::less<>>;
    auto assign_state(const StateDict& state, bool strict, bool copy) -> Result<void>;
    auto collect(Slots& result, const std::string& prefix) const -> void;
    auto contains(const Module* module) const -> bool;
    auto check_name(std::string_view name) const -> void;
    tensor::Device device_ = module_device;
    Slots parameters_;
    std::map<std::string, std::string, std::less<>> aliases_;
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
    template <typename Child>
    auto push_back(const ModuleHolder<Child>& module) -> void {
        push_back(module.ptr());
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
using ModuleList = ModuleHolder<ModuleListImpl<Implementation>>;

template <typename Implementation = Module>
class ModuleMapImpl : public Module {
    static_assert(std::is_base_of_v<Module, Implementation>);

public:
    using Pointer = std::shared_ptr<Implementation>;
    auto insert(std::string name, Pointer module) -> void {
        register_module(name, module);
        modules_.emplace(std::move(name), std::move(module));
    }
    template <typename Child>
    auto insert(std::string name, const ModuleHolder<Child>& module) -> void {
        insert(std::move(name), module.ptr());
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
using ModuleMap = ModuleHolder<ModuleMapImpl<Implementation>>;
} // namespace kidi::core

namespace kidi {
using core::allocate_parameters;
using core::default_module_device;
using core::Module;
using core::module_device;
using core::module_dtype;
using core::ModuleHolder;
using core::ModuleList;
using core::ModuleListImpl;
using core::ModuleMap;
using core::ModuleMapImpl;
using core::ModuleScope;
using core::StateDict;
} // namespace kidi