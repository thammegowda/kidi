#pragma once

#include <coroutine>
#include <optional>
#include <utility>
#include <variant>

#include "kidi/ops/context.h"

namespace kidi::inference {

template <typename Value, typename Interface>
class CoroutineSession final : public Interface {
public:
    struct promise_type;
    using Handle = std::coroutine_handle<promise_type>;
    struct promise_type {
        std::optional<Result<Value>> completed;
        auto get_return_object() -> CoroutineSession { return CoroutineSession(Handle::from_promise(*this)); }
        static auto initial_suspend() noexcept -> std::suspend_always { return {}; }
        static auto final_suspend() noexcept -> std::suspend_always { return {}; }
        static auto yield_value(std::monostate) noexcept -> std::suspend_always { return {}; }
        auto return_value(Result<Value> value) -> void { completed.emplace(std::move(value)); }
        auto unhandled_exception() -> void {
            try { throw; }
            catch (const ops::Failure& error) { completed.emplace(std::unexpected(error.error())); }
            catch (const std::exception& error) {
                completed.emplace(std::unexpected(Error{ErrorCode::RUNTIME, error.what()}));
            } catch (...) {
                completed.emplace(std::unexpected(Error{ErrorCode::RUNTIME, "unknown inference failure"}));
            }
        }
    };

    explicit CoroutineSession(Handle handle) : handle_(handle) {}
    CoroutineSession(CoroutineSession&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}
    auto operator=(CoroutineSession&& other) noexcept -> CoroutineSession& {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = std::exchange(other.handle_, {});
        }
        return *this;
    }
    ~CoroutineSession() override {
        if (handle_) handle_.destroy();
    }
    CoroutineSession(const CoroutineSession&) = delete;
    auto operator=(const CoroutineSession&) -> CoroutineSession& = delete;

    auto step() -> Result<std::optional<Value>> override {
        if (!handle_ || handle_.done())
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "inference session is already complete"});
        handle_.resume();
        auto& promise = handle_.promise();
        if (!promise.completed) return std::optional<Value>{};
        auto completed = std::move(*promise.completed);
        handle_.destroy();
        handle_ = {};
        if (!completed) return std::unexpected(std::move(completed.error()));
        return std::optional<Value>{std::move(*completed)};
    }

private:
    Handle handle_;
};
} // namespace kidi::inference
