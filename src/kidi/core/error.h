#pragma once

#include <expected>
#include <string>

namespace kidi::core {

enum class ErrorCode {
    INVALID_ARGUMENT,
    IO,
    INVALID_MANIFEST,
    UNSUPPORTED,
    RUNTIME,
};

struct Error {
    ErrorCode code;
    std::string message;
};

template <typename T>
using Result = std::expected<T, Error>;

} // namespace kidi::core

namespace kidi {

using core::Error;
using core::ErrorCode;
using core::Result;

} // namespace kidi