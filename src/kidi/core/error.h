#pragma once

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

} // namespace kidi::core