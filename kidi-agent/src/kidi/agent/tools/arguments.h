#pragma once

#include "kidi/agent/agent.h"

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace kidi::agent::tools {

// Tool-call argument readers; failures are INVALID_ARGUMENT errors the model can act on.
inline auto string_argument(const Json& args, std::string_view key) -> Result<std::string> {
    const auto item = args.find(key);
    if (item == args.end() || !item->is_string())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, std::string(key) + " must be a string"});
    return item->get<std::string>();
}

inline auto size_argument(const Json& args, std::string_view key, std::size_t fallback) -> Result<std::size_t> {
    const auto item = args.find(key);
    if (item == args.end()) return fallback;
    if (!item->is_number_unsigned() && (!item->is_number_integer() || item->get<std::int64_t>() < 0))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, std::string(key) + " must be a nonnegative integer"});
    const auto value = item->get<std::uint64_t>();
    if (value > std::numeric_limits<std::size_t>::max())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, std::string(key) + " is too large"});
    return static_cast<std::size_t>(value);
}

} // namespace kidi::agent::tools
