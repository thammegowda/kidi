#pragma once

#include <string_view>

namespace kidi::core {

[[nodiscard]] std::string_view version() noexcept;

} // namespace kidi::core