#pragma once

#include <string_view>

namespace kidi::core {

auto version() noexcept -> std::string_view;

} // namespace kidi::core

namespace kidi {

using core::version;

} // namespace kidi