#pragma once

#include "kidi/agent/agent.h"

#include <chrono>
#include <functional>
#include <string>

namespace kidi::agent::net {

// A browser sign-in that returns to a temporary listener on 127.0.0.1, for front ends without their own web server.
// `show` presents the sign-in link (and may open it); the listener waits up to `timeout` for the browser's return.
auto loopback_sign_in(std::function<void(const std::string& url, int port)> show,
                      std::chrono::seconds timeout = std::chrono::minutes(5)) -> Result<BrowserSignIn>;

// Opens `url` in the user's browser when this session runs on a local desktop (not over SSH); false otherwise.
auto open_in_browser(const std::string& url) -> bool;

// The page a browser shows after returning from a sign-in.
auto sign_in_done_page() -> std::string;

} // namespace kidi::agent::net
