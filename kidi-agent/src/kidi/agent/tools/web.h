#pragma once

#include "kidi/agent/config.h"
#include "kidi/agent/net/http.h"
#include "kidi/agent/net/mcp.h"

#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace kidi::agent::tools {


struct WebToolOptions {
    net::McpServers servers;                                                           // web_search providers, by name
    net::HttpTransport transport;                                                      // default: curl_transport()
    std::function<Result<std::vector<std::string>>(const std::string&)> resolve; // default: getaddrinfo
};

// Handlers for the web_search and web_fetch tools (Capability::NETWORK), for configure_tools.
//   web_search: config: {server: tinyfish} names an mcp_servers entry and calls its `search` tool with {query}
//               (`tool: NAME` picks another); the user signs in to the server the first time it is needed.
//   web_fetch:  reads public pages itself as compact Markdown, and only addresses that already appeared in a user
//               message or tool result.
// Network failures are retried once; every failure tells the model what went wrong and what the user can do.
auto web_tool_handlers(WebToolOptions options = {}) -> std::vector<ToolHandler>;

// Whether a public http(s) address: loopback, private, link-local, carrier-grade NAT and multicast ranges are not.
auto public_address(std::string_view address) -> bool;

} // namespace kidi::agent::tools
