#pragma once

#include "kidi/agent/agent.h"
#include "kidi/agent/config.h"
#include "kidi/agent/net/credentials.h"
#include "kidi/agent/net/http.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>

namespace kidi::agent::net {

struct McpToolResult {
    std::string text;               // the text content blocks, joined
    Json structured;                // structuredContent, when the server sent it
    bool is_error = false;          // the tool reported failure
};

struct McpServerOptions {
    std::string url;                // the server's MCP endpoint (Streamable HTTP)
    std::string service;            // the name users know, such as "TinyFish"
    HttpTransport transport;        // default: curl_transport()
    std::shared_ptr<CredentialStore> credentials; // default: the user's credential store
    std::string client_name = "Kidi Agent";
};

// A remote MCP server over Streamable HTTP. It signs in with OAuth through the user's browser (see setup()),
// keeps and renews the tokens in the credential store, and lists and calls the server's tools.
class McpServer : public std::enable_shared_from_this<McpServer> {
public:
    static auto create(McpServerOptions options) -> std::shared_ptr<McpServer>;

    // The sign-in step for tools that use this server; it is needed until the user has signed in.
    auto setup() -> std::shared_ptr<const ToolSetup>;
    auto signed_in() const -> bool;
    auto sign_in(const BrowserSignIn& browser, std::stop_token stop) -> Result<void>;
    auto list_tools(std::stop_token stop) -> Result<Json>;
    auto call_tool(const std::string& name, Json arguments, std::stop_token stop,
                   std::chrono::seconds timeout = std::chrono::seconds(60)) -> Result<McpToolResult>;
    auto service() const -> const std::string& { return options_.service; }

private:
    explicit McpServer(McpServerOptions options);
    auto bearer(std::stop_token stop) -> Result<std::string>;
    auto post(const Json& message, const std::string& token, std::stop_token stop, std::chrono::seconds timeout)
        -> Result<HttpResponse>;
    auto request(const std::string& method, Json params, std::stop_token stop, std::chrono::seconds timeout)
        -> Result<Json>;
    auto initialize(std::stop_token stop) -> Result<void>;

    McpServerOptions options_;
    std::mutex mutex_;
    std::string session_id_;
    std::string protocol_version_;
    bool initialized_ = false;
    std::uint64_t next_id_ = 1;
};

// The agent's MCP servers by their agent.yaml key. Connections and sign-ins happen on first use.
using McpServers = std::map<std::string, std::shared_ptr<McpServer>, std::less<>>;

struct McpServersOptions {
    HttpTransport transport;                       // default: curl_transport()
    std::shared_ptr<CredentialStore> credentials;  // default: the user's credential store
};
auto mcp_servers(const std::map<std::string, McpServerConfig, std::less<>>& servers, McpServersOptions options = {})
    -> McpServers;

// How tool discovery talks to the user, through the session's front end.
struct DiscoveryPrompts {
    // Asks to connect (sign in to) a server; nullopt when the user declines. See Callbacks::setup.
    std::function<Result<std::optional<BrowserSignIn>>(const ToolSpec&, const ToolSetup&, const std::string& reason)>
        setup;
    std::function<void(const std::string& text)> notice;
};

// Discovers the tools of the mcp_servers entries that have an `include`, once per conversation. A server's tools are
// listed from the server itself and exposed as <server>_<tool>, with the server's descriptions and schemas, limited
// to `include` (exact names, or all). Servers that are not signed in are offered for connection first.
using ToolDiscovery = std::function<std::vector<Tool>(const DiscoveryPrompts& prompts,
                                                      const std::vector<std::string>& taken_names,
                                                      std::stop_token stop)>;
// Tool lists are cached for 12 hours, so conversations usually start without asking the server (and without
// signing in: a cached tool asks to connect when first called). `cache` defaults to $XDG_CACHE_HOME/kidi/
// mcp-tools.json, else ~/.cache/kidi/mcp-tools.json.
auto mcp_tool_discovery(std::map<std::string, McpServerConfig, std::less<>> config, McpServers servers,
                        std::shared_ptr<CredentialStore> cache = {}) -> ToolDiscovery;

} // namespace kidi::agent::net
