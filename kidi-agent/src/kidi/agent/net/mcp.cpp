#include "kidi/agent/net/mcp.h"

#include "kidi/agent/net/oauth.h"
#include "kidi/agent/text.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace kidi::agent::net {
namespace {

using namespace std::chrono_literals;

constexpr auto PROTOCOL_VERSION = "2025-06-18";

auto now_seconds() -> std::int64_t {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// The JSON-RPC message answering `id`, from a JSON body or a server-sent event stream.
auto reply_for(const HttpResponse& response, std::uint64_t id) -> std::optional<Json> {
    const auto matches = [&](const Json& message) {
        return message.is_object() && message.contains("id") && message["id"] == id;
    };
    if (response.content_type.find("text/event-stream") == std::string::npos) {
        auto json = Json::parse(response.body, nullptr, false);
        if (json.is_array())
            for (auto& message : json)
                if (matches(message)) return message;
        return matches(json) ? std::optional<Json>(std::move(json)) : std::nullopt;
    }
    std::string data;
    std::size_t start = 0;
    while (start <= response.body.size()) {
        auto end = response.body.find('\n', start);
        if (end == std::string::npos) end = response.body.size();
        auto line = std::string_view(response.body).substr(start, end - start);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        start = end + 1;
        if (line.starts_with("data:")) {
            line.remove_prefix(5);
            if (!line.empty() && line.front() == ' ') line.remove_prefix(1);
            if (!data.empty()) data += '\n';
            data += line;
        } else if (line.empty() && !data.empty()) {
            auto message = Json::parse(data, nullptr, false);
            data.clear();
            if (matches(message)) return message;
        }
    }
    if (!data.empty())
        if (auto message = Json::parse(data, nullptr, false); matches(message)) return message;
    return std::nullopt;
}

} // namespace

McpServer::McpServer(McpServerOptions options) : options_(std::move(options)) {
    if (!options_.transport) options_.transport = curl_transport();
    if (!options_.credentials) options_.credentials = std::make_shared<CredentialStore>();
}

auto McpServer::create(McpServerOptions options) -> std::shared_ptr<McpServer> {
    return std::shared_ptr<McpServer>(new McpServer(std::move(options)));
}

auto McpServer::signed_in() const -> bool {
    return options_.credentials->get(options_.url).has_value();
}

auto McpServer::setup() -> std::shared_ptr<const ToolSetup> {
    auto self = shared_from_this();
    return std::make_shared<const ToolSetup>(ToolSetup{
        options_.service,
        [self]() -> std::optional<std::string> {
            if (self->signed_in()) return std::nullopt;
            return "sign in to " + self->options_.service + " once so the agent can use it";
        },
        [self](const BrowserSignIn& browser, std::stop_token stop) { return self->sign_in(browser, stop); }});
}

auto McpServer::sign_in(const BrowserSignIn& browser, std::stop_token stop) -> Result<void> {
    // An unauthenticated request draws the challenge that names the server's sign-in metadata.
    std::string challenge;
    if (auto probe = post({{"jsonrpc", "2.0"}, {"id", 0}, {"method", "ping"}}, {}, stop, 15s); probe) {
        if (const auto header = probe->headers.find("www-authenticate"); header != probe->headers.end())
            challenge = header->second;
    }
    auto server = discover_oauth(options_.transport, options_.url, challenge, stop);
    if (!server) return std::unexpected(server.error());
    auto client = register_oauth_client(options_.transport, *server, options_.client_name, browser.redirect_uri, stop);
    if (!client) return std::unexpected(client.error());
    auto pending = begin_sign_in(*server, *client, browser.redirect_uri);
    if (!pending) return std::unexpected(pending.error());
    auto callback = browser.visit(pending->authorize_url, stop);
    if (!callback) return std::unexpected(callback.error());
    auto token = finish_sign_in(options_.transport, *pending, *callback, stop);
    if (!token) return std::unexpected(token.error());
    if (auto saved = options_.credentials->put(options_.url, {{"client_id", *client},
                                                              {"token_endpoint", server->token_endpoint},
                                                              {"resource", server->resource},
                                                              {"access_token", token->access_token},
                                                              {"refresh_token", token->refresh_token},
                                                              {"expires_at", token->expires_at}});
        !saved)
        return std::unexpected(saved.error());
    const std::lock_guard lock(mutex_);
    initialized_ = false;
    session_id_.clear();
    return {};
}

auto McpServer::bearer(std::stop_token stop) -> Result<std::string> {
    auto saved = options_.credentials->get(options_.url);
    if (!saved) return failure(ErrorCode::UNSUPPORTED, "not signed in to " + options_.service);
    const auto expires_at = saved->value("expires_at", std::int64_t{0});
    const auto refresh = saved->value("refresh_token", "");
    if (expires_at && now_seconds() > expires_at - 60 && !refresh.empty()) {
        auto renewed = refresh_oauth_token(options_.transport, saved->value("token_endpoint", ""),
                                           saved->value("client_id", ""), saved->value("resource", ""), refresh,
                                           stop);
        if (!renewed) {
            if (renewed.error().code == ErrorCode::INVALID_ARGUMENT) {
                (void)options_.credentials->erase(options_.url);
                return failure(ErrorCode::UNSUPPORTED, "the " + options_.service +
                                                           " sign-in expired; the next call asks the user to sign in "
                                                           "again");
            }
            return std::unexpected(renewed.error());
        }
        (*saved)["access_token"] = renewed->access_token;
        (*saved)["refresh_token"] = renewed->refresh_token;
        (*saved)["expires_at"] = renewed->expires_at;
        if (auto stored = options_.credentials->put(options_.url, *saved); !stored)
            return std::unexpected(stored.error());
    }
    return saved->value("access_token", "");
}

auto McpServer::post(const Json& message, const std::string& token, std::stop_token stop,
                     std::chrono::seconds timeout) -> Result<HttpResponse> {
    HttpRequest request{.url = options_.url,
                        .headers = {"Content-Type: application/json", "Accept: application/json, text/event-stream"},
                        .body = message.dump(),
                        .timeout = timeout,
                        .max_bytes = 8 * 1024 * 1024};
    if (!token.empty()) request.headers.push_back("Authorization: Bearer " + token);
    {
        const std::lock_guard lock(mutex_);
        if (!protocol_version_.empty()) request.headers.push_back("MCP-Protocol-Version: " + protocol_version_);
        if (!session_id_.empty()) request.headers.push_back("Mcp-Session-Id: " + session_id_);
    }
    return options_.transport(request, stop);
}

auto McpServer::request(const std::string& method, Json params, std::stop_token stop, std::chrono::seconds timeout)
    -> Result<Json> {
    for (int attempt = 0;; ++attempt) {
        if (method != "initialize") {
            bool ready;
            {
                const std::lock_guard lock(mutex_);
                ready = initialized_;
            }
            if (!ready)
                if (auto started = initialize(stop); !started) return std::unexpected(started.error());
        }
        auto token = bearer(stop);
        if (!token) return std::unexpected(token.error());
        std::uint64_t id;
        {
            const std::lock_guard lock(mutex_);
            id = next_id_++;
        }
        auto response = post({{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}}, *token, stop,
                             timeout);
        if (!response) return std::unexpected(response.error());
        if (response->status == 401) {
            // The server refused the token: renew it once, else forget it so the next call asks to sign in again.
            auto saved = options_.credentials->get(options_.url);
            if (attempt == 0 && saved && !saved->value("refresh_token", "").empty()) {
                (*saved)["expires_at"] = 1;
                (void)options_.credentials->put(options_.url, *saved);
                continue;
            }
            (void)options_.credentials->erase(options_.url);
            return failure(ErrorCode::UNSUPPORTED, options_.service +
                                                       " no longer accepts the saved sign-in; the next call asks the "
                                                       "user to sign in again");
        }
        if (response->status == 404 && attempt == 0 && method != "initialize") {
            // The server forgot the session; start a new one.
            const std::lock_guard lock(mutex_);
            initialized_ = false;
            session_id_.clear();
            continue;
        }
        if (response->status == 429 || response->status >= 500)
            return failure(ErrorCode::IO, options_.service + " is busy or unavailable (HTTP " +
                                              std::to_string(response->status) + ")");
        if (response->status < 200 || response->status >= 300)
            return failure(ErrorCode::IO, options_.service + " answered HTTP " + std::to_string(response->status));
        if (method == "initialize")
            if (const auto session = response->headers.find("mcp-session-id"); session != response->headers.end()) {
                const std::lock_guard lock(mutex_);
                session_id_ = session->second;
            }
        auto reply = reply_for(*response, id);
        if (!reply) return failure(ErrorCode::IO, options_.service + " sent no reply to " + method);
        if (reply->contains("error")) {
            const auto& error = (*reply)["error"];
            const auto message = error.is_object() ? error.value("message", error.dump()) : error.dump();
            return failure(ErrorCode::RUNTIME, options_.service + " reported an error: " + message);
        }
        return reply->value("result", Json::object());
    }
}

auto McpServer::initialize(std::stop_token stop) -> Result<void> {
    auto result = request("initialize",
                          {{"protocolVersion", PROTOCOL_VERSION},
                           {"capabilities", Json::object()},
                           {"clientInfo", {{"name", "kidi-agent"}, {"version", "1.0"}}}},
                          stop, 20s);
    if (!result) return std::unexpected(result.error());
    {
        const std::lock_guard lock(mutex_);
        protocol_version_ = result->value("protocolVersion", PROTOCOL_VERSION);
        initialized_ = true;
    }
    auto token = bearer(stop);
    if (!token) return std::unexpected(token.error());
    auto notified = post({{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}}, *token, stop, 15s);
    if (!notified) return std::unexpected(notified.error());
    return {};
}

auto McpServer::list_tools(std::stop_token stop) -> Result<Json> {
    auto result = request("tools/list", Json::object(), stop, 30s);
    if (!result) return std::unexpected(result.error());
    return result->value("tools", Json::array());
}

auto McpServer::call_tool(const std::string& name, Json arguments, std::stop_token stop, std::chrono::seconds timeout)
    -> Result<McpToolResult> {
    auto result = request("tools/call", {{"name", name}, {"arguments", std::move(arguments)}}, stop, timeout);
    if (!result) return std::unexpected(result.error());
    McpToolResult output;
    output.is_error = result->value("isError", false);
    if (const auto content = result->find("content"); content != result->end() && content->is_array())
        for (const auto& block : *content)
            if (block.is_object() && block.value("type", "") == "text") {
                if (!output.text.empty()) output.text += '\n';
                output.text += block.value("text", "");
            }
    if (const auto structured = result->find("structuredContent"); structured != result->end())
        output.structured = *structured;
    return output;
}

auto mcp_servers(const std::map<std::string, McpServerConfig, std::less<>>& servers, McpServersOptions options)
    -> McpServers {
    if (!options.transport) options.transport = curl_transport();
    if (!options.credentials) options.credentials = std::make_shared<CredentialStore>();
    McpServers result;
    for (const auto& [key, server] : servers) {
        result.emplace(key, McpServer::create({.url = server.url,
                                               .service = server.name.empty() ? key : server.name,
                                               .transport = options.transport,
                                               .credentials = options.credentials}));
    }
    return result;
}

namespace {

constexpr std::size_t MAX_DESCRIPTION = 2000;
constexpr std::int64_t TOOL_LIST_LIFETIME = 12 * 3600; // seconds a cached tool list is used without asking
constexpr std::size_t MAX_SCHEMA_BYTES = 16 * 1024;
constexpr std::size_t MAX_OUTPUT = 32 * 1024;

auto valid_tool_name(std::string_view name) -> bool {
    return !name.empty() && name.size() <= 64 && std::ranges::all_of(name, [](unsigned char byte) {
        return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || (byte >= '0' && byte <= '9') ||
               byte == '_' || byte == '-';
    });
}

// One remote tool as an agent tool: network capability, the server's sign-in as its setup, bounded output.
auto remote_tool(ToolSpec spec, std::shared_ptr<McpServer> server, std::string name) -> Tool {
    auto setup = server->setup();
    return Tool{.spec = std::move(spec),
                .capability = Capability::NETWORK,
                .invoke = [server, name](const Json& args, std::stop_token stop) -> Result<std::string> {
                    auto reply = server->call_tool(name, args, stop, std::chrono::seconds(120));
                    if (!reply) return std::unexpected(reply.error());
                    auto text =
                        reply->text.empty() && !reply->structured.is_null() ? reply->structured.dump() : reply->text;
                    if (reply->is_error)
                        return failure(ErrorCode::RUNTIME,
                                       server->service() + " reported: " + std::string(utf8_prefix(text, 2000)));
                    if (text.size() > MAX_OUTPUT) {
                        text = std::string(utf8_prefix(text, MAX_OUTPUT)) + "\n[cut to 32 KB]";
                    }
                    return text;
                },
                .setup = std::move(setup)};
}

auto default_cache_path() -> std::filesystem::path {
    if (const auto* cache = std::getenv("XDG_CACHE_HOME"); cache && *cache)
        return std::filesystem::path(cache) / "kidi" / "mcp-tools.json";
    const auto* home = std::getenv("HOME");
    return std::filesystem::path(home && *home ? home : ".") / ".cache" / "kidi" / "mcp-tools.json";
}

} // namespace

auto mcp_tool_discovery(std::map<std::string, McpServerConfig, std::less<>> config, McpServers servers,
                        std::shared_ptr<CredentialStore> cache) -> ToolDiscovery {
    if (!cache) cache = std::make_shared<CredentialStore>(default_cache_path());
    return [config = std::move(config), servers = std::move(servers), cache](
               const DiscoveryPrompts& prompts, const std::vector<std::string>& taken_names,
               std::stop_token stop) -> std::vector<Tool> {
        const auto notice = [&](const std::string& text) {
            if (prompts.notice) prompts.notice(text);
        };
        std::vector<Tool> tools;
        std::vector<std::string> taken = taken_names;
        for (const auto& [key, server_config] : config) {
            if (!server_config.include_all && server_config.include.empty()) continue;
            const auto found = servers.find(key);
            if (found == servers.end()) continue;
            const auto& server = found->second;
            const auto& service = server->service();
            const auto before = tools.size();
            // The tool list comes from the cache while it is fresh; otherwise from the server, which must be
            // connected first.
            const auto cached = cache->get(server_config.url);
            const auto age = now_seconds() - (cached ? cached->value("listed_at", std::int64_t{0}) : 0);
            std::optional<Json> listed;
            if (cached && age >= 0 && age < TOOL_LIST_LIFETIME && cached->contains("tools"))
                listed = (*cached)["tools"];
            if (!listed && !server->signed_in()) {
                const auto setup = server->setup();
                if (!prompts.setup) {
                    notice(service + " tools are unavailable until it is connected; start an interactive kidi agent "
                                     "to connect it.");
                    continue;
                }
                auto browser = prompts.setup({.name = key, .description = service + " tools"}, *setup,
                                             "connect " + service + " to give the agent its tools");
                if (!browser || !*browser) {
                    notice(service + " is not connected, so its tools are off for this conversation.");
                    continue;
                }
                if (auto signed_in = setup->run(**browser, stop); !signed_in) {
                    notice("Connecting " + service + " failed: " + signed_in.error().message);
                    continue;
                }
            }
            if (!listed) {
                auto fresh = server->list_tools(stop);
                if (fresh) {
                    listed = *fresh;
                    if (auto saved = cache->put(server_config.url, {{"listed_at", now_seconds()}, {"tools", *fresh}});
                        !saved)
                        notice(saved.error().message);
                } else if (cached && cached->contains("tools")) {
                    listed = (*cached)["tools"];
                    notice("Could not refresh " + service + "'s tools (" + fresh.error().message +
                           "); using the list from " + std::to_string(age / 3600) + " hours ago.");
                } else {
                    notice("Could not load " + service + "'s tools: " + fresh.error().message);
                    continue;
                }
            }
            std::vector<std::string> seen;
            if (!listed->is_array()) {
                notice(service + " returned an unreadable tool list.");
                continue;
            }
            for (const auto& definition : *listed) {
                if (!definition.is_object() || !definition.contains("name") || !definition["name"].is_string()) {
                    continue;
                }
                const auto name = definition["name"].get<std::string>();
                const bool included = server_config.include_all ||
                                      std::ranges::find(server_config.include, name) != server_config.include.end();
                if (!included) continue;
                seen.push_back(name);
                const auto exposed = key + "_" + name;
                auto schema = definition.value("inputSchema", Json{{"type", "object"}});
                if (!valid_tool_name(name) || !valid_tool_name(exposed) || !schema.is_object() ||
                    schema.value("type", "") != "object" || schema.dump().size() > MAX_SCHEMA_BYTES) {
                    notice("Skipped " + service + " tool " + std::string(utf8_prefix(name, 64)) +
                           ": its name or parameter schema is invalid or too large.");
                    continue;
                }
                if (std::ranges::find(taken, exposed) != taken.end()) {
                    notice("Skipped " + service + " tool " + name + ": " + exposed + " is already a tool name.");
                    continue;
                }
                auto description = definition.value("description", std::string{});
                if (description.size() > MAX_DESCRIPTION) {
                    description = std::string(utf8_prefix(description, MAX_DESCRIPTION)) + "…";
                }
                taken.push_back(exposed);
                tools.push_back(remote_tool({.name = exposed,
                                             .description = description.empty() ? name + " (" + service + ")"
                                                                                : description,
                                             .parameters = std::move(schema)},
                                            server, name));
            }
            for (const auto& name : server_config.include)
                if (std::ranges::find(seen, name) == seen.end())
                    notice(service + " offers no tool named " + name + " (listed in include).");
            if (server_config.include_all)
                notice("Using all " + std::to_string(tools.size() - before) + " tools that " + service +
                       " offers (include: all); list only the ones you need to keep control.");
        }
        return tools;
    };
}

} // namespace kidi::agent::net
