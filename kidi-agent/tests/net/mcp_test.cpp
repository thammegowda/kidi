#include "kidi/agent/net/mcp.h"
#include "kidi/agent/net/sign_in.h"

#include <liteapp/core/base64.h>
#include <liteapp/core/sha256.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <set>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace kidi::agent::net {
namespace {

auto query_parameter(const std::string& url, const std::string& name) -> std::string {
    const auto at = url.find(name + "=");
    if (at == std::string::npos) return {};
    const auto start = at + name.size() + 1;
    return url.substr(start, url.find('&', start) - start);
}

// A scripted MCP server behind a scripted OAuth server, as TinyFish's are.
class McpSignIn : public testing::Test {
protected:
    static constexpr auto SERVER = "https://mcp.example.com/mcp";
    std::filesystem::path path_ = std::filesystem::temp_directory_path() /
                                  ("kidi-mcp-test-" + std::to_string(::getpid()) + ".json");
    std::shared_ptr<CredentialStore> store_ = std::make_shared<CredentialStore>(path_);
    std::set<std::string> valid_tokens_{"access-1"};
    std::string challenge_;
    std::vector<std::string> redirects_;
    int token_requests_ = 0;
    bool refresh_works_ = true;
    Json listed_ = Json::array({{{"name", "search"}}});
    int lists_ = 0;
    std::shared_ptr<CredentialStore> cache_ = std::make_shared<CredentialStore>(path_.string() + ".cache");
    std::shared_ptr<McpServer> server_;

    auto SetUp() -> void override {
        std::filesystem::remove(path_);
        std::filesystem::remove(path_.string() + ".cache");
        server_ = McpServer::create({.url = SERVER,
                                     .service = "Example",
                                     .transport = [this](const HttpRequest& request, std::stop_token) {
                                         return answer(request);
                                     },
                                     .credentials = store_});
    }
    auto TearDown() -> void override {
        std::filesystem::remove(path_);
        std::filesystem::remove(path_.string() + ".cache");
    }

    auto json(int status, Json body) -> HttpResponse {
        return {.status = status, .content_type = "application/json", .body = body.dump()};
    }

    auto answer(const HttpRequest& request) -> Result<HttpResponse> {
        if (request.url == SERVER) {
            const auto authorization = std::ranges::find_if(request.headers, [](const std::string& header) {
                return header.starts_with("Authorization: Bearer ");
            });
            if (authorization == request.headers.end() || !valid_tokens_.contains(authorization->substr(22))) {
                auto refused = json(401, {{"error", "unauthorized"}});
                refused.headers["www-authenticate"] =
                    R"(Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource/mcp")";
                return refused;
            }
            const auto message = Json::parse(*request.body);
            if (!message.contains("id")) return HttpResponse{.status = 202};
            const auto method = message["method"].get<std::string>();
            if (method == "tools/list") ++lists_;
            Json result = method == "initialize" ? Json{{"protocolVersion", "2025-06-18"}}
                          : method == "tools/list" ? Json{{"tools", listed_}}
                                                   : Json{{"content", Json::array({{{"type", "text"},
                                                                                    {"text", "found it"}}})},
                                                          {"structuredContent", {{"hits", 1}}}};
            // Tool calls answer as a server-sent event stream.
            const auto body = Json{{"jsonrpc", "2.0"}, {"id", message["id"]}, {"result", result}}.dump();
            if (method == "tools/call")
                return HttpResponse{.status = 200,
                                    .content_type = "text/event-stream",
                                    .body = "event: message\ndata: " + body + "\n\n"};
            auto response = HttpResponse{.status = 200, .content_type = "application/json", .body = body};
            response.headers["mcp-session-id"] = "session-7";
            return response;
        }
        if (request.url == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp")
            return json(200, {{"resource", SERVER}, {"authorization_servers", {"https://auth.example.com"}}});
        if (request.url == "https://auth.example.com/.well-known/oauth-authorization-server")
            return json(200, {{"issuer", "https://auth.example.com"},
                              {"authorization_endpoint", "https://auth.example.com/authorize"},
                              {"token_endpoint", "https://auth.example.com/token"},
                              {"registration_endpoint", "https://auth.example.com/register"},
                              {"code_challenge_methods_supported", {"S256"}},
                              {"scopes_supported", {"openid", "email", "offline_access", "admin"}}});
        if (request.url == "https://auth.example.com/register") {
            const auto body = Json::parse(*request.body);
            redirects_.push_back(body["redirect_uris"][0]);
            EXPECT_EQ(body["token_endpoint_auth_method"], "none");
            return json(201, {{"client_id", "client-1"}});
        }
        if (request.url == "https://auth.example.com/token") {
            ++token_requests_;
            const auto& form = *request.body;
            if (form.find("grant_type=authorization_code") != std::string::npos) {
                // PKCE: the verifier must hash to the challenge sent with the sign-in.
                const auto digest = liteapp::sha256(query_parameter(form, "code_verifier"));
                EXPECT_EQ(liteapp::base64_encode(std::span<const std::byte>(digest), liteapp::Base64::url),
                          challenge_);
                EXPECT_EQ(query_parameter(form, "code"), "code-1");
                EXPECT_EQ(query_parameter(form, "resource"), "https%3A%2F%2Fmcp.example.com%2Fmcp");
                return json(200, {{"access_token", "access-1"}, {"refresh_token", "refresh-1"}, {"expires_in", 3600}});
            }
            if (!refresh_works_) return json(400, {{"error", "invalid_grant"}});
            valid_tokens_.insert("access-2");
            return json(200, {{"access_token", "access-2"}, {"expires_in", 3600}});
        }
        return std::unexpected(Error{ErrorCode::IO, "unexpected " + request.url});
    }

    auto browser() -> BrowserSignIn {
        return {"http://127.0.0.1:4321/oauth/callback", [this](const std::string& url, std::stop_token) {
                    EXPECT_TRUE(url.starts_with("https://auth.example.com/authorize?"));
                    EXPECT_EQ(query_parameter(url, "scope"), "openid%20email%20offline_access");
                    challenge_ = query_parameter(url, "code_challenge");
                    return Result<std::string>("http://127.0.0.1:4321/oauth/callback?code=code-1&state=" +
                                               query_parameter(url, "state"));
                }};
    }
};

TEST_F(McpSignIn, SignsInThroughTheBrowserAndCallsTools) {
    const auto setup = server_->setup();
    ASSERT_TRUE(setup->needed());
    EXPECT_FALSE(server_->call_tool("search", {{"query", "x"}}, {})) << "no call before signing in";
    ASSERT_TRUE(setup->run(browser(), {}));
    EXPECT_FALSE(setup->needed());
    EXPECT_EQ(redirects_, std::vector<std::string>{"http://127.0.0.1:4321/oauth/callback"});
    const auto saved = store_->get(SERVER);
    ASSERT_TRUE(saved);
    EXPECT_EQ((*saved)["refresh_token"], "refresh-1");
    EXPECT_EQ(std::filesystem::status(path_).permissions() & std::filesystem::perms::all,
              std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);

    auto tools = server_->list_tools({});
    ASSERT_TRUE(tools) << tools.error().message;
    EXPECT_EQ((*tools)[0]["name"], "search");
    auto result = server_->call_tool("search", {{"query", "kidi"}}, {});
    ASSERT_TRUE(result) << result.error().message;
    EXPECT_EQ(result->text, "found it");
    EXPECT_EQ(result->structured["hits"], 1);
}

TEST_F(McpSignIn, RenewsExpiredTokensAndForgetsRevokedOnes) {
    ASSERT_TRUE(server_->sign_in(browser(), {}));
    auto saved = *store_->get(SERVER);
    saved["expires_at"] = 1;
    ASSERT_TRUE(store_->put(SERVER, saved));
    ASSERT_TRUE(server_->call_tool("search", Json::object(), {}));
    EXPECT_EQ((*store_->get(SERVER))["access_token"], "access-2");
    EXPECT_EQ((*store_->get(SERVER))["refresh_token"], "refresh-1") << "a refresh token that was not rotated stays";

    // The server revokes the sign-in and renewing fails: it is forgotten, so the next call asks to sign in.
    valid_tokens_.clear();
    refresh_works_ = false;
    auto refused = server_->call_tool("search", Json::object(), {});
    ASSERT_FALSE(refused);
    EXPECT_NE(refused.error().message.find("sign in again"), std::string::npos);
    EXPECT_FALSE(store_->get(SERVER));
    EXPECT_TRUE(server_->setup()->needed());
}

TEST_F(McpSignIn, RejectsForeignSignInReplies) {
    BrowserSignIn forged{"http://127.0.0.1:4321/oauth/callback", [](const std::string&, std::stop_token) {
                             return Result<std::string>("http://127.0.0.1:4321/oauth/callback?code=x&state=other");
                         }};
    auto signed_in = server_->sign_in(forged, {});
    ASSERT_FALSE(signed_in);
    EXPECT_NE(signed_in.error().message.find("does not belong to this sign-in"), std::string::npos);
    BrowserSignIn denied{"http://127.0.0.1:4321/oauth/callback", [](const std::string& url, std::stop_token) {
                             return Result<std::string>("?error=access_denied&error_description=User+cancelled&state=" +
                                                        query_parameter(url, "state"));
                         }};
    auto cancelled = server_->sign_in(denied, {});
    ASSERT_FALSE(cancelled);
    EXPECT_NE(cancelled.error().message.find("User cancelled"), std::string::npos);
    EXPECT_FALSE(store_->get(SERVER));
}

TEST_F(McpSignIn, DiscoversExactlyTheIncludedTools) {
    const auto schema = Json{{"type", "object"}, {"properties", {{"q", {{"type", "string"}}}}}};
    listed_ = Json::array({{{"name", "search"}, {"description", "Search."}, {"inputSchema", schema}},
                           {{"name", "fetch"}, {"description", "Fetch."}, {"inputSchema", schema}},
                           {{"name", "pay"}, {"description", "Spend money."}, {"inputSchema", schema}},
                           {{"name", "bad name!"}, {"inputSchema", schema}}});
    McpServerConfig config{"Example", SERVER, {"search", "fetch", "missing"}, false};
    std::vector<std::string> notices;
    int setups = 0;
    DiscoveryPrompts prompts{
        .setup = [&](const ToolSpec& spec, const ToolSetup& setup,
                     const std::string&) -> Result<std::optional<BrowserSignIn>> {
            ++setups;
            EXPECT_EQ(spec.name, "example");
            EXPECT_EQ(setup.service, "Example");
            return std::optional<BrowserSignIn>(browser());
        },
        .notice = [&](const std::string& text) { notices.push_back(text); }};
    const auto names = [](const std::vector<Tool>& tools) {
        std::vector<std::string> result;
        for (const auto& tool : tools) result.push_back(tool.spec.name);
        return result;
    };

    // First use: the user connects, then exactly the included tools appear.
    const auto discover = [&] { return mcp_tool_discovery({{"example", config}}, {{"example", server_}}, cache_); };
    auto tools = discover()(prompts, {"example_fetch"}, {});
    EXPECT_EQ(setups, 1);
    EXPECT_EQ(names(tools), std::vector<std::string>{"example_search"}) << "a taken name is skipped";
    EXPECT_EQ(tools[0].spec.description, "Search.");
    EXPECT_EQ(tools[0].spec.parameters, schema);
    EXPECT_EQ(tools[0].capability, Capability::NETWORK);
    auto output = tools[0].invoke({{"q", "kidi"}}, {});
    ASSERT_TRUE(output) << output.error().message;
    EXPECT_EQ(*output, "found it");
    const auto said = [&](std::string_view text) {
        return std::ranges::any_of(notices,
                                   [&](const std::string& notice) { return notice.find(text) != notice.npos; });
    };
    EXPECT_TRUE(said("example_fetch is already a tool name"));
    EXPECT_TRUE(said("offers no tool named missing"));

    // include: all exposes every valid tool and says so.
    config.include = {};
    config.include_all = true;
    notices.clear();
    tools = discover()(prompts, {}, {});
    EXPECT_EQ(setups, 1) << "a signed-in server needs no new sign-in";
    EXPECT_EQ(names(tools), (std::vector<std::string>{"example_search", "example_fetch", "example_pay"}));
    EXPECT_TRUE(said("Using all 3 tools that Example offers"));
}

TEST_F(McpSignIn, DiscoverySkipsServersTheUserDoesNotConnect) {
    std::vector<std::string> notices;
    DiscoveryPrompts declining{
        .setup = [](const ToolSpec&, const ToolSetup&, const std::string&) -> Result<std::optional<BrowserSignIn>> {
            return std::optional<BrowserSignIn>{};
        },
        .notice = [&](const std::string& text) { notices.push_back(text); }};
    const McpServerConfig config{"Example", SERVER, {"search"}, false};
    EXPECT_TRUE(mcp_tool_discovery({{"example", config}}, {{"example", server_}}, cache_)(declining, {}, {})
                    .empty());
    ASSERT_EQ(notices.size(), 1);
    EXPECT_NE(notices[0].find("not connected"), std::string::npos);
    const McpServerConfig silent{"Example", SERVER, {}, false};
    EXPECT_TRUE(mcp_tool_discovery({{"example", silent}}, {{"example", server_}}, cache_)(declining, {}, {})
                    .empty());
    EXPECT_EQ(notices.size(), 1) << "a server without include is never offered";
}

TEST_F(McpSignIn, CachesToolListsForTwelveHours) {
    const McpServerConfig config{"Example", SERVER, {"search"}, false};
    std::vector<std::string> notices;
    int setups = 0;
    DiscoveryPrompts prompts{
        .setup = [&](const ToolSpec&, const ToolSetup&, const std::string&) -> Result<std::optional<BrowserSignIn>> {
            ++setups;
            return std::optional<BrowserSignIn>(browser());
        },
        .notice = [&](const std::string& text) { notices.push_back(text); }};
    const auto discover = mcp_tool_discovery({{"example", config}}, {{"example", server_}}, cache_);
    ASSERT_EQ(discover(prompts, {}, {}).size(), 1);
    EXPECT_EQ(lists_, 1);
    ASSERT_EQ(discover(prompts, {}, {}).size(), 1);
    EXPECT_EQ(lists_, 1) << "a fresh cache is not listed again";

    // A cached list needs no sign-in to start a conversation; the tool asks to connect when first called.
    ASSERT_TRUE(store_->erase(SERVER));
    auto cached = discover(prompts, {}, {});
    ASSERT_EQ(cached.size(), 1);
    EXPECT_EQ(setups, 1);
    ASSERT_TRUE(cached[0].setup);
    EXPECT_TRUE(cached[0].setup->needed());
    ASSERT_TRUE(cached[0].setup->run(browser(), {}));

    // After twelve hours the list is fetched again; if that fails, the old list is used and the user told.
    auto entry = *cache_->get(SERVER);
    entry["listed_at"] = entry["listed_at"].get<std::int64_t>() - 12 * 3600 - 1;
    ASSERT_TRUE(cache_->put(SERVER, entry));
    ASSERT_EQ(discover(prompts, {}, {}).size(), 1);
    EXPECT_EQ(lists_, 2);
    entry = *cache_->get(SERVER);
    entry["listed_at"] = entry["listed_at"].get<std::int64_t>() - 13 * 3600;
    ASSERT_TRUE(cache_->put(SERVER, entry));
    valid_tokens_.clear();
    refresh_works_ = false;
    ASSERT_EQ(discover(prompts, {}, {}).size(), 1);
    EXPECT_NE(notices.back().find("using the list from 13 hours ago"), std::string::npos);
}

// Every agent process opens its own store on the same file; concurrent sign-ins must not overwrite each other.
TEST(CredentialStore, ConcurrentWritersKeepEveryEntry) {
    const auto path = std::filesystem::temp_directory_path() / ("kidi-credentials-" + std::to_string(::getpid()));
    std::filesystem::remove_all(path);
    const auto file = path / "credentials.json";
    {
        std::vector<std::jthread> writers;
        for (int writer = 0; writer < 4; ++writer)
            writers.emplace_back([&file, writer] {
                CredentialStore store(file);
                for (int index = 0; index < 25; ++index)
                    ASSERT_TRUE(store.put(std::to_string(writer) + "-" + std::to_string(index), {{"token", index}}));
            });
    }
    const CredentialStore store(file);
    for (int writer = 0; writer < 4; ++writer)
        for (int index = 0; index < 25; ++index)
            EXPECT_TRUE(store.get(std::to_string(writer) + "-" + std::to_string(index))) << writer << "-" << index;
    std::size_t leftovers = 0;
    for (const auto& entry : std::filesystem::directory_iterator(path))
        leftovers += entry.path().filename() != "credentials.json" &&
                     entry.path().filename() != "credentials.json.lock";
    EXPECT_EQ(leftovers, 0) << "no temporary files are left behind";
    std::filesystem::remove_all(path);
}

// The terminal's sign-in: the browser returns to a temporary listener on 127.0.0.1, which needs no cookie.
TEST(LoopbackSignIn, ReceivesTheBrowsersReturn) {
    int shown_port = 0;
    auto sign_in = loopback_sign_in([&](const std::string&, int port) { shown_port = port; }, std::chrono::seconds(10));
    ASSERT_TRUE(sign_in) << sign_in.error().message;
    const auto port = std::stoi(sign_in->redirect_uri.substr(sign_in->redirect_uri.rfind(':') + 1));
    EXPECT_EQ(sign_in->redirect_uri, "http://127.0.0.1:" + std::to_string(port) + "/oauth/callback");
    const auto get = [port](const std::string& target) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<std::uint16_t>(port));
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        std::string reply;
        if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
            const auto request = "GET " + target + " HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
            (void)::send(fd, request.data(), request.size(), 0);
            char buffer[512];
            for (ssize_t count; (count = ::recv(fd, buffer, sizeof(buffer), 0)) > 0;)
                reply.append(buffer, static_cast<std::size_t>(count));
        }
        ::close(fd);
        return reply;
    };
    std::string favicon, returned;
    std::jthread browser([&] {
        favicon = get("/favicon.ico");
        returned = get("/oauth/callback?code=abc&state=xyz");
    });
    auto callback = sign_in->visit("https://auth.example.com/authorize", {});
    browser.join();
    ASSERT_TRUE(callback) << callback.error().message;
    EXPECT_EQ(*callback, "http://127.0.0.1:" + std::to_string(port) + "/oauth/callback?code=abc&state=xyz");
    EXPECT_EQ(shown_port, port);
    EXPECT_TRUE(favicon.starts_with("HTTP/1.1 404")) << "other requests do not end the wait";
    EXPECT_TRUE(returned.starts_with("HTTP/1.1 200")) << returned;
    EXPECT_NE(returned.find("Signed in"), std::string::npos);
}

} // namespace
} // namespace kidi::agent::net
