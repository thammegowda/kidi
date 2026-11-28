#include "kidi/agent/tools/html.h"
#include "kidi/agent/tools/web.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <map>

#include <unistd.h>

namespace kidi::agent::tools {
namespace {

TEST(HtmlToMarkdown, KeepsContentStructureAndDropsChrome) {
    const auto page = html_to_markdown(R"(<!doctype html><html><head><title>Rust &amp; You</title>
        <style>body { color: red }</style><script>var tracking = 1;</script></head>
        <body><nav><a href="/">Home</a> <a href="/blog">Blog</a></nav>
        <div class="cookie-banner">We use cookies. <button>Accept</button></div>
        <main><h1>Async   fn in <em>traits</em></h1>
        <p>Starting with Rust&nbsp;1.75, you can use <code>async fn</code> in
           <a href="../docs/traits.html?utm_source=x&amp;page=2#top">traits</a>.</p>
        <ul><li>Stable</li><li>Fast<ol><li>first</li><li>second</li></ol></li></ul>
        <pre><code class="language-rust">trait T {
    async fn f(&amp;self);
}</code></pre>
        <table><tr><th>Version</th><th>Date</th></tr><tr><td>1.75</td><td>2023</td></tr></table>
        <div hidden>secret</div><div style="display: none">invisible</div>
        <img src="x.png" alt="diagram"><div class="share-buttons">Share on X</div>
        </main><footer>Copyright</footer></body></html>)",
                                       "https://blog.example.org/posts/async.html");
    EXPECT_EQ(page.title, "Rust & You");
    EXPECT_EQ(page.markdown,
              "# Async fn in traits\n"
              "\n"
              "Starting with Rust 1.75, you can use `async fn` in "
              "[traits](/docs/traits.html?page=2).\n"
              "\n"
              "- Stable\n"
              "- Fast\n"
              "  1. first\n"
              "  2. second\n"
              "\n"
              "```rust\n"
              "trait T {\n"
              "    async fn f(&self);\n"
              "}\n"
              "```\n"
              "\n"
              "| Version | Date |\n"
              "| --- | --- |\n"
              "| 1.75 | 2023 |");
}

TEST(HtmlToMarkdown, KeepsContentWhenChromeWordsMarkTheWholePage) {
    std::string paragraphs;
    for (int index = 0; index < 20; ++index)
        paragraphs += "<p>Paragraph " + std::to_string(index) + " of the story.</p>";
    const auto page = html_to_markdown("<body><div class=\"layout has-sidebar\">" + paragraphs + "</div></body>",
                                       "https://example.com/");
    EXPECT_NE(page.markdown.find("Paragraph 19 of the story."), std::string::npos);
}

TEST(HtmlToMarkdown, ResolvesLinksAndConvertsText) {
    EXPECT_EQ(resolve_url("https://a.com/x/y/z.html", "../b/c"), "https://a.com/x/b/c");
    EXPECT_EQ(resolve_url("https://a.com/x/y", "//cdn.b.com/f"), "https://cdn.b.com/f");
    EXPECT_EQ(resolve_url("https://a.com/x/y?q=1", "?p=2"), "https://a.com/x/y?p=2");
    EXPECT_EQ(resolve_url("https://a.com/x/", "/root?utm_medium=m"), "https://a.com/root");
    EXPECT_EQ(resolve_url("http://a.com", "HTTPS://b.com/Path#frag"), "https://b.com/Path");
    for (const auto* skipped : {"#top", "javascript:void(0)", "mailto:a@b.com", "data:text/plain,x", ""})
        EXPECT_EQ(resolve_url("https://a.com/", skipped), "") << skipped;
    EXPECT_EQ(to_utf8("caf\xe9 \x93quoted\x94", "ISO-8859-1"), "café “quoted”");
    EXPECT_EQ(to_utf8("ok \xff end", "utf-8"), "ok \xef\xbf\xbd end");
}

TEST(WebAddresses, AcceptsOnlyPublicAddresses) {
    for (const auto* address : {"93.184.216.34", "8.8.8.8", "2606:4700::1111", "[2001:4860:4860::8888]"})
        EXPECT_TRUE(public_address(address)) << address;
    for (const auto* address : {"127.0.0.1", "10.1.2.3", "172.16.0.9", "192.168.1.1", "169.254.169.254", "100.64.0.1",
                                "0.0.0.0", "224.0.0.1", "::1", "::", "fe80::1", "fd00::5", "::ffff:127.0.0.1",
                                "64:ff9b::a00:1", "localhost", "not-an-address"})
        EXPECT_FALSE(public_address(address)) << address;
}

// web_search and web_fetch over a scripted transport and a scripted TinyFish MCP server: no network, no waiting.
class WebTools : public testing::Test {
protected:
    struct Reply {
        int status = 200;
        std::string content_type = "text/html; charset=utf-8";
        std::string body;
        std::string redirect;
        std::optional<Error> error;
    };
    // A scripted tools/call answer: tool text (and isError), or an HTTP status, or a transport failure.
    struct ToolReply {
        std::string text;
        bool is_error = false;
        int status = 200;
        std::optional<Error> error;
    };
    static constexpr auto MCP = "https://agent.tinyfish.ai/mcp";

    std::map<std::string, std::vector<Reply>> replies_; // url -> replies, consumed in order (the last repeats)
    std::map<std::string, std::vector<ToolReply>> tool_replies_; // MCP tool -> replies, likewise
    std::vector<net::HttpRequest> requests_;
    std::vector<Json> tool_calls_; // tools/call params, in order
    std::map<std::string, std::string> addresses_{{"news.example.com", "93.184.216.34"},
                                                   {"intranet.example.com", "10.0.0.7"}};
    std::filesystem::path credentials_ = std::filesystem::temp_directory_path() /
                                         ("kidi-web-tools-" + std::to_string(::getpid()) + ".json");
    std::vector<Tool> tools_;

    template <class Item> static auto next(std::vector<Item>& queue) -> Item {
        auto item = queue.front();
        if (queue.size() > 1) queue.erase(queue.begin());
        return item;
    }

    auto mcp(const net::HttpRequest& request) -> Result<net::HttpResponse> {
        const auto message = Json::parse(*request.body);
        const auto method = message.value("method", "");
        if (!message.contains("id")) return net::HttpResponse{.status = 202};
        Json result = Json::object();
        if (method == "initialize") {
            result = {{"protocolVersion", "2025-06-18"}, {"capabilities", Json::object()}};
        } else if (method == "tools/call") {
            tool_calls_.push_back(message["params"]);
            auto& queue = tool_replies_[message["params"]["name"].get<std::string>()];
            if (queue.empty()) return std::unexpected(Error{ErrorCode::IO, "no scripted reply"});
            const auto reply = next(queue);
            if (reply.error) return std::unexpected(*reply.error);
            if (reply.status != 200) return net::HttpResponse{.status = reply.status};
            result = {{"content", Json::array({{{"type", "text"}, {"text", reply.text}}})},
                      {"isError", reply.is_error}};
        }
        return net::HttpResponse{.status = 200,
                            .content_type = "application/json",
                            .body = Json{{"jsonrpc", "2.0"}, {"id", message["id"]}, {"result", result}}.dump(),
                            .headers = {{"mcp-session-id", "session-1"}}};
    }

    auto build() -> void {
        tools_.clear();
        const net::HttpTransport transport = [this](const net::HttpRequest& request,
                                                    std::stop_token) -> Result<net::HttpResponse> {
            requests_.push_back(request);
            if (request.url == MCP) return mcp(request);
            auto key = request.url.substr(0, request.url.find('?'));
            auto found = replies_.find(key);
            if (found == replies_.end()) return std::unexpected(Error{ErrorCode::IO, "unexpected " + key});
            auto reply = next(found->second);
            if (reply.error) return std::unexpected(*reply.error);
            return net::HttpResponse{reply.status, reply.content_type, reply.redirect, reply.body};
        };
        auto servers = net::mcp_servers({{"tinyfish", {"TinyFish", MCP}}},
                                   {.transport = transport,
                                    .credentials = std::make_shared<net::CredentialStore>(credentials_)});
        auto handlers = web_tool_handlers(
            {.servers = servers,
             .transport = transport,
             .resolve = [this](const std::string& host) -> Result<std::vector<std::string>> {
                 const auto address = addresses_.find(host);
                 if (address == addresses_.end())
                     return std::unexpected(Error{ErrorCode::IO, "could not resolve " + host});
                 return std::vector<std::string>{address->second};
             }});
        for (auto& handler : handlers) {
            auto tool = handler.make(handler.name == "web_search" ? Json{{"server", "tinyfish"}} : Json::object());
            ASSERT_TRUE(tool) << tool.error().message;
            tool->spec.name = handler.name;
            tools_.push_back(std::move(*tool));
        }
    }

    // Signed in to TinyFish, as after the browser sign-in.
    auto SetUp() -> void override {
        std::filesystem::remove(credentials_);
        ASSERT_TRUE(net::CredentialStore(credentials_).put(MCP, {{"access_token", "test-key"}}));
        build();
    }
    auto TearDown() -> void override { std::filesystem::remove(credentials_); }

    auto tool(std::string_view name) -> Tool& {
        return *std::ranges::find_if(tools_, [&](const Tool& item) { return item.spec.name == name; });
    }
    auto call(std::string_view name, Json args) -> Result<std::string> { return tool(name).invoke(args, {}); }

    auto requested(std::string_view url) const -> std::size_t {
        return static_cast<std::size_t>(std::ranges::count_if(requests_, [&](const net::HttpRequest& request) {
            return request.url.starts_with(url);
        }));
    }
};

TEST(WebToolConfig, NamesAnMcpServer) {
    const auto servers = net::mcp_servers({{"tinyfish", {"TinyFish", "https://x.test/mcp"}}});
    const auto handlers = web_tool_handlers({.servers = servers});
    const auto& search = handlers[0];
    const auto& fetch = handlers[1];
    ASSERT_EQ(search.name, "web_search");
    ASSERT_EQ(fetch.name, "web_fetch");
    auto made = search.make({{"server", "tinyfish"}});
    ASSERT_TRUE(made) << made.error().message;
    ASSERT_TRUE(made->setup) << "web_search signs in to its server";
    EXPECT_EQ(made->setup->service, "TinyFish");
    EXPECT_TRUE(search.make({{"server", "tinyfish"}, {"tool", "search_v2"}}));
    EXPECT_TRUE(fetch.make(Json::object()));
    for (const auto& bad : {Json::object(), Json{{"server", "bing"}}, Json{{"server", "tinyfish"}, {"timeout", "3"}},
                            Json{{"provider", "tinyfish"}}}) {
        const auto rejected = search.make(bad);
        ASSERT_FALSE(rejected) << bad.dump();
        EXPECT_FALSE(rejected.error().message.empty());
    }
    EXPECT_NE(search.make({{"server", "bing"}}).error().message.find("define it under mcp_servers"), std::string::npos);
}

TEST_F(WebTools, SearchListsCompactResults) {
    tool_replies_["search"] = {{.text = R"({"query":"q","results":[
                                   {"position":1,"title":"First\n result","url":"https://a.com/1","snippet":"One."},
                                   {"position":2,"title":"Second","url":"https://b.com/2","snippet":"Two."},
                                   {"position":3,"title":"Third","url":"https://c.com/3"}]})"}};
    auto result = call("web_search", {{"query", "rust async & traits"}, {"count", 2}});
    ASSERT_TRUE(result) << result.error().message;
    EXPECT_EQ(*result, "Web results for \"rust async & traits\" (read one with web_fetch):\n"
                       "1. First result\n   https://a.com/1\n   One.\n"
                       "2. Second\n   https://b.com/2\n   Two.\n");
    ASSERT_EQ(tool_calls_.size(), 1);
    EXPECT_EQ(tool_calls_[0], (Json{{"name", "search"}, {"arguments", {{"query", "rust async & traits"}}}}));
    EXPECT_NE(std::ranges::find(requests_.back().headers, "Authorization: Bearer test-key"),
              requests_.back().headers.end());
    EXPECT_NE(std::ranges::find(requests_.back().headers, "Mcp-Session-Id: session-1"), requests_.back().headers.end());
    EXPECT_FALSE(tool("web_search").setup->needed()) << "signed in already";
}

TEST_F(WebTools, SearchFailuresTellTheUserWhatToDo) {
    tool_replies_["search"] = {{.text = "Daily search allowance used up", .is_error = true}};
    auto refused = call("web_search", {{"query", "news"}});
    ASSERT_FALSE(refused);
    EXPECT_NE(refused.error().message.find("TinyFish could not do it: Daily search allowance used up"),
              std::string::npos);

    tool_calls_.clear();
    tool_replies_["search"] = {{.status = 503}, {.text = R"({"results":[]})"}};
    auto retried = call("web_search", {{"query", "news"}});
    ASSERT_TRUE(retried) << retried.error().message;
    EXPECT_EQ(*retried, "No web results for \"news\". Try other or fewer words.");
    EXPECT_EQ(tool_calls_.size(), 2) << "an unavailable server is retried once";

    tool_calls_.clear();
    tool_replies_["search"] = {{.error = Error{ErrorCode::IO, "timed out after 15 s"}}};
    auto offline = call("web_search", {{"query", "news"}});
    ASSERT_FALSE(offline);
    EXPECT_NE(offline.error().message.find("timed out after 15 s. Tell the user; if the network or TinyFish is down"),
              std::string::npos);
    EXPECT_EQ(tool_calls_.size(), 2);

    // Without a sign-in, the tool asks to be set up before it runs.
    std::filesystem::remove(credentials_);
    build();
    const auto reason = tool("web_search").setup->needed();
    ASSERT_TRUE(reason);
    EXPECT_NE(reason->find("sign in to TinyFish"), std::string::npos);
    EXPECT_EQ(tool("web_search").setup->service, "TinyFish");
}

TEST_F(WebTools, FetchConvertsFollowsRedirectsAndPages) {
    std::string paragraphs;
    for (int index = 0; index < 400; ++index) paragraphs += "<p>Sentence number " + std::to_string(index) + ".</p>";
    replies_["https://news.example.com/old"] = {{.status = 301, .redirect = "/story"}};
    replies_["https://news.example.com/story"] = {
        {.body = "<title>Story</title><article>" + paragraphs + "<p><a href=\"/more\">More</a></p></article>"}};
    auto first = call("web_fetch", {{"url", "https://news.example.com/old"}});
    ASSERT_TRUE(first) << first.error().message;
    EXPECT_TRUE(first->starts_with("# Story\nhttps://news.example.com/story\n(characters 0-"));
    const auto hint = first->find("offset=");
    ASSERT_NE(hint, std::string::npos);
    EXPECT_LE(first->size(), 6200);
    EXPECT_NE(first->find("Sentence number 0."), std::string::npos);
    EXPECT_EQ(requests_.back().resolve, "news.example.com:443:93.184.216.34") << "connections are pinned";

    const auto offset = std::stoul(first->substr(hint + 7));
    auto second = call("web_fetch", {{"url", "https://news.example.com/old"}, {"offset", offset}});
    ASSERT_TRUE(second) << second.error().message;
    EXPECT_EQ(second->find("Sentence number 0."), std::string::npos);
    EXPECT_EQ(requests_.size(), 2) << "later parts come from the cache";

    auto past = call("web_fetch", {{"url", "https://news.example.com/old"}, {"offset", 10'000'000}});
    ASSERT_FALSE(past);
    EXPECT_NE(past.error().message.find("past the end"), std::string::npos);

    auto last = call("web_fetch", {{"url", "https://news.example.com/old"}, {"offset", 0}, {"max_chars", 16000}});
    ASSERT_TRUE(last);
    replies_["https://news.example.com/more"] = {{.body = "<p>The rest.</p>"}};
    auto more = call("web_fetch", {{"url", "/more"}});
    ASSERT_TRUE(more) << more.error().message;
    EXPECT_EQ(*more, "https://news.example.com/more\n\nThe rest.") << "a /path is relative to the page read last";
}

TEST_F(WebTools, FetchExplainsPagesItCannotRead) {
    replies_["https://news.example.com/blocked"] = {{.status = 403, .body = "denied"}};
    auto blocked = call("web_fetch", {{"url", "https://news.example.com/blocked"}});
    ASSERT_FALSE(blocked);
    EXPECT_NE(blocked.error().message.find("the site answered HTTP 403. Tell the user, or ask them to paste"),
              std::string::npos);

    replies_["https://news.example.com/gone"] = {{.status = 404, .body = "<p>Not found</p>"}};
    auto gone = call("web_fetch", {{"url", "https://news.example.com/gone"}});
    ASSERT_FALSE(gone);
    EXPECT_NE(gone.error().message.find("does not exist (HTTP 404)"), std::string::npos);

    replies_["https://news.example.com/app"] = {
        {.body = "<html><body><div id=\"root\"></div><script>" + std::string(5000, 'x') + "</script></body></html>"}};
    auto app = call("web_fetch", {{"url", "https://news.example.com/app"}});
    ASSERT_TRUE(app) << app.error().message;
    EXPECT_NE(app->find("the page builds its content with JavaScript"), std::string::npos);

    replies_["https://news.example.com/paper.pdf"] = {{.content_type = "application/pdf", .body = "%PDF-1.7"}};
    auto pdf = call("web_fetch", {{"url", "https://news.example.com/paper.pdf"}});
    ASSERT_FALSE(pdf);
    EXPECT_NE(pdf.error().message.find("not application/pdf documents"), std::string::npos);
    EXPECT_TRUE(tool_calls_.empty()) << "pages never go to TinyFish";
}

TEST_F(WebTools, FetchRefusesPrivateAddressesAndUnseenUrls) {
    auto intranet = call("web_fetch", {{"url", "http://intranet.example.com/admin"}});
    ASSERT_FALSE(intranet);
    EXPECT_NE(intranet.error().message.find("private or local address (10.0.0.7)"), std::string::npos);
    replies_["https://news.example.com/hop"] = {{.status = 302, .redirect = "http://intranet.example.com/"}};
    auto hop = call("web_fetch", {{"url", "https://news.example.com/hop"}});
    ASSERT_FALSE(hop);
    EXPECT_NE(hop.error().message.find("private or local address"), std::string::npos);
    EXPECT_EQ(requested("http://intranet"), 0);
    EXPECT_FALSE(call("web_fetch", {{"url", "file:///etc/passwd"}}));
    EXPECT_FALSE(call("web_fetch", {{"url", "https://user:pass@news.example.com/"}}));

    const auto fetch = std::ranges::find_if(tools_, [](const Tool& tool) { return tool.spec.name == "web_fetch"; });
    ASSERT_TRUE(fetch->check);
    const std::vector<Message> history{
        {.role = Role::USER, .content = "Summarize news.example.com/today please"},
        {.role = Role::TOOL, .content = "1. Story\n   https://www.other.org/a/b/\n"},
        {.role = Role::TOOL, .content = "# Docs\nhttps://docs.example.com/guide\n\nSee [setup](/guide/setup?v=2)."},
        {.role = Role::ASSISTANT, .content = "I will open https://evil.example.net/?q=secret"}};
    const auto allowed = [&](std::string url) {
        return static_cast<bool>(fetch->check({.id = "1", .name = "web_fetch", .arguments = {{"url", url}}}, history));
    };
    EXPECT_TRUE(allowed("https://news.example.com/today"));
    EXPECT_TRUE(allowed("https://other.org/a/b"));
    EXPECT_TRUE(allowed("/guide/setup?v=2")) << "a /path link from a page";
    EXPECT_TRUE(allowed("https://docs.example.com/guide/setup?v=2")) << "the page's host plus its /path link";
    EXPECT_FALSE(allowed("https://evil.example.net/guide/setup?v=2")) << "the path alone does not admit other hosts";
    EXPECT_FALSE(allowed("/guide/leak?data=1"));
    EXPECT_FALSE(allowed("https://evil.example.net/?q=secret")) << "the model's own text is not a source";
    EXPECT_FALSE(allowed("https://news.example.com/today?leak=1"));
}

} // namespace
} // namespace kidi::agent::tools
