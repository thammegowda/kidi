#include "kidi/agent/tools/web.h"

#include "kidi/agent/net/mcp.h"
#include "kidi/agent/text.h"
#include "kidi/agent/tools/arguments.h"
#include "kidi/agent/tools/html.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>

namespace kidi::agent::tools {
namespace {

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

constexpr std::size_t PAGE_CHARS = 6000; // one web_fetch page: about 1,500 tokens
constexpr std::size_t MIN_PAGE_CHARS = 1000;
constexpr std::size_t MAX_PAGE_CHARS = 16000;
constexpr std::size_t MAX_DOCUMENT_CHARS = 1024 * 1024; // converted text kept per page
constexpr std::size_t SNIPPET_CHARS = 300;
constexpr std::size_t MAX_REDIRECTS = 5;
constexpr std::size_t CACHED_PAGES = 16;
constexpr auto CACHE_LIFETIME = 15min;
constexpr auto SEARCH_TIMEOUT = 15s;
constexpr auto FETCH_TIMEOUT = 20s;
constexpr auto RETRY_DELAY = 500ms;

// Runs `attempt`, and once more after a short pause if the network failed.
template <class Attempt> auto retry_once(Attempt attempt, std::stop_token stop) -> decltype(attempt()) {
    auto result = attempt();
    if (result || result.error().code != ErrorCode::IO || stop.stop_requested()) return result;
    std::mutex mutex;
    std::unique_lock lock(mutex);
    std::condition_variable_any().wait_for(lock, stop, RETRY_DELAY, [] { return false; });
    if (stop.stop_requested()) return cancelled();
    return attempt();
}


// ---- addresses ----

auto public_ipv4(std::uint32_t address) -> bool {
    const auto first = address >> 24;
    if (first == 0 || first == 10 || first == 127 || first >= 224) return false;
    return (address & 0xFFC00000U) != 0x64400000U     // 100.64.0.0/10, carrier-grade NAT
           && (address & 0xFFFF0000U) != 0xA9FE0000U  // 169.254.0.0/16, link-local
           && (address & 0xFFF00000U) != 0xAC100000U  // 172.16.0.0/12
           && (address & 0xFFFFFF00U) != 0xC0000000U  // 192.0.0.0/24
           && (address & 0xFFFF0000U) != 0xC0A80000U  // 192.168.0.0/16
           && (address & 0xFFFE0000U) != 0xC6120000U; // 198.18.0.0/15, benchmarking
}

struct UrlParts {
    std::string scheme, host, port;
};

// An absolute http(s) URL without credentials; the host is lowercase and without IPv6 brackets.
auto parse_url(std::string_view url) -> std::optional<UrlParts> {
    const auto scheme_end = url.find("://");
    if (scheme_end == std::string_view::npos) return std::nullopt;
    UrlParts parts{lower(url.substr(0, scheme_end)), {}, {}};
    if (parts.scheme != "http" && parts.scheme != "https") return std::nullopt;
    const auto authority_start = scheme_end + 3;
    auto authority = url.substr(authority_start, url.find_first_of("/?#", authority_start) - authority_start);
    if (authority.empty() || authority.find('@') != std::string_view::npos) return std::nullopt;
    std::string_view port;
    if (authority.front() == '[') {
        const auto close = authority.find(']');
        if (close == std::string_view::npos) return std::nullopt;
        parts.host = lower(authority.substr(1, close - 1));
        if (close + 1 < authority.size()) {
            if (authority[close + 1] != ':') return std::nullopt;
            port = authority.substr(close + 2);
        }
    } else {
        const auto colon = authority.rfind(':');
        parts.host = lower(authority.substr(0, colon));
        if (colon != std::string_view::npos) port = authority.substr(colon + 1);
    }
    if (parts.host.empty() || parts.host.find_first_of(" \t\r\n\\") != std::string::npos) return std::nullopt;
    if (!port.empty() && (port.size() > 5 || port.find_first_not_of("0123456789") != std::string_view::npos))
        return std::nullopt;
    parts.port = port.empty() ? (parts.scheme == "https" ? "443" : "80") : std::string(port);
    return parts;
}

auto resolve_host(const std::string& host) -> Result<std::vector<std::string>> {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* list = nullptr;
    if (const int status = ::getaddrinfo(host.c_str(), nullptr, &hints, &list); status != 0)
        return failure(ErrorCode::IO, "could not resolve " + host + " (" + ::gai_strerror(status) + ")");
    std::vector<std::string> addresses;
    for (auto* entry = list; entry; entry = entry->ai_next) {
        std::array<char, INET6_ADDRSTRLEN> text{};
        const void* address =
            entry->ai_family == AF_INET
                ? static_cast<const void*>(&reinterpret_cast<const sockaddr_in*>(entry->ai_addr)->sin_addr)
                : static_cast<const void*>(&reinterpret_cast<const sockaddr_in6*>(entry->ai_addr)->sin6_addr);
        if ((entry->ai_family == AF_INET || entry->ai_family == AF_INET6) &&
            ::inet_ntop(entry->ai_family, address, text.data(), text.size()) &&
            std::ranges::find(addresses, text.data()) == addresses.end())
            addresses.emplace_back(text.data());
    }
    ::freeaddrinfo(list);
    if (addresses.empty()) return failure(ErrorCode::IO, "could not resolve " + host + " (no addresses)");
    return addresses;
}

// The part of a URL that identifies it in conversation text: no scheme, "www.", fragment or trailing slash.
auto url_key(std::string_view url) -> std::string {
    auto key = lower(trim(url));
    if (const auto hash = key.find('#'); hash != std::string::npos) key.resize(hash);
    if (const auto scheme = key.find("://"); scheme != std::string::npos) key.erase(0, scheme + 3);
    if (key.starts_with("www.")) key.erase(0, 4);
    while (!key.empty() && key.back() == '/') key.pop_back();
    return key;
}

struct FetchedPage {
    std::string title;
    std::string url; // after redirects
    std::string text;
    std::string note; // a caveat shown with every page, e.g. that the page may need JavaScript
};

auto charset_of(std::string_view content_type, std::string_view body) -> std::string {
    const auto find = [](std::string_view text) -> std::string {
        const auto lowered = lower(text);
        const auto at = lowered.find("charset=");
        if (at == std::string::npos) return {};
        auto value = std::string_view(lowered).substr(at + 8);
        if (!value.empty() && (value.front() == '"' || value.front() == '\'')) value.remove_prefix(1);
        const auto end = value.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-_");
        return std::string(value.substr(0, end));
    };
    if (auto charset = find(content_type); !charset.empty()) return charset;
    return find(body.substr(0, 4096));
}

auto looks_like_html(std::string_view body) -> bool {
    const auto start = lower(trim(body.substr(0, 512)));
    return start.starts_with("<!doctype html") || start.starts_with("<html") ||
           start.find("<head") != std::string::npos;
}

// A page's text as compact Markdown (HTML) or as it is (other text); other documents are refused.
auto convert(const std::string& url, const net::HttpResponse& response) -> Result<FetchedPage> {
    const auto media_type = std::string_view(response.content_type).substr(0, response.content_type.find(';'));
    const auto type = lower(trim(media_type));
    const auto charset = charset_of(response.content_type, response.body);
    FetchedPage page{.url = url};
    if (type == "text/html" || type == "application/xhtml+xml" ||
        ((type.empty() || type == "text/plain") && looks_like_html(response.body))) {
        auto html = html_to_markdown(to_utf8(response.body, charset), url);
        page.title = std::move(html.title);
        page.text = std::move(html.markdown);
        // A large document that converts to almost nothing is built by script.
        if (page.text.size() < 200 && response.body.size() > 2000)
            page.note = "Some content may be missing: the page builds its content with JavaScript.";
    } else if (type.starts_with("text/") || type == "application/json" || type.ends_with("+json") ||
               type == "application/xml" || type.ends_with("+xml") || type == "application/javascript") {
        page.text = to_utf8(response.body, charset);
    } else {
        return failure(ErrorCode::UNSUPPORTED, "web_fetch reads web pages and text, not " +
                                                   (type.empty() ? std::string("unknown") : type) +
                                                   " documents. Tell the user, or ask them to paste the text.");
    }
    if (page.text.size() > MAX_DOCUMENT_CHARS) {
        page.text.resize(utf8_prefix(page.text, MAX_DOCUMENT_CHARS).size());
        page.note = "The document was cut to its first 1 MiB of text.";
    }
    return page;
}

// A page of at most `max_chars` from `offset`, cut at a paragraph or line break when one is near.
auto render(const FetchedPage& page, std::size_t offset, std::size_t max_chars) -> Result<std::string> {
    const auto& text = page.text;
    if (offset > 0 && offset >= text.size())
        return failure(ErrorCode::INVALID_ARGUMENT,
                       "offset is past the end of the page (" + std::to_string(text.size()) + " characters)");
    while (offset < text.size() && (static_cast<unsigned char>(text[offset]) & 0xC0) == 0x80) ++offset;
    auto end = text.size();
    if (text.size() - offset > max_chars) {
        const auto window = utf8_prefix(std::string_view(text).substr(offset), max_chars);
        auto cut = window.rfind("\n\n");
        if (cut == std::string_view::npos || cut < max_chars / 2) cut = window.rfind('\n');
        if (cut == std::string_view::npos || cut < max_chars / 2) cut = window.rfind(' ');
        if (cut == std::string_view::npos || cut < max_chars / 2) cut = window.size();
        end = offset + cut;
    }
    std::string out;
    if (!page.title.empty()) out += "# " + one_line(page.title, 200) + "\n";
    out += page.url + "\n";
    if (offset > 0 || end < text.size()) {
        out += "(characters " + std::to_string(offset) + "-" + std::to_string(end) + " of " +
               std::to_string(text.size());
        if (end < text.size()) out += "; for more, call web_fetch with offset=" + std::to_string(end);
        out += ")\n";
    }
    if (!page.note.empty()) out += page.note + "\n";
    out += "\n";
    out += trim(std::string_view(text).substr(offset, end - offset));
    if (trim(text).empty()) out += "(The page has no readable text.)";
    return out;
}

// web_search's provider: an MCP server and its search tool, from config {server: <mcp_servers name>[, tool: NAME]}.
struct SearchProvider {
    std::shared_ptr<net::McpServer> server;
    std::string tool;
};

auto search_provider(const Json& config, const net::McpServers& servers) -> Result<SearchProvider> {
    const std::string usage = "expected config: {server: <mcp_servers name>, tool: search}, with tool optional";
    if (!config.contains("server")) return failure(ErrorCode::INVALID_ARGUMENT, usage);
    for (const auto& [key, value] : config.items())
        if ((key != "server" && key != "tool") || !value.is_string() || value.get<std::string>().empty())
            return failure(ErrorCode::INVALID_ARGUMENT, usage);
    const auto name = config["server"].get<std::string>();
    const auto server = servers.find(name);
    if (server == servers.end())
        return failure(ErrorCode::INVALID_ARGUMENT, "unknown MCP server " + name + "; define it under mcp_servers");
    return SearchProvider{server->second, config.value("tool", std::string("search"))};
}

auto search(const SearchProvider& provider, const Json& args, std::stop_token stop) -> Result<std::string> {
    auto query = string_argument(args, "query");
    if (!query) return std::unexpected(query.error());
    const auto text = std::string(trim(*query));
    if (text.empty() || text.size() > 400)
        return failure(ErrorCode::INVALID_ARGUMENT, "query must be 1 to 400 characters of search words");
    auto count = size_argument(args, "count", 5);
    if (!count) return std::unexpected(count.error());
    const auto limit = std::clamp<std::size_t>(*count, 1, 10);
    const auto& service = provider.server->service();
    auto reply = retry_once(
        [&] { return provider.server->call_tool(provider.tool, {{"query", text}}, stop, SEARCH_TIMEOUT); }, stop);
    if (!reply && stop.stop_requested()) return cancelled();
    if (!reply)
        return failure(reply.error().code, "web_search failed: " + reply.error().message +
                                               ". Tell the user; if the network or " + service +
                                               " is down, they can retry later.");
    if (reply->is_error)
        return failure(ErrorCode::RUNTIME, "web_search failed: " + service + " could not do it: " +
                                               one_line(reply->text, 300) + ". Tell the user.");
    const auto json = reply->structured.is_null() ? Json::parse(reply->text, nullptr, false) : reply->structured;
    const auto results = json.is_array() ? json : json.is_object() ? json.value("results", Json()) : Json();
    if (!results.is_array()) {
        // An unfamiliar layout is still useful to the model, within bounds.
        const auto text_reply = std::string(utf8_prefix(trim(reply->text), 3000));
        if (text_reply.empty())
            return failure(ErrorCode::IO, "web_search failed: " + service + " returned nothing. Tell the user.");
        return "Web results for \"" + text + "\":\n" + text_reply;
    }
    std::string out;
    std::size_t shown = 0;
    for (const auto& result : results) {
        if (shown == limit) break;
        if (!result.is_object() || !result.contains("url") || !result["url"].is_string()) continue;
        const auto url = result["url"].get<std::string>();
        const auto title = result.contains("title") && result["title"].is_string()
                               ? one_line(result["title"].get<std::string>(), 150)
                               : url;
        out += std::to_string(++shown) + ". " + title + "\n   " + url + "\n";
        if (result.contains("snippet") && result["snippet"].is_string())
            if (auto snippet = one_line(result["snippet"].get<std::string>(), SNIPPET_CHARS); !snippet.empty())
                out += "   " + snippet + "\n";
    }
    if (!shown) return "No web results for \"" + text + "\". Try other or fewer words.";
    return "Web results for \"" + text + "\" (read one with web_fetch):\n" + out;
}

// web_fetch: reads public pages itself, remembering recent ones so later parts need no new request.
class WebFetch {
public:
    explicit WebFetch(WebToolOptions options) : options_(std::move(options)) {}

    auto fetch(const Json& args, std::stop_token stop) -> Result<std::string> {
        auto url_argument = string_argument(args, "url");
        if (!url_argument) return std::unexpected(url_argument.error());
        auto url = std::string(trim(*url_argument));
        // A /path link from a page is relative to the page read last.
        if (url.starts_with('/') && !url.starts_with("//")) {
            const auto base = last_url();
            if (base.empty())
                return failure(ErrorCode::INVALID_ARGUMENT,
                               "a /path needs a page to be relative to; pass the absolute URL instead");
            url = resolve_url(base, url);
        }
        if (url.size() > 2048 || !parse_url(url))
            return failure(ErrorCode::INVALID_ARGUMENT,
                           "url must be an absolute http:// or https:// address without a user name or password, or "
                           "a /path link from the page read last");
        auto offset = size_argument(args, "offset", 0);
        auto max_chars = size_argument(args, "max_chars", PAGE_CHARS);
        if (!offset) return std::unexpected(offset.error());
        if (!max_chars) return std::unexpected(max_chars.error());
        auto page = cached(url);
        if (!page) {
            auto loaded = load(url, stop);
            if (!loaded) return std::unexpected(loaded.error());
            page = std::move(*loaded);
            remember(url, *page);
        }
        {
            const std::lock_guard lock(mutex_);
            last_url_ = page->url;
        }
        return render(*page, *offset, std::clamp(*max_chars, MIN_PAGE_CHARS, MAX_PAGE_CHARS));
    }

    // web_fetch opens only addresses that a user message or a tool result contains, so text injected into a page
    // cannot make the model send data to an address it composed. Pages list same-site links as /paths, so an
    // address also counts when a tool result shows both its host and its path.
    auto check(const ToolCall& call, std::span<const Message> history) const -> Result<void> {
        const auto url = call.arguments.find("url");
        if (url == call.arguments.end() || !url->is_string()) return {};
        const auto text = std::string(trim(url->get<std::string>()));
        const auto key = url_key(text);
        if (key.empty()) return {};
        const auto path_start = text.starts_with('/') ? 0 : key.find('/');
        const auto path = text.starts_with('/') ? key : path_start == std::string::npos ? "" : key.substr(path_start);
        const auto host = text.starts_with('/') ? std::string{} : key.substr(0, path_start);
        for (const auto& message : history) {
            if (message.role != Role::USER && message.role != Role::TOOL) continue;
            const auto content = lower(message.content);
            if (!text.starts_with('/') && content.find(key) != std::string::npos) return {};
            if (message.role == Role::TOOL && path.size() > 1 && content.find("(" + path) != std::string::npos &&
                (host.empty() || content.find(host) != std::string::npos))
                return {};
        }
        return failure(ErrorCode::INVALID_ARGUMENT,
                       "web_fetch opens only addresses that the user gave or that earlier web_search or web_fetch "
                       "results listed, and " + text +
                           " did not appear. Find it with web_search, or ask the user for the address.");
    }

private:
    // Follows redirects hop by hop, connecting only to public addresses, pinned to the address that was checked.
    auto load(const std::string& requested, std::stop_token stop) const -> Result<FetchedPage> {
        const auto fail = [&](ErrorCode code, const std::string& reason, std::string_view advice = {}) {
            return failure(code, "web_fetch could not read " + requested + ": " + reason + "." +
                                     (advice.empty() ? "" : " ") + std::string(advice));
        };
        auto url = requested;
        for (std::size_t redirects = 0;; ++redirects) {
            const auto parts = parse_url(url);
            if (!parts) return fail(ErrorCode::INVALID_ARGUMENT, "it redirects to an unsupported address");
            auto addresses = options_.resolve(parts->host);
            if (!addresses)
                return fail(addresses.error().code, addresses.error().message,
                            "Check the address; if it is right, the network may be down, so tell the user.");
            for (const auto& address : *addresses)
                if (!public_address(address))
                    return fail(ErrorCode::INVALID_ARGUMENT,
                                parts->host + " is a private or local address (" + address +
                                    "); web_fetch reads only the public internet");
            const auto& pinned = addresses->front();
            const auto ip = pinned.find(':') == std::string::npos ? pinned : "[" + pinned + "]";
            const net::HttpRequest request{.url = url,
                                      .headers = {"Accept: text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
                                                  "Accept-Language: en-US,en;q=0.8"},
                                      .timeout = FETCH_TIMEOUT,
                                      .resolve = parts->host + ":" + parts->port + ":" + ip};
            auto response = retry_once([&] { return options_.transport(request, stop); }, stop);
            if (!response && stop.stop_requested()) return cancelled();
            if (!response)
                return fail(response.error().code, response.error().message,
                            "The site or the network may be down; tell the user to retry later.");
            const auto status = response->status;
            if (status >= 300 && status < 400 && !response->redirect_url.empty()) {
                if (redirects == MAX_REDIRECTS)
                    return fail(ErrorCode::IO, "it redirects more than " + std::to_string(MAX_REDIRECTS) + " times");
                url = resolve_url(url, response->redirect_url);
                if (url.empty()) return fail(ErrorCode::INVALID_ARGUMENT, "it redirects to an unsupported address");
                continue;
            }
            if (status == 404 || status == 410)
                return fail(ErrorCode::INVALID_ARGUMENT,
                            "the page does not exist (HTTP " + std::to_string(status) + ")",
                            "Search for its current address instead.");
            if (status < 200 || status >= 300)
                return fail(ErrorCode::IO, "the site answered HTTP " + std::to_string(status),
                            "Tell the user, or ask them to paste the page text.");
            return convert(url, *response);
        }
    }

    auto last_url() -> std::string {
        const std::lock_guard lock(mutex_);
        return last_url_;
    }

    auto cached(const std::string& url) -> std::optional<FetchedPage> {
        const std::lock_guard lock(mutex_);
        std::erase_if(cache_, [](const auto& entry) { return Clock::now() - entry.loaded > CACHE_LIFETIME; });
        const auto entry = std::ranges::find(cache_, url, &CacheEntry::url);
        if (entry == cache_.end()) return std::nullopt;
        cache_.splice(cache_.begin(), cache_, entry);
        return cache_.front().page;
    }

    auto remember(const std::string& url, const FetchedPage& page) -> void {
        const std::lock_guard lock(mutex_);
        cache_.push_front({url, page, Clock::now()});
        if (cache_.size() > CACHED_PAGES) cache_.pop_back();
    }

    struct CacheEntry {
        std::string url;
        FetchedPage page;
        Clock::time_point loaded;
    };

    WebToolOptions options_;
    std::mutex mutex_;
    std::list<CacheEntry> cache_;
    std::string last_url_; // the page read last, which /path links are relative to
};

} // namespace

auto public_address(std::string_view address) -> bool {
    auto text = std::string(address);
    if (text.size() > 2 && text.front() == '[' && text.back() == ']') text = text.substr(1, text.size() - 2);
    in_addr ipv4{};
    if (::inet_pton(AF_INET, text.c_str(), &ipv4) == 1) return public_ipv4(ntohl(ipv4.s_addr));
    in6_addr ipv6{};
    if (::inet_pton(AF_INET6, text.c_str(), &ipv6) != 1) return false;
    const auto* bytes = ipv6.s6_addr;
    const auto embedded_ipv4 = [&] {
        return (std::uint32_t{bytes[12]} << 24) | (std::uint32_t{bytes[13]} << 16) | (std::uint32_t{bytes[14]} << 8) |
               std::uint32_t{bytes[15]};
    };
    const auto zero_until = [&](int end) { return std::all_of(bytes, bytes + end, [](auto byte) { return !byte; }); };
    if (zero_until(15) && (bytes[15] == 0 || bytes[15] == 1)) return false; // :: and ::1
    if ((bytes[0] & 0xFE) == 0xFC) return false;                            // fc00::/7, unique local
    if (bytes[0] == 0xFE && (bytes[1] & 0xC0) == 0x80) return false;        // fe80::/10, link-local
    if (bytes[0] == 0xFF) return false;                                     // multicast
    if (zero_until(10) && bytes[10] == 0xFF && bytes[11] == 0xFF) return public_ipv4(embedded_ipv4()); // mapped
    if (bytes[0] == 0 && bytes[1] == 0x64 && bytes[2] == 0xFF && bytes[3] == 0x9B) return public_ipv4(embedded_ipv4());
    return true;
}

auto web_tool_handlers(WebToolOptions options) -> std::vector<ToolHandler> {
    if (!options.transport) options.transport = net::curl_transport();
    if (!options.resolve) options.resolve = resolve_host;
    return {
        {"web_search",
         [options](const Json& config) -> Result<Tool> {
             auto provider = search_provider(config, options.servers);
             if (!provider) return std::unexpected(provider.error());
             return Tool{.spec = {.name = "web_search"},
                         .capability = Capability::NETWORK,
                         .invoke = [provider = *provider](const Json& args, std::stop_token stop) {
                             return search(provider, args, stop);
                         },
                         .setup = provider->server->setup()};
         }},
        {"web_fetch",
         [options](const Json&) -> Result<Tool> {
             auto fetch = std::make_shared<WebFetch>(options);
             return Tool{.spec = {.name = "web_fetch"},
                         .capability = Capability::NETWORK,
                         .invoke = [fetch](const Json& args, std::stop_token stop) { return fetch->fetch(args, stop); },
                         .check = [fetch](const ToolCall& call, std::span<const Message> history) {
                             return fetch->check(call, history);
                         }};
         }},
    };
}

} // namespace kidi::agent::tools
