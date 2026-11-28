#pragma once

#include "kidi/core/error.h"

#include <chrono>
#include <functional>
#include <map>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

namespace kidi::agent::net {

struct HttpRequest {
    std::string url;
    std::vector<std::string> headers; // "Name: value"
    std::optional<std::string> body;  // sent as POST when present
    std::chrono::seconds timeout{20};
    std::size_t max_bytes = 5 * 1024 * 1024;
    std::string resolve; // optional "host:port:address" pin for the connection
};

struct HttpResponse {
    int status = 0;
    std::string content_type;
    std::string redirect_url; // Location of a 3xx response, resolved; redirects are not followed
    std::string body;
    std::map<std::string, std::string> headers; // lowercase names; the first value of repeated headers
};

// Transport failures use ErrorCode::IO when a retry may help (timeouts, resets, TLS and DNS failures),
// UNSUPPORTED when curl is missing, and RUNTIME "cancelled" when `stop` fired.
using HttpTransport = std::function<Result<HttpResponse>(const HttpRequest&, std::stop_token)>;

// HTTP(S) through the curl program. Requests (headers with credentials included) reach curl on its standard input,
// never on its command line; the process is killed on cancellation, timeout or an oversized response.
auto curl_transport() -> HttpTransport;

// RFC 3986 percent-encoding of everything but unreserved characters, for query strings and form bodies.
auto url_encode(std::string_view text) -> std::string;

} // namespace kidi::agent::net
