#include "kidi/agent/net/oauth.h"

#include <liteapp/core/base64.h>
#include <liteapp/core/random.h>
#include <liteapp/core/sha256.h>

#include <algorithm>
#include <chrono>

namespace kidi::agent::net {
namespace {

using namespace std::chrono_literals;

auto get_json(const HttpTransport& transport, const std::string& url, std::stop_token stop) -> Result<Json> {
    auto response = transport({.url = url, .headers = {"Accept: application/json"}, .timeout = 15s,
                               .max_bytes = 1024 * 1024},
                              stop);
    if (!response) return std::unexpected(response.error());
    if (response->status != 200) {
        return failure(ErrorCode::IO, url + " answered HTTP " + std::to_string(response->status));
    }
    auto json = Json::parse(response->body, nullptr, false);
    if (!json.is_object()) return failure(ErrorCode::IO, url + " returned no JSON object");
    return json;
}

auto string_field(const Json& json, const char* key) -> std::string {
    const auto value = json.find(key);
    return value != json.end() && value->is_string() ? value->get<std::string>() : std::string{};
}

// scheme://host[:port] and the path of a URL.
auto split_url(const std::string& url) -> std::pair<std::string, std::string> {
    const auto scheme = url.find("://");
    const auto path = scheme == std::string::npos ? std::string::npos : url.find('/', scheme + 3);
    if (path == std::string::npos) return {url, ""};
    auto rest = url.substr(path);
    rest = rest.substr(0, rest.find_first_of("?#"));
    while (!rest.empty() && rest.back() == '/') rest.pop_back();
    return {url.substr(0, path), rest};
}

// A token endpoint's answer, success or a described failure.
auto token_response(const Result<HttpResponse>& response, std::string_view action) -> Result<OAuthToken> {
    if (!response) return std::unexpected(response.error());
    const auto json = Json::parse(response->body, nullptr, false);
    if (response->status != 200 || !json.is_object() || string_field(json, "access_token").empty()) {
        auto reason = json.is_object() ? string_field(json, "error_description") : std::string{};
        if (reason.empty() && json.is_object()) reason = string_field(json, "error");
        if (reason.empty()) reason = "HTTP " + std::to_string(response->status);
        return failure(response->status >= 500 ? ErrorCode::IO : ErrorCode::INVALID_ARGUMENT,
                       std::string(action) + " failed: " + reason);
    }
    OAuthToken token{string_field(json, "access_token"), string_field(json, "refresh_token")};
    if (const auto lifetime = json.find("expires_in"); lifetime != json.end() && lifetime->is_number()) {
        const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
        token.expires_at = now + lifetime->get<std::int64_t>();
    }
    return token;
}

auto form(std::initializer_list<std::pair<std::string_view, std::string_view>> fields) -> std::string {
    std::string body;
    for (const auto& [name, value] : fields) {
        if (value.empty()) continue;
        if (!body.empty()) body += '&';
        body += url_encode(name) + "=" + url_encode(value);
    }
    return body;
}

auto post_form(const HttpTransport& transport, const std::string& url, std::string body, std::stop_token stop)
    -> Result<HttpResponse> {
    return transport({.url = url,
                      .headers = {"Content-Type: application/x-www-form-urlencoded", "Accept: application/json"},
                      .body = std::move(body),
                      .timeout = 20s,
                      .max_bytes = 1024 * 1024},
                     stop);
}

auto query_value(std::string_view query, std::string_view name) -> std::string {
    for (std::size_t begin = 0; begin <= query.size();) {
        auto end = query.find('&', begin);
        if (end == std::string_view::npos) end = query.size();
        const auto pair = query.substr(begin, end - begin);
        const auto equals = pair.find('=');
        if (pair.substr(0, equals) == name && equals != std::string_view::npos) {
            std::string value;
            const auto raw = pair.substr(equals + 1);
            for (std::size_t index = 0; index < raw.size(); ++index) {
                if (raw[index] == '+') {
                    value += ' ';
                } else if (raw[index] == '%' && index + 2 < raw.size()) {
                    value += static_cast<char>(std::stoi(std::string(raw.substr(index + 1, 2)), nullptr, 16));
                    index += 2;
                } else {
                    value += raw[index];
                }
            }
            return value;
        }
        begin = end + 1;
    }
    return {};
}

} // namespace

auto discover_oauth(const HttpTransport& transport, const std::string& resource, const std::string& challenge,
                    std::stop_token stop) -> Result<OAuthServer> {
    const auto [origin, path] = split_url(resource);
    // The resource's metadata: named by its 401 challenge, else at the well-known location for its path.
    std::string metadata_url;
    if (const auto at = challenge.find("resource_metadata=\""); at != std::string::npos) {
        const auto start = at + 19;
        metadata_url = challenge.substr(start, challenge.find('"', start) - start);
    }
    Result<Json> metadata = failure(ErrorCode::IO, "no metadata");
    for (const auto& url : {metadata_url, origin + "/.well-known/oauth-protected-resource" + path,
                            origin + "/.well-known/oauth-protected-resource"}) {
        if (url.empty()) continue;
        metadata = get_json(transport, url, stop);
        if (metadata) break;
    }
    if (!metadata) return failure(ErrorCode::UNSUPPORTED, resource + " publishes no OAuth sign-in information");
    const auto servers = metadata->find("authorization_servers");
    if (servers == metadata->end() || !servers->is_array() || servers->empty() || !(*servers)[0].is_string())
        return failure(ErrorCode::UNSUPPORTED, resource + " names no sign-in server");
    const auto issuer = (*servers)[0].get<std::string>();
    const auto [issuer_origin, issuer_path] = split_url(issuer);
    Result<Json> server = failure(ErrorCode::IO, "no metadata");
    for (const auto& url : {issuer_origin + "/.well-known/oauth-authorization-server" + issuer_path,
                            issuer_origin + "/.well-known/openid-configuration" + issuer_path,
                            issuer + "/.well-known/openid-configuration"}) {
        server = get_json(transport, url, stop);
        if (server) break;
    }
    if (!server) return failure(ErrorCode::UNSUPPORTED, "cannot read the sign-in server's configuration at " + issuer);
    OAuthServer result{string_field(*metadata, "resource").empty() ? resource : string_field(*metadata, "resource"),
                       string_field(*server, "authorization_endpoint"), string_field(*server, "token_endpoint"),
                       string_field(*server, "registration_endpoint"), {}};
    if (result.authorization_endpoint.empty() || result.token_endpoint.empty())
        return failure(ErrorCode::UNSUPPORTED, "the sign-in server at " + issuer + " lacks required endpoints");
    if (const auto methods = server->find("code_challenge_methods_supported");
        methods != server->end() && methods->is_array() && std::ranges::find(*methods, "S256") == methods->end())
        return failure(ErrorCode::UNSUPPORTED, "the sign-in server at " + issuer + " does not support PKCE (S256)");
    // The resource's scopes when it lists them; else the identity scopes, with offline_access for refresh tokens.
    std::vector<std::string> scopes;
    if (const auto listed = metadata->find("scopes_supported"); listed != metadata->end() && listed->is_array()) {
        for (const auto& scope : *listed)
            if (scope.is_string()) scopes.push_back(scope.get<std::string>());
    } else if (const auto offered = server->find("scopes_supported"); offered != server->end() && offered->is_array()) {
        for (const auto* wanted : {"openid", "profile", "email", "offline_access"})
            if (std::ranges::find(*offered, wanted) != offered->end()) scopes.emplace_back(wanted);
    }
    for (const auto& scope : scopes) result.scope += (result.scope.empty() ? "" : " ") + scope;
    return result;
}

auto register_oauth_client(const HttpTransport& transport, const OAuthServer& server, const std::string& client_name,
                           const std::string& redirect_uri, std::stop_token stop) -> Result<std::string> {
    if (server.registration_endpoint.empty())
        return failure(ErrorCode::UNSUPPORTED, "the sign-in server does not let applications register themselves");
    Json request{{"client_name", client_name},
                 {"redirect_uris", Json::array({redirect_uri})},
                 {"grant_types", Json::array({"authorization_code", "refresh_token"})},
                 {"response_types", Json::array({"code"})},
                 {"token_endpoint_auth_method", "none"}};
    if (!server.scope.empty()) request["scope"] = server.scope;
    auto response = transport({.url = server.registration_endpoint,
                               .headers = {"Content-Type: application/json", "Accept: application/json"},
                               .body = request.dump(),
                               .timeout = 20s,
                               .max_bytes = 1024 * 1024},
                              stop);
    if (!response) return std::unexpected(response.error());
    const auto json = Json::parse(response->body, nullptr, false);
    if ((response->status != 200 && response->status != 201) || !json.is_object() ||
        string_field(json, "client_id").empty())
        return failure(ErrorCode::IO, "registering with the sign-in server failed (HTTP " +
                                          std::to_string(response->status) + ")");
    return string_field(json, "client_id");
}

auto begin_sign_in(const OAuthServer& server, std::string client_id, std::string redirect_uri)
    -> Result<PendingSignIn> {
    auto verifier = liteapp::secure_random_token(32);
    auto state = liteapp::secure_random_token(16);
    if (!verifier || !state) return failure(ErrorCode::IO, "cannot create sign-in secrets");
    const auto digest = liteapp::sha256(*verifier);
    const auto challenge = liteapp::base64_encode(std::span<const std::byte>(digest), liteapp::Base64::url);
    PendingSignIn pending{server, std::move(client_id), std::move(redirect_uri), *verifier, *state, {}};
    pending.authorize_url = server.authorization_endpoint +
                            (server.authorization_endpoint.find('?') == std::string::npos ? "?" : "&") +
                            form({{"response_type", "code"},
                                  {"client_id", pending.client_id},
                                  {"redirect_uri", pending.redirect_uri},
                                  {"code_challenge", challenge},
                                  {"code_challenge_method", "S256"},
                                  {"state", pending.state},
                                  {"scope", server.scope},
                                  {"resource", server.resource}});
    return pending;
}

auto finish_sign_in(const HttpTransport& transport, const PendingSignIn& pending, std::string_view callback,
                    std::stop_token stop) -> Result<OAuthToken> {
    const auto query = callback.substr(callback.find('?') == std::string_view::npos ? 0 : callback.find('?') + 1);
    if (const auto error = query_value(query, "error"); !error.empty()) {
        const auto description = query_value(query, "error_description");
        return failure(ErrorCode::INVALID_ARGUMENT, "sign-in was not completed: " +
                                                        (description.empty() ? error : description));
    }
    if (query_value(query, "state") != pending.state)
        return failure(ErrorCode::INVALID_ARGUMENT, "the sign-in reply does not belong to this sign-in");
    const auto code = query_value(query, "code");
    if (code.empty()) return failure(ErrorCode::INVALID_ARGUMENT, "the sign-in reply carries no authorization code");
    return token_response(post_form(transport, pending.server.token_endpoint,
                                    form({{"grant_type", "authorization_code"},
                                          {"code", code},
                                          {"redirect_uri", pending.redirect_uri},
                                          {"client_id", pending.client_id},
                                          {"code_verifier", pending.verifier},
                                          {"resource", pending.server.resource}}),
                                    stop),
                          "signing in");
}

auto refresh_oauth_token(const HttpTransport& transport, const std::string& token_endpoint,
                         const std::string& client_id, const std::string& resource, const std::string& refresh_token,
                         std::stop_token stop) -> Result<OAuthToken> {
    auto token = token_response(post_form(transport, token_endpoint,
                                          form({{"grant_type", "refresh_token"},
                                                {"refresh_token", refresh_token},
                                                {"client_id", client_id},
                                                {"resource", resource}}),
                                          stop),
                                "renewing the sign-in");
    // Servers that do not rotate refresh tokens keep the old one valid.
    if (token && token->refresh_token.empty()) token->refresh_token = refresh_token;
    return token;
}

} // namespace kidi::agent::net
