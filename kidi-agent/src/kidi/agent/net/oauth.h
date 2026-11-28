#pragma once

#include "kidi/agent/agent.h"
#include "kidi/agent/net/http.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kidi::agent::net {

// OAuth 2.1 for a public client signing in to a protected resource (an MCP server), as MCP specifies: discovery
// from the resource's metadata, dynamic client registration, the authorization code flow with PKCE (S256) and
// the RFC 8707 resource parameter, and refresh tokens.

struct OAuthServer {
    std::string resource; // the protected resource the tokens are for
    std::string authorization_endpoint;
    std::string token_endpoint;
    std::string registration_endpoint;
    std::string scope; // space-separated scopes to request
};

struct OAuthToken {
    std::string access_token;
    std::string refresh_token;
    std::int64_t expires_at = 0; // seconds since the epoch; 0 when the server gave no lifetime
};

struct PendingSignIn {
    OAuthServer server;
    std::string client_id;
    std::string redirect_uri;
    std::string verifier;
    std::string state;
    std::string authorize_url; // where the user signs in
};

// `challenge` is the resource's WWW-Authenticate header from a 401 response, which may name its metadata.
auto discover_oauth(const HttpTransport& transport, const std::string& resource, const std::string& challenge,
                    std::stop_token stop) -> Result<OAuthServer>;
auto register_oauth_client(const HttpTransport& transport, const OAuthServer& server, const std::string& client_name,
                           const std::string& redirect_uri, std::stop_token stop) -> Result<std::string>;
auto begin_sign_in(const OAuthServer& server, std::string client_id, std::string redirect_uri) -> Result<PendingSignIn>;
// `callback` is the URL (or just its query) that the browser was sent back to.
auto finish_sign_in(const HttpTransport& transport, const PendingSignIn& pending, std::string_view callback,
                    std::stop_token stop) -> Result<OAuthToken>;
auto refresh_oauth_token(const HttpTransport& transport, const std::string& token_endpoint,
                         const std::string& client_id, const std::string& resource, const std::string& refresh_token,
                         std::stop_token stop) -> Result<OAuthToken>;

} // namespace kidi::agent::net
