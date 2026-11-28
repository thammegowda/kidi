#pragma once

#include "kidi/agent/chat/client.h"

namespace kidi::agent::chat {

using CompletionTransport =
    std::function<Result<std::string>(const ChatClientOptions&, std::string, std::stop_token, std::string_view)>;
auto completion_provider(ChatClientOptions options, CompletionTransport transport) -> Result<Provider>;
auto http_completion(const ChatClientOptions& options, std::string payload, std::stop_token stop, int expected_pid = 0,
                     std::string_view route = "/chat/completions") -> Result<std::string>;

} // namespace kidi::agent::chat
