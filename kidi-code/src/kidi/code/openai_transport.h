#pragma once

#include "kidi/code/openai.h"

namespace kidi::code {

using CompletionTransport =
    std::function<Result<std::string>(const OpenAIOptions&, std::string, std::stop_token, std::string_view)>;
auto completion_provider(OpenAIOptions options, CompletionTransport transport) -> Result<Provider>;
auto http_completion(const OpenAIOptions& options, std::string payload, std::stop_token stop, int expected_pid = 0,
                     std::string_view route = "/chat/completions") -> Result<std::string>;

} // namespace kidi::code
