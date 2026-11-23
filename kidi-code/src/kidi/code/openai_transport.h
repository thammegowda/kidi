#pragma once

#include "kidi/code/openai.h"

namespace kidi::code {

using CompletionTransport = std::function<Result<std::string>(const OpenAIOptions&, std::string, std::stop_token)>;
auto completion_provider(OpenAIOptions options, CompletionTransport transport) -> Result<Provider>;
auto http_completion(const OpenAIOptions& options, std::string payload, std::stop_token stop, int expected_pid = 0)
    -> Result<std::string>;

} // namespace kidi::code
