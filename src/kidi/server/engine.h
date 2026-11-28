#pragma once

/// Serving engine shared by every kidi server front end (`kidi serve` and `kidi agent`'s background server).
/// One worker thread batches all queued generations; token inspection is serialized with that worker.
/// Callers block in `run` on their own thread and receive streamed text there.

#include "kidi/inference/generator.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>

namespace kidi::server {

struct ChatContext {
    std::size_t prompt_tokens = 0;
    std::size_t context_tokens = 0;
    std::size_t prompt_limit_tokens = 0;
};

struct ChatRun {
    /// Queues the request on the worker's Generator (text-only JSON chat, multimodal ChatMessage chat, ...).
    std::function<Result<std::uint64_t>(inference::Generator&)> enqueue;
    /// Polled while waiting; returning true cancels the request.
    std::function<bool()> cancelled;
    /// Receives decoded text as it streams; returning false cancels the request.
    std::function<bool(std::string_view)> on_text;
    /// Frees transient workspaces once the engine is idle (for rare large-context jobs such as ebook translation).
    bool release_workspaces = false;
};

class ChatEngine {
public:
    explicit ChatEngine(inference::Generator& generator);
    ~ChatEngine();
    ChatEngine(const ChatEngine&) = delete;
    auto operator=(const ChatEngine&) -> ChatEngine& = delete;

    /// Runs one request to completion, sharing decode steps with any concurrent requests.
    auto run(ChatRun request) -> Result<inference::TextGeneration>;
    auto context(const nlohmann::ordered_json& messages, const nlohmann::ordered_json& template_args,
                 inference::GenerationOptions options) -> Result<ChatContext>;

private:
    struct Job;
    auto work() -> void;

    inference::Generator& generator_;
    std::mutex generator_mutex_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::shared_ptr<Job>> inbox_;
    bool stopping_ = false;
    std::thread worker_;
};

} // namespace kidi::server
