#pragma once

#include "kidi/agent/chat_client.h"
#include "kidi/agent/transcript.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace kidi::agent {

using ProviderFactory = std::function<Result<Provider>(ChatClientOptions)>;

enum class Notice { INFO, CONTEXT, SESSION, WARNING, ERROR };
enum class Approval { DENY, ONCE, SESSION };

struct Speed {
    double tokens_per_second = 0, prefill_seconds = 0;
    bool decode_only = false;
};

struct SessionStatus {
    std::string model;
    std::optional<ContextUsage> context;
    std::optional<Speed> speed;
};

// How a front end follows a session. Every hook is optional and runs on the thread that called into the session.
struct SessionView {
    std::function<void(std::string_view text, bool thinking)> stream; // streamed reply text
    std::function<void(std::string_view phase, bool tool)> busy;      // a timed phase starts
    std::function<void(bool failed)> idle;                            // that phase ends
    std::function<void(const Event&)> event;                          // tool and compaction events
    // Each finished model reply, with whether its reasoning and answer were already streamed.
    std::function<void(const Message& reply, bool streamed_reasoning, bool streamed_content)> reply;
    std::function<void(std::string_view text, Notice kind)> notice;
    std::function<void(const SessionStatus& status, std::string_view phase)> status;
    // Asks the user about a tool call that needs approval; without it such calls are denied.
    std::function<Result<Approval>(const ToolSpec& spec, const ToolCall& call, bool unsandboxed)> approve;
};

struct SessionOptions {
    ChatClientOptions chat;
    RunOptions run;
    std::vector<Tool> tools;
    std::string system_prompt;
    std::filesystem::path workspace;
    std::string mode; // the --permissions mode name, for display
    bool save = true;
    bool announce_transcript = false;
    ProviderFactory create_provider;
};

struct SessionSummary {
    std::filesystem::path path;
    std::string title; // the first user message
    std::int64_t time_ns = 0;
    std::size_t messages = 0;
};

// One conversation with the agent: history, the model provider, the transcript and session approvals. A turn runs
// synchronously on the caller's thread; cancel() may be called from any thread.
class AgentSession {
public:
    AgentSession(SessionOptions options, SessionView view);
    ~AgentSession();
    AgentSession(const AgentSession&) = delete;
    auto operator=(const AgentSession&) -> AgentSession& = delete;

    // images: data:image/...;base64 URLs for vision models; refused up front when the model cannot read them.
    auto run(std::string task, std::vector<std::string> images = {}) -> Result<TurnResult>;
    auto compact() -> Result<void>;
    auto context() -> Result<ContextUsage>;
    auto reset() -> Result<void>;
    // Continues a saved conversation of this workspace; later messages are appended to the same transcript.
    auto resume(const std::filesystem::path& transcript) -> Result<void>;
    auto cancel() -> void;
    // Replaces the system message; the next model request uses it.
    auto set_system_prompt(std::string prompt) -> void;

    auto history() const -> const std::vector<Message>&;
    auto status() const -> SessionStatus;
    auto grants() const -> const std::set<std::string>&;
    auto revoke_grants() -> void;
    auto mode() const -> const std::string&;
    auto tools() const -> const std::vector<Tool>&;
    auto transcript_path() const -> std::optional<std::filesystem::path>;
    // True once the model cannot start or the transcript cannot be written; the session cannot continue.
    auto failed() const -> bool;
    // Whether the last reply's answer text was streamed through the view.
    auto streamed_answer() const -> bool;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Saved conversations of a workspace, newest first.
auto list_sessions(const std::filesystem::path& workspace) -> Result<std::vector<SessionSummary>>;
// The cache directory that holds a workspace's transcripts.
auto sessions_directory(const std::filesystem::path& workspace) -> Result<std::filesystem::path>;

} // namespace kidi::agent
