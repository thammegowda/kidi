#include "kidi/agent/session.h"

#include "kidi/agent/terminal.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stop_token>
#include <utility>

namespace kidi::agent {
namespace {

auto invalid(std::string message) -> std::unexpected<Error> {
    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, std::move(message)});
}

// Drops a trailing tool exchange that a crash or cancellation left without all of its results.
auto drop_incomplete_tools(std::vector<Message>& history) -> void {
    for (auto index = history.size(); index > 0; --index) {
        const auto& message = history[index - 1];
        if (message.role != Role::ASSISTANT || message.tool_calls.empty()) continue;
        std::set<std::string> pending;
        for (const auto& call : message.tool_calls) pending.insert(call.id);
        for (auto later = index; later < history.size(); ++later)
            if (history[later].role == Role::TOOL) pending.erase(history[later].tool_call_id);
        if (!pending.empty()) history.resize(index - 1);
        return;
    }
}

} // namespace

struct AgentSession::Impl {
    SessionOptions options;
    SessionView view;
    std::vector<Message> history;
    Result<Provider> provider = std::unexpected(Error{ErrorCode::RUNTIME, "model not started"});
    bool provider_started = false;
    std::optional<Transcript> transcript;
    std::optional<Error> transcript_error;
    std::set<std::string> session_grants;
    std::optional<ContextUsage> context;
    std::optional<Speed> speed;
    std::string model;
    bool streamed_content = false, streamed_reasoning = false;
    std::mutex stop_mutex;
    std::stop_source stop;
    Callbacks callbacks;

    auto notice(std::string_view text, Notice kind) const -> void {
        if (view.notice) view.notice(text, kind);
    }
    auto busy(std::string_view phase, bool tool = false) const -> void {
        if (view.busy) view.busy(phase, tool);
    }
    auto idle(bool failed) const -> void {
        if (view.idle) view.idle(failed);
    }
    auto show_status(std::string_view phase) const -> void {
        if (view.status) view.status({model, context, speed}, phase);
    }
    auto error(const Error& value) const -> void {
        notice("kidi-agent: " + terminal_text(value.message), Notice::ERROR);
    }

    auto current_stop() -> std::stop_source {
        const std::lock_guard lock(stop_mutex);
        return stop;
    }
    auto request_stop() -> void { current_stop().request_stop(); }

    // Re-arms cancellation when a request ends, so Stop pressed while the model was still starting is not lost.
    struct Rearm {
        Impl& state;
        ~Rearm() {
            const std::lock_guard lock(state.stop_mutex);
            state.stop = std::stop_source{};
        }
    };

    auto save(Json record) -> void {
        if (!transcript || transcript_error) return;
        auto result = transcript->append(std::move(record));
        if (!result) {
            transcript_error = result.error();
            request_stop();
        }
    }

    auto visible_specs() const -> std::vector<ToolSpec> {
        std::vector<ToolSpec> specs;
        for (const auto& tool : options.tools)
            if (options.run.permissions.access(tool.capability) != Access::DENY) specs.push_back(tool.spec);
        return specs;
    }

    // Tell the user up front when the context cannot hold real work, instead of compacting constantly.
    auto check_context() -> void {
        constexpr std::size_t RECOMMENDED = 32768;
        if (!provider->context) return;
        auto usage = provider->context(std::span(history).first(1), visible_specs(), {});
        if (!usage || usage->context_tokens >= RECOMMENDED) return;
        notice("[context] warning: " + std::to_string(usage->context_tokens) +
                   "-token context; the system prompt and tools already use " + std::to_string(usage->prompt_tokens) +
                   ". Long tasks will compact often. Omit --context-size to size it from free memory, or use " +
                   std::to_string(RECOMMENDED) + " or more.",
               Notice::WARNING);
    }

    auto start_provider() -> bool {
        if (provider_started) return static_cast<bool>(provider);
        provider_started = true;
        auto chat = options.chat;
        if (view.stream)
            chat.on_delta = [this](std::string_view text, bool thinking) {
                auto& seen = thinking ? streamed_reasoning : streamed_content;
                if (!seen) {
                    busy(thinking ? "Thinking" : "Answering");
                    seen = true;
                }
                view.stream(text, thinking);
            };
        busy("Starting model");
        provider = options.create_provider(std::move(chat));
        idle(!provider);
        if (!provider) return false;
        if (!provider->model.empty()) model = provider->model;
        auto complete = std::move(provider->complete);
        provider->complete = [this, complete = std::move(complete)](auto messages, auto specs,
                                                                    std::stop_token token) -> Result<Message> {
            const auto started = std::chrono::steady_clock::now();
            streamed_content = streamed_reasoning = false;
            auto reply = complete(messages, specs, token);
            idle(!reply);
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            speed.reset();
            if (!reply) return reply;
            if (reply->usage) {
                const auto& usage = *reply->usage;
                if (usage.decode_ms > 0)
                    speed = Speed{usage.completion_tokens * 1000.0 / usage.decode_ms, usage.prefill_ms / 1000.0, true};
                else if (seconds > 0)
                    speed = Speed{usage.completion_tokens / seconds};
                if (context) context->prompt_tokens = usage.prompt_tokens + usage.completion_tokens;
            }
            save(transcript_message(*reply));
            if (transcript_error) return std::unexpected(*transcript_error);
            if (view.reply) view.reply(*reply, streamed_reasoning, streamed_content);
            return reply;
        };
        if (provider->context) {
            auto inspect = std::move(provider->context);
            provider->context = [this, inspect = std::move(inspect)](auto messages, auto specs,
                                                                     std::stop_token token) -> Result<ContextUsage> {
                busy("Checking context");
                auto usage = inspect(messages, specs, token);
                idle(!usage);
                if (usage) context = *usage;
                return usage;
            };
        }
        check_context();
        return true;
    }

    // Starts the model and the transcript for the next request; fails once the session cannot continue.
    auto prepare() -> Result<void> {
        if (!start_provider()) {
            error(provider.error());
            return std::unexpected(provider.error());
        }
        if (!transcript_error && !transcript && options.save) {
            auto opened = Transcript::open(options.workspace, model);
            if (!opened) {
                transcript_error = opened.error();
                notice("kidi-agent: " + terminal_text(opened.error().message) + " (use --no-save to opt out)",
                       Notice::ERROR);
                return std::unexpected(opened.error());
            }
            transcript.emplace(std::move(*opened));
            save(transcript_message(history.front()));
            if (options.announce_transcript)
                notice("[session] " + terminal_text(transcript->path().string()), Notice::SESSION);
        }
        if (transcript_error) {
            error(*transcript_error);
            return std::unexpected(*transcript_error);
        }
        return {};
    }

    auto approve(const ToolSpec& spec, const ToolCall& call) -> Result<bool> {
        if (session_grants.contains(spec.name)) return true;
        const auto tool =
            std::ranges::find_if(options.tools, [&](const Tool& item) { return item.spec.name == spec.name; });
        const bool unsandboxed = tool != options.tools.end() && tool->capability == Capability::EXECUTE;
        auto decision = view.approve(spec, call, unsandboxed);
        if (!decision) {
            request_stop();
            return std::unexpected(decision.error());
        }
        if (*decision == Approval::SESSION) {
            session_grants.insert(spec.name);
            notice("Session approval granted for " + terminal_text(spec.name) + ". /permissions reset revokes it.",
                   Notice::SESSION);
        }
        return *decision != Approval::DENY;
    }

    auto observe(const Event& event) -> void {
        // A finished phase is closed before the view reports it, so its timing lands on the right line.
        if (event.kind == EventKind::TOOL_FINISH) idle(event.is_error);
        if (event.kind == EventKind::COMPACT_FINISH) idle(false);
        if (event.kind == EventKind::COMPACT_START) busy("Compacting");
        if (event.kind == EventKind::MODEL_START) show_status("Generating");
        if (view.event) view.event(event);
        if (event.kind == EventKind::MODEL_START)
            busy(context ? "Processing " + std::to_string(context->prompt_tokens) + " prompt tokens"
                         : "Waiting for model");
        if (event.kind == EventKind::TOOL_START) {
            show_status("Working");
            busy(event.tool_name, true);
            const auto arguments = Json::parse(event.detail, nullptr, false);
            save({{"type", "tool_start"},
                  {"name", event.tool_name},
                  {"id", event.call_id},
                  {"arguments", arguments.is_discarded() ? Json(event.detail) : arguments}});
        }
        if (event.kind == EventKind::TOOL_FINISH)
            save(transcript_message({.role = Role::TOOL, .content = event.detail, .tool_call_id = event.call_id}));
        if (event.kind == EventKind::COMPACT_FINISH) {
            Json checkpoint{{"type", "compact"}, {"messages", Json::array()}};
            for (const auto& message : history) checkpoint["messages"].push_back(transcript_message(message));
            save(std::move(checkpoint));
        }
    }
};

AgentSession::AgentSession(SessionOptions options, SessionView view) : impl_(std::make_unique<Impl>()) {
    auto& state = *impl_;
    state.options = std::move(options);
    state.view = std::move(view);
    state.model = state.options.chat.model.empty() ? "local model" : state.options.chat.model;
    state.history.push_back({.role = Role::SYSTEM, .content = state.options.system_prompt});
    if (state.view.approve)
        state.callbacks.approve = [&state](const ToolSpec& spec, const ToolCall& call) {
            return state.approve(spec, call);
        };
    state.callbacks.observe = [&state](const Event& event) { state.observe(event); };
}

AgentSession::~AgentSession() = default;

auto AgentSession::run(std::string task, std::vector<std::string> images) -> Result<TurnResult> {
    auto& state = *impl_;
    const Impl::Rearm rearm{state};
    if (auto ready = state.prepare(); !ready) return std::unexpected(ready.error());
    Message user{.role = Role::USER, .content = std::move(task), .images = std::move(images)};
    // A model without vision rejects images on every later request, so refuse them before they enter the history.
    if (!user.images.empty() && state.provider->context) {
        auto request = state.history;
        request.push_back(user);
        if (auto usage = state.provider->context(request, state.visible_specs(), state.current_stop().get_token());
            !usage) {
            state.idle(true);
            state.error(usage.error());
            state.show_status("Ready");
            return std::unexpected(usage.error());
        }
    }
    state.save(transcript_message(user));
    if (state.transcript_error) {
        state.error(*state.transcript_error);
        return std::unexpected(*state.transcript_error);
    }
    auto result = run_turn(*state.provider, state.options.tools, state.history, std::move(user), state.options.run,
                           state.callbacks, state.current_stop().get_token());
    if (state.transcript_error) result = std::unexpected(*state.transcript_error);
    if (!result) {
        state.idle(true);
        state.save({{"type", "error"}, {"message", result.error().message}});
        if (state.transcript_error) result = std::unexpected(*state.transcript_error);
        state.error(result.error());
        state.show_status("Stopped");
        return result;
    }
    state.show_status("Ready");
    return result;
}

auto AgentSession::compact() -> Result<void> {
    auto& state = *impl_;
    const Impl::Rearm rearm{state};
    if (auto ready = state.prepare(); !ready) return ready;
    auto result = compact_history(*state.provider, state.visible_specs(), state.history, state.options.run,
                                  state.callbacks, state.current_stop().get_token());
    state.idle(!result);
    if (result && !state.transcript_error) {
        state.show_status("Ready");
        return result;
    }
    if (state.transcript_error) result = std::unexpected(*state.transcript_error);
    state.error(result.error());
    return result;
}

auto AgentSession::context() -> Result<ContextUsage> {
    auto& state = *impl_;
    const Impl::Rearm rearm{state};
    if (auto ready = state.prepare(); !ready) return std::unexpected(ready.error());
    if (!state.provider->context) {
        state.notice("kidi-agent: provider does not support context inspection", Notice::ERROR);
        return invalid("provider does not support context inspection");
    }
    auto usage = state.provider->context(state.history, state.visible_specs(), state.current_stop().get_token());
    if (!usage) {
        state.error(usage.error());
        return usage;
    }
    state.show_status("Ready");
    return usage;
}

auto AgentSession::reset() -> Result<void> {
    auto& state = *impl_;
    state.history.resize(1);
    state.save({{"type", "reset"}});
    if (state.transcript_error) {
        state.error(*state.transcript_error);
        return std::unexpected(*state.transcript_error);
    }
    state.context.reset();
    state.speed.reset();
    return {};
}

auto AgentSession::resume(const std::filesystem::path& path) -> Result<void> {
    auto& state = *impl_;
    auto directory = sessions_directory(state.options.workspace);
    if (!directory) return std::unexpected(directory.error());
    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(path, error);
    if (error || canonical.parent_path() != std::filesystem::weakly_canonical(*directory, error) ||
        canonical.extension() != ".jsonl")
        return invalid("not a saved conversation of this workspace");
    std::vector<Json> records;
    if (auto read = Transcript::read(canonical, records); !read) return std::unexpected(read.error());
    std::vector<Message> history{state.history.front()};
    for (const auto& record : records) {
        const auto type = record.value("type", "");
        if (type == "reset") history.resize(1);
        if (type == "compact" && record.contains("messages") && record["messages"].is_array()) {
            history.resize(1);
            for (const auto& item : record["messages"]) {
                auto message = message_from_record(item);
                if (message && message->role != Role::SYSTEM) history.push_back(std::move(*message));
            }
        }
        if (type != "message") continue;
        auto message = message_from_record(record);
        if (!message) return std::unexpected(message.error());
        if (message->role != Role::SYSTEM) history.push_back(std::move(*message));
    }
    drop_incomplete_tools(history);
    if (state.options.save) {
        auto reopened = Transcript::reopen(canonical);
        if (!reopened) return std::unexpected(reopened.error());
        state.transcript.emplace(std::move(*reopened));
        state.transcript_error.reset();
    }
    state.history = std::move(history);
    state.context.reset();
    state.speed.reset();
    return {};
}

auto AgentSession::cancel() -> void { impl_->request_stop(); }
auto AgentSession::history() const -> const std::vector<Message>& { return impl_->history; }
auto AgentSession::status() const -> SessionStatus { return {impl_->model, impl_->context, impl_->speed}; }
auto AgentSession::grants() const -> const std::set<std::string>& { return impl_->session_grants; }
auto AgentSession::revoke_grants() -> void { impl_->session_grants.clear(); }
auto AgentSession::set_system_prompt(std::string prompt) -> void {
    impl_->options.system_prompt = prompt;
    impl_->history.front().content = std::move(prompt);
}

auto AgentSession::mode() const -> const std::string& { return impl_->options.mode; }
auto AgentSession::tools() const -> const std::vector<Tool>& { return impl_->options.tools; }
auto AgentSession::failed() const -> bool {
    return (impl_->provider_started && !impl_->provider) || impl_->transcript_error.has_value();
}
auto AgentSession::streamed_answer() const -> bool { return impl_->streamed_content; }
auto AgentSession::transcript_path() const -> std::optional<std::filesystem::path> {
    if (!impl_->transcript) return std::nullopt;
    return impl_->transcript->path();
}

auto sessions_directory(const std::filesystem::path& workspace) -> Result<std::filesystem::path> {
    return Transcript::directory(workspace);
}

auto list_sessions(const std::filesystem::path& workspace) -> Result<std::vector<SessionSummary>> {
    auto directory = sessions_directory(workspace);
    if (!directory) return std::unexpected(directory.error());
    std::vector<SessionSummary> sessions;
    std::error_code error;
    if (!std::filesystem::is_directory(*directory, error)) return sessions;
    for (const auto& entry : std::filesystem::directory_iterator(*directory, error)) {
        if (entry.path().extension() != ".jsonl" || !entry.is_regular_file(error)) continue;
        std::vector<Json> records;
        if (!Transcript::read(entry.path(), records)) continue;
        SessionSummary summary{.path = entry.path()};
        for (const auto& record : records) {
            if (!summary.time_ns) summary.time_ns = record.value("time_ns", std::int64_t{0});
            if (record.value("type", "") != "message") continue;
            const auto role = record.value("role", "");
            if (role == "user" || role == "assistant") ++summary.messages;
            if (role == "user" && summary.title.empty()) summary.title = record.value("content", "");
        }
        if (summary.messages) sessions.push_back(std::move(summary));
    }
    std::ranges::sort(sessions, std::ranges::greater{}, &SessionSummary::time_ns);
    return sessions;
}

} // namespace kidi::agent
