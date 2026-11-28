#include "kidi/agent/web.h"

#include "kidi/agent_web_assets.h"

#include <liteapp/app/desktop.h>
#include <liteapp/core/base64.h>
#include <liteapp/core/random.h>
#include <liteapp/http/server.h>
#include <liteapp/service/event_hub.h>
#include <liteapp/service/router.h>
#include <liteapp/workspace/services.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace kidi::agent {
namespace {

using JsonValue = nlohmann::json;
using liteapp::ServiceContext;
using liteapp::ServiceResponder;

constexpr std::size_t MAX_FILE_BYTES = 4 * 1024 * 1024;
constexpr std::size_t MAX_TOOL_RESULT_BYTES = 64 * 1024;
constexpr std::size_t MAX_SYSTEM_PROMPT_BYTES = 64 * 1024;
constexpr std::size_t MAX_IMAGES = 8;
constexpr std::size_t MAX_DICTATION_BYTES = 2 * 60 * 16000 * 2;
constexpr std::size_t MAX_IMAGE_BYTES = 24 * 1024 * 1024; // base64 text; the server allows 32 MiB of image bytes

auto failure(std::string code, std::string message) -> std::unexpected<liteapp::Error> {
    return std::unexpected(liteapp::Error{std::move(code), std::move(message)});
}

auto failure(const Error& error) -> std::unexpected<liteapp::Error> {
    const char* code = error.code == ErrorCode::UNSUPPORTED        ? "unsupported"
                       : error.code == ErrorCode::INVALID_ARGUMENT ? "invalid"
                       : error.message.starts_with("file changed") ? "conflict"
                                                                   : "failed";
    return failure(code, error.message);
}

auto clip(std::string text, std::size_t limit) -> std::string {
    if (text.size() <= limit) return text;
    auto end = limit;
    while (end && (static_cast<unsigned char>(text[end]) & 0xc0) == 0x80) --end;
    const auto dropped = text.size() - end;
    text.resize(end);
    return text + "\n… (" + std::to_string(dropped) + " more bytes)";
}

auto status_json(const SessionStatus& status) -> JsonValue {
    JsonValue result{{"model", status.model}};
    if (status.context)
        result["context"] = {{"prompt_tokens", status.context->prompt_tokens},
                             {"context_tokens", status.context->context_tokens}};
    if (status.speed)
        result["speed"] = {{"tokens_per_second", status.speed->tokens_per_second},
                           {"prefill_seconds", status.speed->prefill_seconds},
                           {"decode_only", status.speed->decode_only}};
    return result;
}

auto history_json(const std::vector<Message>& history) -> JsonValue {
    auto messages = JsonValue::array();
    for (const auto& message : history) {
        if (message.role == Role::SYSTEM) continue;
        auto record = JsonValue(transcript_message(message));
        record.erase("type");
        if (message.role == Role::TOOL) {
            record["content"] = clip(record["content"].get<std::string>(), MAX_TOOL_RESULT_BYTES);
        }
        messages.push_back(std::move(record));
    }
    return messages;
}

// The images attached to a chat.send payload: up to eight data:image/...;base64 URLs within MAX_IMAGE_BYTES.
auto message_images(const JsonValue& payload) -> liteapp::Result<std::vector<std::string>> {
    std::vector<std::string> images;
    if (!payload.contains("images")) return images;
    const auto& list = payload["images"];
    std::size_t bytes = 0;
    if (!list.is_array() || list.size() > MAX_IMAGES) return failure("invalid", "attach at most eight images");
    for (const auto& image : list) {
        if (!image.is_string()) return failure("invalid", "images must be data URLs");
        auto url = image.get<std::string>();
        const auto comma = url.find(',');
        if (!url.starts_with("data:image/") || comma == std::string::npos ||
            url.substr(0, comma).find(";base64") == std::string::npos)
            return failure("invalid", "images must be base64 data:image/... URLs");
        bytes += url.size();
        if (bytes > MAX_IMAGE_BYTES) return failure("invalid", "images exceed 24 MiB in total");
        images.push_back(std::move(url));
    }
    return images;
}

// The "path" a file-writing tool call targets, when its arguments look like one.
auto written_path(const liteapp::workspace::Workspace& workspace, const Json& arguments)
    -> std::optional<std::string> {
    if (!arguments.is_object() || !arguments.contains("path") || !arguments["path"].is_string() ||
        !arguments.contains("content") || !arguments["content"].is_string())
        return std::nullopt;
    auto path = workspace.normalize(arguments["path"].get<std::string>(), false);
    if (!path) return std::nullopt;
    return *path;
}

class WebAgent {
public:
    WebAgent(SessionOptions options, Speech speech, liteapp::workspace::Workspace workspace)
        : options_(std::move(options)), default_system_prompt_(options_.system_prompt), speech_(std::move(speech)),
          files_(std::move(workspace),
                 {.files_written = [this](const std::vector<std::string>& paths) {
                     events_.publish("files_changed", {{"paths", paths}, {"source", "editor"}}, false);
                 }}) {
        session_ = make_session();
        history_ = history_json(session_->history());
    }

    ~WebAgent() { shutdown(); }

    auto register_methods(liteapp::ServiceRouter& router) -> void {
        // A new subscriber first gets a "sync" event with the conversation and whether a turn is running, then the
        // running turn replayed. Both come from one critical section with turn completion, so a page that connects
        // at any moment neither misses nor repeats part of a turn.
        router.register_async_method("events.subscribe", [this](ServiceContext&, ServiceResponder responder) {
            const std::lock_guard lock(state_mutex_);
            static_cast<void>(responder.emit("sync", {{"busy", turn_running_}, {"messages", history_}}));
            events_.subscribe(std::move(responder));
        });
        files_.register_methods(router);
        router.register_method("app.info", [this](ServiceContext&) { return info(); });
        router.register_method("app.ping", [](ServiceContext&) -> liteapp::Result<JsonValue> {
            return JsonValue::object();
        });
        router.register_method("chat.send", [this](ServiceContext& context) -> liteapp::Result<JsonValue> {
            auto text = context.payload().value("text", std::string{});
            if (text.find_first_not_of(" \t\r\n") == std::string::npos) return failure("invalid", "empty message");
            auto images = message_images(context.payload());
            if (!images) return std::unexpected(images.error());
            return start_turn(text, *images, [this, text, images = *images]() mutable -> JsonValue {
                auto result = session_->run(std::move(text), std::move(images));
                if (!result) return {{"ok", false}, {"error", result.error().message}};
                return {{"ok", true}, {"text", result->text}, {"streamed", session_->streamed_answer()}};
            });
        });
        router.register_method("chat.compact", [this](ServiceContext&) -> liteapp::Result<JsonValue> {
            return start_turn({}, {}, [this]() -> JsonValue {
                auto result = session_->compact();
                if (!result) return {{"ok", false}, {"error", result.error().message}};
                return {{"ok", true}, {"compacted", true}};
            });
        });
        router.register_method("chat.cancel", [this](ServiceContext&) -> liteapp::Result<JsonValue> {
            cancel();
            return JsonValue::object();
        });
        router.register_method("approval.respond", [this](ServiceContext& context) -> liteapp::Result<JsonValue> {
            const auto& payload = context.payload();
            const auto decision = payload.value("decision", std::string{});
            if (decision != "once" && decision != "session" && decision != "deny")
                return failure("invalid", "decision must be once, session or deny");
            const std::lock_guard lock(approval_mutex_);
            if (!pending_ || pending_->id != payload.value("id", std::string{}))
                return failure("stale", "that approval is no longer pending");
            pending_->decision = decision == "once" ? Approval::ONCE
                                 : decision == "session" ? Approval::SESSION
                                                         : Approval::DENY;
            approval_ready_.notify_all();
            return JsonValue::object();
        });
        router.register_method("chat.context", [this](ServiceContext&) -> liteapp::Result<JsonValue> {
            auto idle = idle_session();
            if (!idle) return std::unexpected(idle.error());
            auto usage = session_->context();
            if (!usage) return failure(usage.error());
            return JsonValue{{"prompt_tokens", usage->prompt_tokens}, {"context_tokens", usage->context_tokens}};
        });
        router.register_method("chat.reset", [this](ServiceContext&) -> liteapp::Result<JsonValue> {
            auto idle = idle_session();
            if (!idle) return std::unexpected(idle.error());
            if (auto reset = session_->reset(); !reset) return failure(reset.error());
            return JsonValue::object();
        });
        router.register_method("chat.new", [this](ServiceContext&) -> liteapp::Result<JsonValue> {
            auto idle = idle_session();
            if (!idle) return std::unexpected(idle.error());
            replace_session(make_session());
            const std::lock_guard lock(changes_mutex_);
            baselines_.clear();
            return JsonValue::object();
        });
        router.register_method("permissions.get", [this](ServiceContext&) -> liteapp::Result<JsonValue> {
            auto idle = idle_session();
            if (!idle) return std::unexpected(idle.error());
            return JsonValue{{"mode", session_->mode()}, {"grants", session_->grants()}};
        });
        router.register_method("permissions.reset", [this](ServiceContext&) -> liteapp::Result<JsonValue> {
            auto idle = idle_session();
            if (!idle) return std::unexpected(idle.error());
            session_->revoke_grants();
            return JsonValue{{"mode", session_->mode()}, {"grants", JsonValue::array()}};
        });
        router.register_method("settings.get", [this](ServiceContext&) -> liteapp::Result<JsonValue> {
            auto idle = idle_session();
            if (!idle) return std::unexpected(idle.error());
            return JsonValue{{"system_prompt", options_.system_prompt},
                             {"default_system_prompt", default_system_prompt_}};
        });
        // Applies to the current conversation and to new ones until the server stops; agent.yaml is unchanged.
        router.register_method("settings.update", [this](ServiceContext& context) -> liteapp::Result<JsonValue> {
            auto text = context.payload().value("system_prompt", std::string{});
            if (text.size() > MAX_SYSTEM_PROMPT_BYTES) return failure("invalid", "system prompt is too long");
            if (text.find_first_not_of(" \t\r\n") == std::string::npos) text = default_system_prompt_;
            auto idle = idle_session();
            if (!idle) return std::unexpected(idle.error());
            options_.system_prompt = text;
            session_->set_system_prompt(std::move(text));
            return JsonValue{{"system_prompt", options_.system_prompt}};
        });
        router.register_method("speech.transcribe", [this](ServiceContext& context) -> liteapp::Result<JsonValue> {
            return transcribe(context);
        });
        router.register_async_method("speech.speak", [this](ServiceContext& context, ServiceResponder responder) {
            speak(context.payload().value("text", std::string{}), std::move(responder));
        });
        router.register_method("sessions.list", [this](ServiceContext&) { return sessions(); });
        router.register_method("sessions.open", [this](ServiceContext& context) -> liteapp::Result<JsonValue> {
            auto idle = idle_session();
            if (!idle) return std::unexpected(idle.error());
            const auto id = context.payload().value("id", std::string{});
            auto directory = sessions_directory(options_.workspace);
            if (!directory) return failure(directory.error());
            if (id.empty() || id.find('/') != std::string::npos || id.starts_with('.'))
                return failure("invalid", "invalid conversation id");
            auto session = make_session();
            if (auto resumed = session->resume(*directory / id); !resumed) return failure(resumed.error());
            replace_session(std::move(session));
            {
                const std::lock_guard lock(changes_mutex_);
                baselines_.clear();
            }
            return JsonValue{{"messages", history_json(session_->history())}};
        });
        router.register_method("changes.list", [this](ServiceContext&) { return changes(); });
        router.register_method("changes.diff", [this](ServiceContext& context) -> liteapp::Result<JsonValue> {
            auto path = workspace().normalize(context.payload().value("path", std::string{}), false);
            if (!path) return std::unexpected(path.error());
            return diff(*path, context.payload().value("base", std::string("git")));
        });
    }

    auto shutdown() -> void {
        cancel();
        if (turn_thread_.joinable()) turn_thread_.join();
        files_.close();
        {
            const std::lock_guard lock(speech_mutex_);
            if (speech_stop_) *speech_stop_ = true;
            if (speech_thread_.joinable()) speech_thread_.join();
        }
        events_.close();
    }

private:
    auto workspace() const -> const liteapp::workspace::Workspace& { return files_.workspace(); }

    struct PendingApproval {
        std::string id;
        JsonValue payload;
        std::optional<Approval> decision;
        bool cancelled = false;
    };

    auto make_session() -> std::unique_ptr<AgentSession> {
        SessionView view;
        view.stream = [this](std::string_view text, bool thinking) {
            broadcast("delta", {{"text", text}, {"thinking", thinking}});
        };
        view.busy = [this](std::string_view phase, bool tool) {
            broadcast("busy", {{"phase", phase}, {"tool", tool}});
        };
        view.idle = [this](bool failed) { broadcast("idle", {{"failed", failed}}); };
        view.reply = [this](const Message& reply, bool streamed_reasoning, bool streamed_content) {
            auto record = JsonValue(transcript_message(reply));
            record.erase("type");
            record["streamed_reasoning"] = streamed_reasoning;
            record["streamed_content"] = streamed_content;
            broadcast("reply", std::move(record));
        };
        view.notice = [this](std::string_view text, Notice kind) {
            static constexpr const char* KINDS[] = {"info", "context", "session", "warning", "error"};
            broadcast("notice", {{"text", text}, {"kind", KINDS[static_cast<int>(kind)]}});
        };
        view.status = [this](const SessionStatus& status, std::string_view phase) {
            auto json = status_json(status);
            json["phase"] = phase;
            {
                const std::lock_guard lock(state_mutex_);
                last_status_ = json;
            }
            broadcast("status", std::move(json));
        };
        view.event = [this](const Event& event) { observe(event); };
        view.approve = [this](const ToolSpec& spec, const ToolCall& call, bool unsandboxed) {
            return approve(spec, call, unsandboxed);
        };
        return std::make_unique<AgentSession>(options_, std::move(view));
    }

    auto observe(const Event& event) -> void {
        if (event.kind == EventKind::TOOL_START) {
            auto arguments = Json::parse(event.detail, nullptr, false);
            if (const auto path = written_path(workspace(), arguments)) {
                remember_baseline(*path);
                const std::lock_guard lock(changes_mutex_);
                tool_paths_[event.call_id] = *path;
            }
            auto shown = arguments.is_discarded() ? JsonValue(event.detail) : JsonValue(arguments);
            broadcast("tool", {{"id", event.call_id}, {"name", event.tool_name}, {"arguments", std::move(shown)}});
        }
        if (event.kind == EventKind::TOOL_FINISH) {
            broadcast("tool_result", {{"id", event.call_id},
                                      {"name", event.tool_name},
                                      {"content", clip(event.detail, MAX_TOOL_RESULT_BYTES)},
                                      {"error", event.is_error}});
            std::optional<std::string> path;
            {
                const std::lock_guard lock(changes_mutex_);
                if (const auto found = tool_paths_.find(event.call_id); found != tool_paths_.end()) {
                    path = found->second;
                    tool_paths_.erase(found);
                }
            }
            // Commands may change any file, so they refresh everything.
            if (path || (event.tool_name != "read_file" && event.tool_name != "list_files"))
                broadcast("files_changed", {{"paths", path ? JsonValue::array({*path}) : JsonValue(nullptr)},
                                            {"source", "agent"}});
        }
        if (event.kind == EventKind::COMPACT_START) broadcast("compact", {{"started", true}});
        if (event.kind == EventKind::COMPACT_FINISH) {
            broadcast("compact", {{"started", false}, {"detail", event.detail}});
        }
    }

    auto approve(const ToolSpec& spec, const ToolCall& call, bool unsandboxed) -> Result<Approval> {
        JsonValue payload{{"id", std::to_string(++approvals_)},
                          {"tool", spec.name},
                          {"arguments", JsonValue(call.arguments)},
                          {"unsandboxed", unsandboxed}};
        if (const auto path = written_path(workspace(), call.arguments)) {
            auto current = workspace().read(*path);
            payload["diff"] = {{"path", *path},
                               {"old", current ? current->content : std::string{}},
                               {"exists", current.has_value()},
                               {"new", call.arguments["content"].get<std::string>()}};
        }
        const auto id = payload["id"].get<std::string>();
        std::unique_lock lock(approval_mutex_);
        if (turn_cancelled_) return std::unexpected(Error{ErrorCode::RUNTIME, "approval cancelled"});
        pending_ = PendingApproval{.id = id, .payload = payload};
        lock.unlock();
        broadcast("approval", payload);
        lock.lock();
        approval_ready_.wait(lock, [&] { return pending_->decision || pending_->cancelled || turn_cancelled_; });
        const auto decision = pending_->decision;
        pending_.reset();
        lock.unlock();
        broadcast("approval_done", {{"id", id}, {"decision", !decision                      ? "cancelled"
                                                             : *decision == Approval::DENY ? "deny"
                                                                                           : "allow"}});
        if (!decision) return std::unexpected(Error{ErrorCode::RUNTIME, "approval cancelled"});
        return *decision;
    }

    // Stops the running turn. Only a running turn is cancelled, so a late Stop cannot cancel the next one.
    auto cancel() -> void {
        {
            const std::lock_guard lock(state_mutex_);
            if (!turn_running_) return;
            session_->cancel();
        }
        const std::lock_guard lock(approval_mutex_);
        turn_cancelled_ = true;
        if (pending_) pending_->cancelled = true;
        approval_ready_.notify_all();
    }

    // Swaps in a new session; the old one is destroyed outside the lock that cancel() takes.
    auto replace_session(std::unique_ptr<AgentSession> session) -> void {
        {
            const std::lock_guard lock(state_mutex_);
            session_.swap(session);
        }
        session.reset();
    }

    // Holds the session exclusively for a short request, as a turn does for its whole run.
    struct Claim {
        WebAgent* owner = nullptr;
        Claim(WebAgent* value) : owner(value) {}
        Claim(Claim&& other) noexcept : owner(std::exchange(other.owner, nullptr)) {}
        ~Claim() {
            if (!owner) return;
            const std::lock_guard lock(owner->state_mutex_);
            owner->history_ = history_json(owner->session_->history());
            owner->busy_ = false;
        }
    };

    auto idle_session() -> liteapp::Result<Claim> {
        const std::lock_guard lock(state_mutex_);
        if (busy_) return failure("busy", "the agent is working; stop it first");
        busy_ = true;
        return Claim(this);
    }

    auto start_turn(const std::string& text, const std::vector<std::string>& images, std::function<JsonValue()> work)
        -> liteapp::Result<JsonValue> {
        const std::lock_guard lock(state_mutex_);
        if (busy_) return failure("busy", "the agent is already working");
        busy_ = true;
        turn_running_ = true;
        {
            const std::lock_guard approvals(approval_mutex_);
            turn_cancelled_ = false;
        }
        if (turn_thread_.joinable()) turn_thread_.join();
        events_.begin_recording();
        if (!text.empty()) broadcast("user", {{"text", text}, {"images", images}});
        turn_thread_ = std::jthread([this, work = std::move(work)] {
            auto result = work();
            result["status"] = status_json(session_->status());
            const std::lock_guard lock(state_mutex_);
            busy_ = false;
            turn_running_ = false;
            history_ = history_json(session_->history());
            broadcast("done", result);
            events_.end_recording();
        });
        return JsonValue{{"started", true}};
    }

    // Dictation: base64 pcm_s16le mono 16 kHz audio in, text out.
    auto transcribe(ServiceContext& context) -> liteapp::Result<JsonValue> {
        if (!speech_.transcribe) return failure("unsupported", "this build has no speech recognition");
        const auto audio = liteapp::base64_decode(context.payload().value("audio", std::string{}));
        if (!audio || audio->size() % 2) return failure("invalid", "audio must be base64 16-bit PCM");
        if (audio->size() > MAX_DICTATION_BYTES) return failure("invalid", "dictation is limited to two minutes");
        std::vector<float> samples(audio->size() / 2);
        for (std::size_t index = 0; index < samples.size(); ++index) {
            std::int16_t value;
            std::memcpy(&value, audio->data() + 2 * index, 2);
            samples[index] = static_cast<float>(value) / 32768.F;
        }
        auto text = speech_.transcribe(samples, [&context] { return context.cancelled(); });
        if (!text) return failure(text.error());
        return JsonValue{{"text", std::move(*text)}};
    }

    // Read-aloud: streams "audio" events (base64 pcm_s16le chunks). A new request stops the previous one.
    auto speak(std::string text, ServiceResponder responder) -> void {
        if (!speech_.speak || text.find_first_not_of(" \t\r\n") == std::string::npos || text.size() > 64 * 1024) {
            const auto reason = speech_.speak ? "nothing to read" : "this build has no TTS";
            static_cast<void>(responder.fail(liteapp::Error{"invalid", reason}));
            return;
        }
        const std::lock_guard lock(speech_mutex_);
        if (speech_stop_) *speech_stop_ = true;
        if (speech_thread_.joinable()) speech_thread_.join();
        auto stop = std::make_shared<std::atomic<bool>>(false);
        speech_stop_ = stop;
        auto shared = std::make_shared<ServiceResponder>(std::move(responder));
        speech_thread_ = std::jthread([this, text = std::move(text), stop, shared] {
            const auto cancelled = [&] { return *stop || shared->cancelled(); };
            auto spoken = speech_.speak(
                text,
                [&](std::span<const float> samples, std::uint32_t sample_rate) {
                    std::string pcm(samples.size() * 2, '\0');
                    for (std::size_t index = 0; index < samples.size(); ++index) {
                        const auto value = static_cast<std::int16_t>(std::clamp(samples[index], -1.F, 1.F) * 32767.F);
                        std::memcpy(pcm.data() + 2 * index, &value, 2);
                    }
                    return !cancelled() &&
                           shared->emit("audio", {{"data", liteapp::base64_encode(pcm)}, {"sample_rate", sample_rate}});
                },
                cancelled);
            if (!spoken) {
                static_cast<void>(shared->fail(liteapp::Error{"speech", spoken.error().message}));
                return;
            }
            static_cast<void>(shared->complete());
        });
    }

    // Session events; while a turn runs they are recorded, so a page that connects mid-turn sees all of it.
    auto broadcast(std::string_view type, const JsonValue& payload) -> void {
        events_.publish(type, payload, type != "files_changed");
    }

    auto info() -> liteapp::Result<JsonValue> {
        std::error_code error;
        const auto workspace = std::filesystem::canonical(options_.workspace, error);
        JsonValue result{{"workspace", (error ? options_.workspace : workspace).string()},
                         {"name", (error ? options_.workspace : workspace).filename().string()},
                         {"mode", options_.mode},
                         {"git", files_.git().has_value()},
                         {"speech", {{"transcribe", static_cast<bool>(speech_.transcribe)},
                                     {"speak", static_cast<bool>(speech_.speak)}}}};
        {
            const std::lock_guard lock(state_mutex_);
            result["busy"] = turn_running_;
            result["status"] = busy_ ? last_status_ : status_json(session_->status());
            const auto path = busy_ ? std::nullopt : session_->transcript_path();
            result["conversation"] = path ? JsonValue(path->filename().string()) : JsonValue(nullptr);
        }
        const std::lock_guard lock(approval_mutex_);
        result["approval"] = pending_ ? pending_->payload : JsonValue(nullptr);
        return result;
    }

    auto sessions() -> liteapp::Result<JsonValue> {
        auto list = list_sessions(options_.workspace);
        if (!list) return failure(list.error());
        std::optional<std::filesystem::path> current;
        {
            const std::lock_guard lock(state_mutex_);
            if (!busy_) current = session_->transcript_path();
        }
        auto result = JsonValue::array();
        for (const auto& session : *list)
            result.push_back({{"id", session.path.filename().string()},
                              {"title", clip(session.title, 200)},
                              {"time_ns", session.time_ns},
                              {"messages", session.messages},
                              {"current", current && *current == session.path}});
        return JsonValue{{"sessions", std::move(result)}};
    }

    auto remember_baseline(const std::string& path) -> void {
        {
            const std::lock_guard lock(changes_mutex_);
            if (baselines_.contains(path)) return;
        }
        auto current = workspace().read(path);
        if (!current && current.error().code != "io") return; // binary or too large: not diffable
        const std::lock_guard lock(changes_mutex_);
        baselines_.try_emplace(path, current ? std::optional<std::string>(std::move(current->content)) : std::nullopt);
    }

    auto changes() -> liteapp::Result<JsonValue> {
        auto session = JsonValue::array();
        std::map<std::string, std::optional<std::string>> baselines;
        {
            const std::lock_guard lock(changes_mutex_);
            baselines = baselines_;
        }
        for (const auto& [path, before] : baselines) {
            auto current = workspace().read(path);
            const bool exists = current.has_value() || current.error().code != "io";
            if (!before && !exists) continue;
            if (before && current && *before == current->content) continue;
            session.push_back({{"path", path}, {"status", !before ? "added" : !exists ? "deleted" : "modified"}});
        }
        auto git = JsonValue::array();
        if (files_.git())
            for (auto& change : files_.git()->status())
                git.push_back({{"path", std::move(change.path)}, {"status", std::move(change.status)}});
        return JsonValue{
            {"session", std::move(session)}, {"git", std::move(git)}, {"repository", files_.git().has_value()}};
    }

    auto diff(const std::string& path, const std::string& base) -> liteapp::Result<JsonValue> {
        JsonValue result{{"path", path}, {"base", base}};
        if (base == "session") {
            const std::lock_guard lock(changes_mutex_);
            const auto found = baselines_.find(path);
            if (found == baselines_.end()) return failure("invalid", "the agent has not changed " + path);
            result["old"] = found->second.value_or("");
            result["old_exists"] = found->second.has_value();
        } else {
            if (!files_.git()) return failure("unsupported", "the workspace is not a Git repository");
            auto old = files_.git()->head_text(path);
            result["old"] = old.value_or("");
            result["old_exists"] = old.has_value();
        }
        auto current = workspace().read(path);
        if (!current && current.error().code != "io") return std::unexpected(current.error());
        result["new"] = current ? current->content : std::string{};
        result["new_exists"] = current.has_value();
        result["modified_ns"] = std::to_string(current ? current->modified_ns : 0);
        return result;
    }

    SessionOptions options_;
    const std::string default_system_prompt_;
    const Speech speech_;
    std::mutex speech_mutex_;
    std::shared_ptr<std::atomic<bool>> speech_stop_;
    std::jthread speech_thread_;
    liteapp::EventHub events_;
    liteapp::workspace::WorkspaceServices files_; // fs.*, git.* and terminal.* for the editor and terminal

    std::mutex state_mutex_;
    std::unique_ptr<AgentSession> session_;
    bool busy_ = false, turn_running_ = false;
    JsonValue history_ = JsonValue::array(); // the conversation as of the last finished turn or request
    JsonValue last_status_ = JsonValue::object();
    std::jthread turn_thread_;

    std::mutex approval_mutex_;
    std::condition_variable approval_ready_;
    std::optional<PendingApproval> pending_;
    bool turn_cancelled_ = false;
    std::atomic<std::uint64_t> approvals_{0};

    std::mutex changes_mutex_;
    std::map<std::string, std::optional<std::string>> baselines_;
    std::map<std::string, std::string> tool_paths_;
};

// Serves the UI from a checkout, re-read per request: <directory>/assets.txt maps routes to paths in the repository
// two levels up, exactly as the build embeds them.
auto serve_asset_sources(liteapp::HttpServer& server, const std::filesystem::path& directory)
    -> liteapp::Result<void> {
    std::error_code error;
    const auto web = std::filesystem::canonical(directory, error);
    if (error) return std::unexpected(liteapp::Error{"invalid_assets", "no such directory: " + directory.string()});
    return server.serve_manifest(web.parent_path().parent_path(), web / "assets.txt");
}

} // namespace

auto web_agent(SessionOptions options, Speech speech, std::int32_t port, const std::filesystem::path& assets_directory)
    -> int {
    auto workspace = liteapp::workspace::Workspace::open(
        options.workspace, {.limits = {.max_file_bytes = MAX_FILE_BYTES,
                                       .max_directory_entries = 20000,
                                       .max_command_output = 2 * MAX_FILE_BYTES,
                                       .command_timeout = std::chrono::seconds(20)},
                            .temporary_prefix = ".kidi-agent-"});
    if (!workspace) {
        std::cerr << "kidi-agent: " << workspace.error().message << '\n';
        return 1;
    }
    const auto token = liteapp::secure_random_token();
    if (!token) {
        std::cerr << "kidi-agent: cannot create an access token: " << token.error().message << '\n';
        return 1;
    }
    auto created = liteapp::HttpServer::create({
        .host = "127.0.0.1",
        .port = port,
        .max_body_bytes = 64 * 1024,
        .local_access = liteapp::LocalAccess{.token = *token, .cookie_name = "kidi_agent_access"},
        // CodeMirror and xterm.js insert their own style sheets; scripts stay limited to the embedded files.
        .content_security_policy =
            "default-src 'self'; base-uri 'none'; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; "
            "img-src 'self' data: blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self' data:",
    });
    if (!created) {
        std::cerr << "kidi-agent: cannot create the web server: " << created.error().message << '\n';
        return 1;
    }
    auto server = std::move(*created);
    auto served = assets_directory.empty() ? server.serve_archive(kidi::agent_web_assets::archive())
                                           : serve_asset_sources(server, assets_directory);
    if (!served) {
        std::cerr << "kidi-agent: cannot serve the web UI: " << served.error().message << '\n';
        return 1;
    }
    std::error_code canonical_error;
    auto workspace_path = std::filesystem::canonical(options.workspace, canonical_error);
    if (canonical_error) workspace_path = options.workspace;
    options.announce_transcript = false; // History lists saved conversations
    WebAgent agent(std::move(options), std::move(speech), std::move(*workspace));
    liteapp::ServiceRouter router({.max_active_requests = 64});
    agent.register_methods(router);
    server.websocket("/ws", [&router](const liteapp::Request&, liteapp::WebSocket& socket) { router.run(socket); },
                     {.max_text_bytes = MAX_IMAGE_BYTES + 3 * MAX_FILE_BYTES, .max_binary_bytes = 64 * 1024});

    liteapp::DesktopApplicationOptions desktop;
    desktop.app_id = "kidi-agent";
    desktop.title = "Kidi Agent";
    desktop.on_ready = [&workspace_path](std::string_view url) {
        std::cerr << "Kidi Agent web UI for " << workspace_path.string() << ":\n  " << url
                  << "\nThe link works once; open it in one browser tab. Ctrl-C stops the server.\n"
                  << std::flush;
    };
    desktop.on_stopping = [&agent] {
        std::cerr << "Stopping the web UI...\n";
        agent.shutdown();
    };
    if (auto ran = liteapp::run_desktop_application(server, std::move(desktop)); !ran) {
        std::cerr << "kidi-agent: cannot run the web UI on 127.0.0.1:" << port << ": " << ran.error().message << '\n';
        return 1;
    }
    return 0;
}

} // namespace kidi::agent
