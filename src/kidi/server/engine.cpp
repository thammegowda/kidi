#include "kidi/server/engine.h"

#include <chrono>
#include <exception>
#include <unordered_map>

namespace kidi::server {

struct ChatEngine::Job {
    ChatRun request;
    std::mutex mutex;
    std::condition_variable changed;
    std::string text;
    std::optional<Result<inference::TextGeneration>> outcome;
    bool cancelled = false;

    auto settle(Result<inference::TextGeneration> result) -> void {
        {
            const std::lock_guard lock(mutex);
            outcome = std::move(result);
        }
        changed.notify_all();
    }
};

namespace {

template <class Work>
auto guarded(Work&& work) -> decltype(work()) {
    try {
        return work();
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

} // namespace

ChatEngine::ChatEngine(inference::Generator& generator) : generator_(generator), worker_([this] { work(); }) {}

ChatEngine::~ChatEngine() {
    {
        const std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    worker_.join();
}

auto ChatEngine::run(ChatRun request) -> Result<inference::TextGeneration> {
    auto job = std::make_shared<Job>();
    job->request = std::move(request);
    {
        const std::lock_guard lock(mutex_);
        if (stopping_) return std::unexpected(Error{ErrorCode::RUNTIME, "server stopped"});
        inbox_.push_back(job);
    }
    wake_.notify_all();

    const auto& cancelled = job->request.cancelled;
    const auto& on_text = job->request.on_text;
    std::size_t sent = 0;
    while (true) {
        std::string fresh;
        bool done = false;
        {
            std::unique_lock lock(job->mutex);
            // Wake periodically so caller-side cancellation (client disconnect, UI stop) is noticed during prefill.
            job->changed.wait_for(lock, std::chrono::milliseconds(20),
                                  [&] { return job->text.size() != sent || job->outcome; });
            fresh = job->text.substr(sent);
            sent = job->text.size();
            done = job->outcome.has_value();
        }
        bool stop = cancelled && cancelled();
        if (!stop && !fresh.empty() && on_text) stop = !on_text(fresh);
        if (done) return std::move(*job->outcome);
        if (stop) {
            {
                const std::lock_guard lock(job->mutex);
                job->cancelled = true;
            }
            wake_.notify_all();
            return std::unexpected(Error{ErrorCode::RUNTIME, "generation cancelled"});
        }
    }
}

auto ChatEngine::context(const nlohmann::ordered_json& messages, const nlohmann::ordered_json& template_args,
                         inference::GenerationOptions options) -> Result<ChatContext> {
    const std::lock_guard lock(generator_mutex_);
    return guarded([&]() -> Result<ChatContext> {
        auto capacity = generator_.context_capacity(options);
        if (!capacity) return std::unexpected(capacity.error());
        auto tokens = generator_.chat_prompt_tokens(messages, template_args);
        if (!tokens) return std::unexpected(tokens.error());
        const auto limit =
            options.maximum_prompt_tokens ? std::min(*capacity - 1, options.maximum_prompt_tokens) : *capacity - 1;
        return ChatContext{*tokens, *capacity, limit};
    });
}

auto ChatEngine::work() -> void {
    std::unordered_map<std::uint64_t, std::shared_ptr<Job>> running;
    bool release_when_idle = false;
    while (true) {
        std::deque<std::shared_ptr<Job>> arrived;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stopping_ || !inbox_.empty() || !running.empty(); });
            if (stopping_) break;
            arrived.swap(inbox_);
        }
        // Tokenization uses the same tokenizer/template state as the generation worker.
        const std::lock_guard generator_lock(generator_mutex_);
        for (const auto& job : arrived) {
            auto id = guarded([&] { return job->request.enqueue(generator_); });
            if (!id) {
                job->settle(std::unexpected(id.error()));
                continue;
            }
            running.emplace(*id, job);
            release_when_idle |= job->request.release_workspaces;
        }
        std::erase_if(running, [&](const auto& entry) {
            const std::lock_guard lock(entry.second->mutex);
            if (entry.second->cancelled) (void)generator_.cancel(entry.first);
            return entry.second->cancelled;
        });
        if (running.empty()) {
            if (release_when_idle && !generator_.pending_requests()) {
                (void)generator_.release_workspaces();
                release_when_idle = false;
            }
            continue;
        }
        auto step = guarded([&] { return generator_.step(); });
        if (!step) {
            for (const auto& [id, job] : running) {
                (void)generator_.cancel(id);
                job->settle(std::unexpected(step.error()));
            }
            running.clear();
            continue;
        }
        for (auto& event : step->events) {
            const auto found = running.find(event.request_id);
            if (found == running.end()) continue;
            auto& job = *found->second;
            {
                const std::lock_guard lock(job.mutex);
                job.text += event.text;
            }
            if (event.completed) {
                job.settle(std::move(*event.completed));
                running.erase(found);
            } else {
                job.changed.notify_all();
            }
        }
    }
    for (const auto& [id, job] : running) job->settle(std::unexpected(Error{ErrorCode::RUNTIME, "server stopped"}));
    const std::lock_guard lock(mutex_);
    for (const auto& job : inbox_) job->settle(std::unexpected(Error{ErrorCode::RUNTIME, "server stopped"}));
}

} // namespace kidi::server
