#include "kidi/agent/command.h"
#include "kidi/agent/markdown.h"
#include "kidi/agent/terminal.h"
#include "kidi/agent/transcript.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <termios.h>
#include <thread>
#include <unistd.h>

namespace kidi::agent {
namespace {

auto test_provider(ChatClientOptions options) -> Result<Provider> {
    Provider provider([options](std::span<const Message> messages, std::span<const ToolSpec>,
                         std::stop_token stop) -> Result<Message> {
        if (messages.back().role == Role::TOOL)
            return Message{
                .role = Role::ASSISTANT, .content = "result: " + messages.back().content, .usage = TokenUsage{500, 16}};
        const auto& text = messages.back().content;
        if (text == "stream") {
            if (options.on_delta) {
                options.on_delta("Inspecting.\n", true);
                options.on_delta("first piece", false);
                options.on_delta(" and last piece", false);
            }
            return Message{.role = Role::ASSISTANT, .content = "first piece and last piece",
                           .reasoning_content = "Inspecting.\n"};
        }
        if (text == "wait") {
            std::cerr << "provider waiting\n" << std::flush;
            while (!stop.stop_requested()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        }
        if (text == "large")
            return Message{.role = Role::ASSISTANT, .content = std::string(65536, 'x')};
        if (text == "commands")
            return Message{.role = Role::ASSISTANT,
                           .tool_calls = {{"first", "run_command", {{"command", "printf first"}}},
                                          {"second", "run_command", {{"command", "printf second"}}}}};
        if (text == "presentation")
            return Message{.role = Role::ASSISTANT,
                           .content = "Checking the workspace.",
                           .reasoning_content = "Inspect output first.\nKeep the answer brief.\033]0;untrusted\a",
                           .tool_calls = {{"pretty", "run_command", {{"command", "printf 'hello\\nworld\\n'"}}}},
                           .usage = TokenUsage{100, 32}};
        if (text == "write")
            return Message{.role = Role::ASSISTANT,
                           .tool_calls = {{"write", "write_file", {{"path", "output"}, {"content", "written"}}}}};
        return Message{.role = Role::ASSISTANT, .content = "echo: " + text, .usage = TokenUsage{100, 8}};
    });
    provider.context = [](auto, auto, std::stop_token) -> Result<ContextUsage> {
        return ContextUsage{100, 65536, 4096};
    };
    provider.summarize = [](auto, std::size_t, std::stop_token) -> Result<Message> {
        return Message{.role = Role::ASSISTANT, .content = "A handoff."};
    };
    provider.model = "test-model";
    return provider;
}

class AgentTerminal : public testing::Test {
protected:
    struct TerminalState {
        int result;
        int error;
        tcflag_t flags;
    };

    std::filesystem::path root_;
    int input_ = -1, output_ = -1, slave_ = -1, terminal_state_ = -1;
    pid_t child_ = -1;
    std::string output_text_;

    auto SetUp() -> void override {
        char path[] = "/tmp/kidi-terminal-test-XXXXXX";
        ASSERT_NE(::mkdtemp(path), nullptr);
        root_ = path;
    }

    auto TearDown() -> void override {
        if (child_ > 0) {
            ::kill(child_, SIGKILL);
            ::waitpid(child_, nullptr, 0);
        }
        if (slave_ >= 0) ::close(slave_);
        if (input_ >= 0) ::close(input_);
        if (output_ >= 0 && output_ != input_) ::close(output_);
        if (terminal_state_ >= 0) ::close(terminal_state_);
        std::filesystem::remove_all(root_);
    }

    auto start(bool tty = true, std::vector<std::string> extra = {}, bool no_color = false) -> void {
        int child_input = -1, child_output = -1;
        int state_pipe[2] = {-1, -1};
        if (tty) {
            ASSERT_EQ(::pipe(state_pipe), 0);
            terminal_state_ = state_pipe[0];
            ASSERT_EQ(::fcntl(state_pipe[1], F_SETFD, FD_CLOEXEC), 0);
            input_ = ::posix_openpt(O_RDWR | O_NOCTTY);
            ASSERT_GE(input_, 0);
            ASSERT_EQ(::grantpt(input_), 0);
            ASSERT_EQ(::unlockpt(input_), 0);
            slave_ = ::open(::ptsname(input_), O_RDWR | O_NOCTTY);
            ASSERT_GE(slave_, 0);
            winsize size{.ws_row = 24, .ws_col = 80};
            ASSERT_EQ(::ioctl(slave_, TIOCSWINSZ, &size), 0);
            output_ = input_;
            child_input = child_output = slave_;
        } else {
            int in[2], out[2];
            ASSERT_EQ(::pipe(in), 0);
            ASSERT_EQ(::pipe(out), 0);
            input_ = in[1];
            output_ = out[0];
            child_input = in[0];
            child_output = out[1];
        }
        child_ = ::fork();
        ASSERT_GE(child_, 0);
        if (child_ == 0) {
            ::alarm(15);
            if (tty) {
                ::close(state_pipe[0]);
                ::setsid();
                ::ioctl(slave_, TIOCSCTTY, 0);
            }
            ::dup2(child_input, STDIN_FILENO);
            ::dup2(child_output, STDOUT_FILENO);
            ::dup2(child_output, STDERR_FILENO);
            ::close(input_);
            if (output_ != input_) ::close(output_);
            if (child_input > STDERR_FILENO) ::close(child_input);
            if (child_output != child_input && child_output > STDERR_FILENO) ::close(child_output);
            ::setenv("TERM", "xterm", 1);
            ::setenv("XDG_CACHE_HOME", (root_ / "cache").c_str(), 1);
            ::unsetenv("LINENOISE_ASSUME_TTY");
            if (no_color)
                ::setenv("NO_COLOR", "1", 1);
            else
                ::unsetenv("NO_COLOR");
            cli::ArgumentParser parser("test-code");
            configure_parser(parser);
            std::vector<std::string> args{"--workspace", root_.string()};
            args.insert(args.end(), extra.begin(), extra.end());
            const std::vector<std::string_view> views(args.begin(), args.end());
            const auto status = agent_command(parser.parse_args(views), test_provider);
            std::cout.flush();
            std::cerr.flush();
            if (tty) {
                termios state{};
                const int result = ::tcgetattr(STDIN_FILENO, &state);
                const TerminalState report{result, result == 0 ? 0 : errno, state.c_lflag};
                if (::write(state_pipe[1], &report, sizeof(report)) != sizeof(report)) ::_exit(125);
                ::close(state_pipe[1]);
            }
            ::_exit(status);
        }
        if (tty) ::close(state_pipe[1]);
        if (!tty) {
            ::close(child_input);
            ::close(child_output);
        }
        for (const auto fd : {input_, output_}) {
            const auto flags = ::fcntl(fd, F_GETFL);
            ASSERT_GE(flags, 0);
            ASSERT_EQ(::fcntl(fd, F_SETFL, flags | O_NONBLOCK), 0);
        }
        if (std::ranges::find(extra, "-p") == extra.end()) {
            ASSERT_TRUE(tty ? wait_empty_prompt() : wait_for("> "));
        }
    }

    // macOS poll() does not support tty devices (it reports POLLNVAL), so PTY waits use select().
    static auto ready(int fd, bool write, int timeout_ms) -> bool {
        fd_set set;
        FD_ZERO(&set);
        FD_SET(fd, &set);
        timeval timeout{.tv_sec = timeout_ms / 1000, .tv_usec = (timeout_ms % 1000) * 1000};
        return ::select(fd + 1, write ? nullptr : &set, write ? &set : nullptr, nullptr, &timeout) > 0;
    }

    // Reads whatever output is available; false once the child side is gone.
    auto drain(int timeout_ms) -> bool {
        if (!ready(output_, false, timeout_ms)) return true;
        char buffer[4096];
        const auto count = ::read(output_, buffer, sizeof(buffer));
        if (count > 0) output_text_.append(buffer, static_cast<std::size_t>(count));
        return count > 0 || (count < 0 && (errno == EAGAIN || errno == EINTR));
    }

    auto send(std::string_view text) -> void {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!text.empty()) {
            const auto count = ::write(input_, text.data(), text.size());
            if (count < 0 && (errno == EAGAIN || errno == EINTR)) {
                ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "terminal input did not drain";
                // Keep reading so the child never blocks on a full output buffer while we wait to write.
                drain(0);
                ready(input_, true, 20);
                continue;
            }
            ASSERT_GT(count, 0);
            text.remove_prefix(static_cast<std::size_t>(count));
        }
    }

    auto wait_for(std::string_view text, std::size_t from = 0) -> bool {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        do {
            if (output_text_.find(text, from) != std::string::npos) return true;
            if (!drain(20)) break;
        } while (std::chrono::steady_clock::now() < deadline);
        const bool found = output_text_.find(text, from) != std::string::npos;
        if (!found) std::cerr << "Expected " << text << " in terminal output:\n" << terminal_text(output_text_) << '\n';
        return found;
    }

    auto wait_task(std::string_view text, std::size_t from = 0) -> bool {
        if (!wait_for(text, from)) return false;
        return wait_for("> ", output_text_.find(text, from) + text.size());
    }

    // The input shows a placeholder only while it is empty and ready for a task.
    auto wait_empty_prompt(std::size_t from = 0) -> bool { return wait_for("Ask Kidi; /help lists commands", from); }

    auto wait_approval(std::size_t from = 0) -> bool {
        if (!wait_for("Approval required:", from)) return false;
        return wait_for("Approve?", output_text_.find("Approval required:", from));
    }

    auto finish(int expected_status = 0) -> void {
        int status = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            const auto result = ::waitpid(child_, &status, WNOHANG);
            ASSERT_GE(result, 0);
            if (result == child_) break;
            ASSERT_LT(std::chrono::steady_clock::now(), deadline) << output_text_;
            if (!drain(10)) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        child_ = -1;
        ASSERT_TRUE(WIFEXITED(status)) << status;
        EXPECT_EQ(WEXITSTATUS(status), expected_status);
        while (ready(output_, false, 0)) {
            char buffer[4096];
            const auto count = ::read(output_, buffer, sizeof(buffer));
            if (count <= 0) break;
            output_text_.append(buffer, static_cast<std::size_t>(count));
        }
        if (terminal_state_ >= 0) {
            TerminalState state{};
            ASSERT_EQ(::read(terminal_state_, &state, sizeof(state)), sizeof(state));
            ::close(terminal_state_);
            terminal_state_ = -1;
            ASSERT_EQ(state.result, 0) << std::strerror(state.error);
            EXPECT_NE(state.flags & ICANON, 0u);
            EXPECT_NE(state.flags & ECHO, 0u);
            EXPECT_NE(state.flags & ISIG, 0u);
        }
    }
};

// TODO: restore these timing-dependent terminal tests once they can run deterministically on macOS and Linux CI:
// - Ctrl-C while an approved `run_command` child is blocked (needed a sleep to let the child start first).
// - The live "Processing N prompt tokens <seconds>" timer advancing while the model is busy.
// - Status-bar layout after a terminal resize (SIGWINCH races with the next frame).
TEST_F(AgentTerminal, EditHistoryPaste) {
    start();
    send("helo\033[Dl\r");
    ASSERT_TRUE(wait_task("echo: hello"));
    auto from = output_text_.size();
    send("\033[A\033[F!\r");
    ASSERT_TRUE(wait_task("echo: hello!", from));
    from = output_text_.size();
    send("abc\033[D\033[DX\033[CY\r");
    ASSERT_TRUE(wait_task("echo: aXbYc", from));
    from = output_text_.size();
    send("abc\033[HX\033[FY\r");
    ASSERT_TRUE(wait_task("echo: XabcY", from));
    from = output_text_.size();
    send("caf\xc3\xa9\033[D\033[DX\r");
    ASSERT_TRUE(wait_task("echo: caXf\xc3\xa9", from));
    from = output_text_.size();
    send("\033[200~first\nsecond\033[201~\r");
    ASSERT_TRUE(wait_for("echo: first", from));
    ASSERT_TRUE(wait_for("second", from));
    ASSERT_TRUE(wait_for("> ", output_text_.find("second", from)));
    from = output_text_.size();
    const std::string large_paste(6000, 'x');
    send("\033[200~" + large_paste + "\033[201~\r");
    // The echoed reply wraps across many rows, so its head scrolls off; a full wrapped row must be visible.
    ASSERT_TRUE(wait_task(large_paste.substr(0, 70), from));
    from = output_text_.size();
    send("/he\t\r");
    ASSERT_TRUE(wait_task("/permissions reset  Revoke", from));
    send("\004");
    finish();
    EXPECT_NE(output_text_.find("\033[32m"), std::string::npos);
    bool found_large_paste = false;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_ / "cache")) {
        if (entry.path().extension() != ".jsonl") continue;
        std::ifstream input(entry.path());
        for (std::string line; std::getline(input, line);) {
            const auto record = Json::parse(line, nullptr, false);
            if (record.is_object() && record.value("role", "") == "user" && record.value("content", "") == large_paste)
                found_large_paste = true;
        }
    }
    EXPECT_TRUE(found_large_paste);
}

TEST_F(AgentTerminal, SessionGrants) {
    start();
    // Typed words starting with y/a/n are normal input; only an approval prompt takes single keys.
    send("yarn\r");
    ASSERT_TRUE(wait_task("echo: yarn"));
    send("commands\r");
    ASSERT_TRUE(wait_approval());
    send("a");
    ASSERT_TRUE(wait_task("result: {\"exit_code\":0,\"output\":\"second\"}"));
    auto from = output_text_.size();
    send("/reset\r");
    ASSERT_TRUE(wait_task("Conversation cleared", from));
    send("write\r");
    ASSERT_TRUE(wait_approval(from));
    send("n");
    ASSERT_TRUE(wait_task("approval denied: write_file"));
    from = output_text_.size();
    send("/permissions reset\r");
    ASSERT_TRUE(wait_task("Session-approved tools: none", from));
    send("commands\r");
    ASSERT_TRUE(wait_approval(from));
    from = output_text_.size();
    send("y");
    ASSERT_TRUE(wait_approval(from));
    send("n");
    ASSERT_TRUE(wait_task("approval denied: run_command", from));
    send("/exit\r");
    finish();
    EXPECT_FALSE(std::filesystem::exists(root_ / "output"));
}

TEST_F(AgentTerminal, ApprovalPasteAndInvalidInput) {
    start();
    send("write\r");
    ASSERT_TRUE(wait_approval());
    send("\033[200~n\033[201~\177y");
    ASSERT_TRUE(wait_task("result:"));
    EXPECT_TRUE(std::filesystem::exists(root_ / "output"));
    auto from = output_text_.size();
    send("write\r");
    ASSERT_TRUE(wait_approval(from));
    send("xy\r");
    ASSERT_TRUE(wait_for("Choose y, a or n", from));
    ASSERT_TRUE(wait_approval(output_text_.find("Choose y, a or n", from)));
    send("n");
    ASSERT_TRUE(wait_task("approval denied: write_file", from));
    send("/exit\r");
    finish();
}

TEST_F(AgentTerminal, InterruptInputAndApproval) {
    start();
    auto from = output_text_.size();
    send("discard me\003");
    ASSERT_TRUE(wait_empty_prompt(from));
    from = output_text_.size();
    send("wait\r");
    ASSERT_TRUE(wait_for("provider waiting", from));
    send("\003");
    ASSERT_TRUE(wait_for("cancelled", from));
    ASSERT_TRUE(wait_empty_prompt(output_text_.find("cancelled", from)));
    from = output_text_.size();
    send("write\r");
    ASSERT_TRUE(wait_approval(from));
    send("\003");
    ASSERT_TRUE(wait_task("approval cancelled", from));
    from = output_text_.size();
    send("hello\r");
    ASSERT_TRUE(wait_task("echo: hello", from));
    send("/exit\r");
    finish();
    EXPECT_FALSE(std::filesystem::exists(root_ / "output"));
}

TEST_F(AgentTerminal, StreamsTextAndSavesReplyOnce) {
    start();
    send("stream\r");
    ASSERT_TRUE(wait_task("first piece and last piece"));
    send("/exit\r");
    finish();
    int saved_replies = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_ / "cache")) {
        if (entry.path().extension() != ".jsonl") continue;
        std::ifstream input(entry.path());
        for (std::string line; std::getline(input, line);) {
            const auto record = Json::parse(line, nullptr, false);
            if (record.is_object() && record.value("role", "") == "assistant" &&
                record.value("content", "") == "first piece and last piece") {
                ++saved_replies;
                EXPECT_EQ(record["reasoning_content"], "Inspecting.\n");
            }
        }
    }
    EXPECT_EQ(saved_replies, 1);
}

TEST_F(AgentTerminal, PipedInput) {
    start(false);
    send("hello\nstream\n/exit\n");
    ASSERT_TRUE(wait_task("echo: hello"));
    ASSERT_TRUE(wait_task("first piece and last piece"));
    finish();
    EXPECT_EQ(output_text_.find('\033'), std::string::npos);
}

TEST_F(AgentTerminal, FinishDrainsOutput) {
    start(false, {"-p", "large"});
    finish();
    EXPECT_NE(output_text_.find(std::string(65536, 'x')), std::string::npos);
}

TEST_F(AgentTerminal, PipedApprovalUsesLineInput) {
    start(false);
    send("write\n");
    ASSERT_TRUE(wait_approval());
    send("yes\n");
    ASSERT_TRUE(wait_task("result:"));
    EXPECT_TRUE(std::filesystem::exists(root_ / "output"));
    send("/exit\n");
    finish();
}

TEST_F(AgentTerminal, NoColor) {
    start(true, {}, true);
    send("helo\033[Dl\r");
    ASSERT_TRUE(wait_task("echo: hello"));
    send("/exit\r");
    finish();
    EXPECT_EQ(output_text_.find("\033[1;32m"), std::string::npos);
}

TEST_F(AgentTerminal, ApprovalEof) {
    start();
    send("write\r");
    ASSERT_TRUE(wait_approval());
    send("\004");
    ASSERT_TRUE(wait_for("end of input"));
    finish();
    EXPECT_FALSE(std::filesystem::exists(root_ / "output"));
}

TEST_F(AgentTerminal, OneShotNoApproval) {
    start(false, {"-p", "commands"});
    ASSERT_TRUE(wait_for("approval required; no approval callback"));
    finish();
    EXPECT_EQ(output_text_.find("Approve?"), std::string::npos);
}

TEST_F(AgentTerminal, PresentationAndWorkspaceTranscript) {
    start(true, {"--permissions", "full-access"});
    send("presentation\r");
    ASSERT_TRUE(wait_task("result: "));
    ASSERT_NE(output_text_.find("Thinking"), std::string::npos);
    EXPECT_NE(output_text_.find("$ printf 'hello\\nworld\\n'"), std::string::npos);
    EXPECT_EQ(output_text_.find("{\"command\":"), std::string::npos);
    EXPECT_NE(output_text_.find("\\x1b]0;untrusted\\x07"), std::string::npos);
    EXPECT_EQ(output_text_.find("\033]0;untrusted"), std::string::npos);
    auto from = output_text_.size();
    send("/reset\r");
    ASSERT_TRUE(wait_task("Conversation cleared", from));
    send("/exit\r");
    finish();

    std::vector<std::filesystem::path> sessions;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_ / "cache"))
        if (entry.path().extension() == ".jsonl") sessions.push_back(entry.path());
    ASSERT_EQ(sessions.size(), 1);
    EXPECT_EQ(std::filesystem::status(sessions[0]).permissions() & std::filesystem::perms::all,
              std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
    std::ifstream input(sessions[0]);
    std::vector<Json> records;
    for (std::string line; std::getline(input, line);) {
        auto record = Json::parse(line, nullptr, false);
        ASSERT_TRUE(record.is_object()) << line;
        records.push_back(std::move(record));
    }
    ASSERT_GE(records.size(), 7);
    EXPECT_EQ(records.front()["workspace"], std::filesystem::canonical(root_).generic_string());
    EXPECT_EQ(records.front()["model"], "test-model");
    EXPECT_EQ(records.back()["type"], "reset");
    const auto assistant = std::ranges::find_if(records, [](const auto& record) {
        return record.value("role", "") == "assistant" && record.contains("tool_calls");
    });
    ASSERT_NE(assistant, records.end());
    EXPECT_EQ((*assistant)["tool_calls"][0]["arguments"]["command"], "printf 'hello\\nworld\\n'");
    EXPECT_EQ((*assistant)["reasoning_content"], "Inspect output first.\nKeep the answer brief.\033]0;untrusted\a");
    const auto tool =
        std::ranges::find_if(records, [](const auto& record) { return record.value("role", "") == "tool"; });
    ASSERT_NE(tool, records.end());
    EXPECT_EQ((*tool)["tool_call_id"], "pretty");
    EXPECT_NE((*tool)["content"].template get<std::string>().find("hello"), std::string::npos);
}

TEST_F(AgentTerminal, NoSave) {
    start(true, {"--no-save"});
    send("hello\r");
    ASSERT_TRUE(wait_task("echo: hello"));
    send("/exit\r");
    finish();
    EXPECT_FALSE(std::filesystem::exists(root_ / "cache"));
    EXPECT_NE(output_text_.find("\033[?2004l"), std::string::npos);
}

TEST_F(AgentTerminal, TranscriptAppendAndWorkspaceIsolation) {
    // Set the cache only in the child, as in the CLI tests, to avoid changing the test runner environment.
    start(false, {"-p", "hello"});
    ASSERT_TRUE(wait_for("echo: hello"));
    finish();
    std::filesystem::path first;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_ / "cache"))
        if (entry.path().extension() == ".jsonl") first = entry.path();
    ASSERT_FALSE(first.empty());
    const auto original_size = std::filesystem::file_size(first);
    ::close(input_);
    ::close(output_);
    input_ = output_ = -1;
    start(false, {"-p", "again"});
    ASSERT_TRUE(wait_for("echo: again"));
    finish();
    std::size_t files = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_ / "cache"))
        if (entry.path().extension() == ".jsonl") {
            ++files;
            EXPECT_EQ(entry.path().parent_path(), first.parent_path());
        }
    EXPECT_EQ(files, 2);
    EXPECT_EQ(std::filesystem::file_size(first), original_size);
    ::close(input_);
    ::close(output_);
    input_ = output_ = -1;
    std::filesystem::create_directory(root_ / "other");
    start(false, {"-p", "elsewhere", "--workspace", (root_ / "other").string()});
    ASSERT_TRUE(wait_for("echo: elsewhere"));
    finish();
    bool other_workspace = false;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_ / "cache"))
        if (entry.path().extension() == ".jsonl" && entry.path().parent_path() != first.parent_path())
            other_workspace = true;
    EXPECT_TRUE(other_workspace);
}

TEST_F(AgentTerminal, TranscriptFailureIsExplicit) {
    std::ofstream(root_ / "cache") << "not a directory";
    start(false, {"-p", "hello"});
    ASSERT_TRUE(wait_for("transcript: create cache directory"));
    ASSERT_TRUE(wait_for("--no-save"));
    finish(1);
    EXPECT_EQ(output_text_.find("echo: hello"), std::string::npos);
}

TEST_F(AgentTerminal, MultilineInputStaysInScrollback) {
    start();
    // Ctrl-J adds a line; Enter submits both lines as one task.
    send("line one\nline two\r");
    ASSERT_TRUE(wait_task("line two"));
    send("/exit\r");
    finish();
    // The transcript is printed inline, never on the alternate screen, so native scrollback and copying work.
    EXPECT_EQ(output_text_.find("\033[?1049h"), std::string::npos);
    bool found = false;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_ / "cache")) {
        if (entry.path().extension() != ".jsonl") continue;
        std::ifstream input(entry.path());
        for (std::string line; std::getline(input, line);) {
            const auto record = Json::parse(line, nullptr, false);
            found = found || (record.is_object() && record.value("content", "") == "line one\nline two");
        }
    }
    EXPECT_TRUE(found);
}

TEST(TerminalMarkdown, StylesAndPassthrough) {
    const auto lines = render_markdown("# Title\nUse **bold** and `code`.\n- [x] done\n```\n**raw**\n```\n", true, 80);
    ASSERT_EQ(lines.size(), 6);
    EXPECT_EQ(lines[0], "\033[1;4mTitle\033[0m");
    EXPECT_EQ(lines[1], "Use \033[1mbold\033[0m and \033[38;5;180mcode\033[0m.\033[0m");
    EXPECT_EQ(lines[2], "☑ done");
    EXPECT_NE(lines[4].find("**raw**"), std::string::npos) << "code blocks keep their text";
    EXPECT_EQ(render_markdown("snake_case_name and 2*3", true, 80)[0], "snake_case_name and 2*3");
    // Without color the source is unchanged, so copied text is exactly what the model wrote.
    EXPECT_EQ(render_markdown("# Title\n| a | b |\n", false, 80), (std::vector<std::string>{"# Title", "| a | b |"}));
}

TEST(TerminalMarkdown, StreamsLinesAndAlignsTables) {
    MarkdownStream stream(true);
    EXPECT_TRUE(stream.feed("| Name | Count |\n|:--|--:|\n| a | 12", 80).empty());
    EXPECT_EQ(stream.preview(80).size(), 5) << "a held table previews as a box";
    const auto table = stream.feed("3 |\nafter\n", 80);
    ASSERT_EQ(table.size(), 6);
    EXPECT_EQ(display_width(table[0]), display_width(table[3]));
    EXPECT_NE(table[3].find(" a    "), std::string::npos) << "left aligned";
    EXPECT_NE(table[3].find("   123 "), std::string::npos) << "right aligned";
    EXPECT_EQ(table[5], "after");
    // A table wider than the terminal wraps its cells at spaces; only a hopelessly narrow one falls back to source.
    const auto wide = render_markdown("| Key | Details |\n|---|---|\n| Name | a long description that must wrap |\n"
                                      "| API | short |\n",
                                      true, 30);
    for (const auto& line : wide) EXPECT_LE(display_width(line), 29u) << line;
    EXPECT_GT(wide.size(), 7u);
    EXPECT_NE(wide[3].find("Name"), std::string::npos);
    EXPECT_NE(wide[3].find("a long"), std::string::npos) << "breaks between words";
    EXPECT_EQ(render_markdown("| a | b |\n|---|---|\n", true, 8).size(), 2);
    // A pipe inside a code span stays in its cell, and stray pipes never add columns beyond the header's.
    const auto piped = render_markdown("| Key | Value |\n|---|---|\n| API | `\"deg\" | \"rad\"` |\n| X | a | b |\n",
                                       true, 80);
    ASSERT_EQ(piped.size(), 6);
    EXPECT_EQ(display_width(piped[3]), display_width(piped[0]));
    EXPECT_NE(piped[3].find("\"deg\" | \"rad\""), std::string::npos) << piped[3];
    EXPECT_EQ(display_width(piped[4]), display_width(piped[0]));
    EXPECT_NE(piped[4].find("a | b"), std::string::npos) << piped[4];
}

TEST(TerminalMarkdown, Widths) {
    EXPECT_EQ(display_width("\033[1mab\033[0m\xe4\xb8\xad"), 4);
    EXPECT_EQ(fit_width("abcdef", 4), "abc…");
    EXPECT_EQ(wrap_width("\033[1mabcd\033[0m", 2), (std::vector<std::string>{"\033[1mab\033[0m", "\033[1mcd\033[0m"}));
}

TEST(CodeTerminalText, Utf8AndControls) {
    EXPECT_EQ(terminal_text("caf\xc3\xa9 \xe4\xb8\xad \xf0\x9f\x98\x80\n\t"),
              "caf\xc3\xa9 \xe4\xb8\xad \xf0\x9f\x98\x80\n\t");
    EXPECT_EQ(terminal_text("\033]0;title\a\r"), "\\x1b]0;title\\x07\\x0d");
    EXPECT_EQ(terminal_text("\xc0\xaf\xed\xa0\x80\xff"), "\\xc0\\xaf\\xed\\xa0\\x80\\xff");
    EXPECT_EQ(terminal_text("\xc2\x9b\xe2\x80\xae"), "\\xc2\\x9b\\xe2\\x80\\xae");
}

} // namespace
} // namespace kidi::agent
