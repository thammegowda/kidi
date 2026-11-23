#include "kidi/code/command.h"
#include "kidi/code/terminal.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <termios.h>
#include <thread>
#include <unistd.h>

namespace kidi::code {
namespace {

auto test_provider(OpenAIOptions) -> Result<Provider> {
    Provider provider(
        [](std::span<const Message> messages, std::span<const ToolSpec>, std::stop_token stop) -> Result<Message> {
            if (messages.back().role == Role::TOOL)
                return Message{.role = Role::ASSISTANT, .content = "result: " + messages.back().content};
            const auto& text = messages.back().content;
            if (text == "wait") {
                std::cerr << "provider waiting\n" << std::flush;
                while (!stop.stop_requested()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
                return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
            }
            if (text == "commands")
                return Message{.role = Role::ASSISTANT,
                               .tool_calls = {{"first", "run_command", {{"command", "printf first"}}},
                                              {"second", "run_command", {{"command", "printf second"}}}}};
            if (text == "sleep")
                return Message{.role = Role::ASSISTANT,
                               .tool_calls = {{"sleep", "run_command", {{"command", "sleep 30"}}}}};
            if (text == "write")
                return Message{.role = Role::ASSISTANT,
                               .tool_calls = {{"write", "write_file", {{"path", "output"}, {"content", "written"}}}}};
            return Message{.role = Role::ASSISTANT, .content = "echo: " + text};
        });
    provider.context = [](auto, auto, std::stop_token) -> Result<ContextUsage> {
        return ContextUsage{100, 65536, 4096};
    };
    provider.summarize = [](auto, std::size_t, std::stop_token) -> Result<Message> {
        return Message{.role = Role::ASSISTANT, .content = "A handoff."};
    };
    return provider;
}

class CodeTerminal : public testing::Test {
protected:
    std::filesystem::path root_;
    int input_ = -1, output_ = -1, slave_ = -1;
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
        std::filesystem::remove_all(root_);
    }

    auto start(bool tty = true, std::vector<std::string> extra = {}, bool no_color = false) -> void {
        int child_input = -1, child_output = -1;
        if (tty) {
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
            const auto status = code_command(parser.parse_args(views), test_provider);
            std::cout.flush();
            std::cerr.flush();
            ::_exit(status);
        }
        if (!tty) {
            ::close(child_input);
            ::close(child_output);
        }
        if (std::ranges::find(extra, "-p") == extra.end()) {
            ASSERT_TRUE(wait_for("> "));
        }
    }

    auto send(std::string_view text) -> void {
        while (!text.empty()) {
            const auto count = ::write(input_, text.data(), text.size());
            ASSERT_GT(count, 0);
            text.remove_prefix(static_cast<std::size_t>(count));
        }
    }

    auto wait_for(std::string_view text, std::size_t from = 0) -> bool {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        do {
            if (output_text_.find(text, from) != std::string::npos) return true;
            pollfd ready{.fd = output_, .events = POLLIN};
            if (::poll(&ready, 1, 20) > 0) {
                char buffer[4096];
                const auto count = ::read(output_, buffer, sizeof(buffer));
                if (count > 0)
                    output_text_.append(buffer, static_cast<std::size_t>(count));
                else
                    break;
            }
        } while (std::chrono::steady_clock::now() < deadline);
        const bool found = output_text_.find(text, from) != std::string::npos;
        if (!found) std::cerr << "Expected " << text << " in terminal output:\n" << terminal_text(output_text_) << '\n';
        return found;
    }

    auto wait_task(std::string_view text, std::size_t from = 0) -> bool {
        if (!wait_for(text, from)) return false;
        return wait_for("> ", output_text_.find(text, from) + text.size());
    }

    auto wait_approval(std::size_t from = 0) -> bool {
        if (!wait_for("Approval required:", from)) return false;
        return wait_for("Approve?", output_text_.find("Approval required:", from));
    }

    auto finish() -> void {
        int status = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            const auto result = ::waitpid(child_, &status, WNOHANG);
            ASSERT_GE(result, 0);
            if (result == child_) break;
            ASSERT_LT(std::chrono::steady_clock::now(), deadline) << output_text_;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        child_ = -1;
        ASSERT_TRUE(WIFEXITED(status)) << status;
        EXPECT_EQ(WEXITSTATUS(status), 0);
        if (slave_ >= 0) {
            termios state{};
            ASSERT_EQ(::tcgetattr(slave_, &state), 0);
            EXPECT_NE(state.c_lflag & ICANON, 0u);
            EXPECT_NE(state.c_lflag & ECHO, 0u);
            EXPECT_NE(state.c_lflag & ISIG, 0u);
        }
    }
};

TEST_F(CodeTerminal, EditHistoryPaste) {
    start();
    send("helo\033[Dl\r");
    ASSERT_TRUE(wait_task("echo: hello"));
    auto from = output_text_.size();
    send("\033[A\r");
    ASSERT_TRUE(wait_task("echo: hello", from));
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
    ASSERT_TRUE(wait_task("echo: first\r\nsecond", from));
    from = output_text_.size();
    const std::string large_paste(6000, 'x');
    send("\033[200~" + large_paste + "\033[201~\r");
    ASSERT_TRUE(wait_task("echo: " + large_paste, from));
    from = output_text_.size();
    send("/he\t\r");
    ASSERT_TRUE(wait_task("/permissions reset  Revoke", from));
    send("\004");
    finish();
    EXPECT_NE(output_text_.find("\033[1;32m"), std::string::npos);
}

TEST_F(CodeTerminal, SessionGrants) {
    start();
    send("commands\r");
    ASSERT_TRUE(wait_approval());
    send("a\r");
    ASSERT_TRUE(wait_task("result: {\"exit_code\":0,\"output\":\"second\"}"));
    auto from = output_text_.size();
    send("\033[A\r");
    ASSERT_TRUE(wait_task("result: {\"exit_code\":0,\"output\":\"second\"}", from));
    EXPECT_EQ(output_text_.find("Approval required:", from), std::string::npos);
    from = output_text_.size();
    send("/reset\r");
    ASSERT_TRUE(wait_task("Conversation cleared", from));
    send("write\r");
    ASSERT_TRUE(wait_approval(from));
    send("n\r");
    ASSERT_TRUE(wait_task("approval denied: write_file"));
    from = output_text_.size();
    send("/permissions reset\r");
    ASSERT_TRUE(wait_task("Session-approved tools: none", from));
    send("commands\r");
    ASSERT_TRUE(wait_approval(from));
    from = output_text_.size();
    send("y\r");
    ASSERT_TRUE(wait_approval(from));
    send("n\r");
    ASSERT_TRUE(wait_task("approval denied: run_command", from));
    send("/exit\r");
    finish();
    EXPECT_FALSE(std::filesystem::exists(root_ / "output"));
}

TEST_F(CodeTerminal, InterruptInputAndApproval) {
    start();
    auto from = output_text_.size();
    send("discard me\003");
    ASSERT_TRUE(wait_task("\033[?2004l", from));
    from = output_text_.size();
    send("wait\r");
    ASSERT_TRUE(wait_for("provider waiting", from));
    ASSERT_EQ(::kill(child_, SIGINT), 0);
    ASSERT_TRUE(wait_task("kidi-code: ", from));
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

TEST_F(CodeTerminal, InterruptCommand) {
    start(true, {"--permissions", "full-access"});
    send("sleep\r");
    ASSERT_TRUE(wait_for("\"command\":\"sleep 30\""));
    // Allow the approved child to enter its blocking command before sending SIGINT.
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    ASSERT_EQ(::kill(child_, SIGINT), 0);
    ASSERT_TRUE(wait_task("cancelled"));
    send("hello\r");
    ASSERT_TRUE(wait_task("echo: hello"));
    send("/exit\r");
    finish();
}

TEST_F(CodeTerminal, PipedInput) {
    start(false);
    send("hello\n/exit\n");
    ASSERT_TRUE(wait_task("echo: hello"));
    finish();
    EXPECT_EQ(output_text_.find('\033'), std::string::npos);
}

TEST_F(CodeTerminal, NoColor) {
    start(true, {}, true);
    send("helo\033[Dl\r");
    ASSERT_TRUE(wait_task("echo: hello"));
    send("/exit\r");
    finish();
    EXPECT_EQ(output_text_.find("\033[1;32m"), std::string::npos);
}

TEST_F(CodeTerminal, ApprovalEof) {
    start();
    send("write\r");
    ASSERT_TRUE(wait_approval());
    send("\004");
    ASSERT_TRUE(wait_for("end of input"));
    finish();
    EXPECT_FALSE(std::filesystem::exists(root_ / "output"));
}

TEST_F(CodeTerminal, OneShotNoApproval) {
    start(false, {"-p", "commands"});
    ASSERT_TRUE(wait_for("approval required; no approval callback"));
    finish();
    EXPECT_EQ(output_text_.find("Approve?"), std::string::npos);
}

TEST(CodeTerminalText, Utf8AndControls) {
    EXPECT_EQ(terminal_text("caf\xc3\xa9 \xe4\xb8\xad \xf0\x9f\x98\x80\n\t"),
              "caf\xc3\xa9 \xe4\xb8\xad \xf0\x9f\x98\x80\n\t");
    EXPECT_EQ(terminal_text("\033]0;title\a\r"), "\\x1b]0;title\\x07\\x0d");
    EXPECT_EQ(terminal_text("\xc0\xaf\xed\xa0\x80\xff"), "\\xc0\\xaf\\xed\\xa0\\x80\\xff");
    EXPECT_EQ(terminal_text("\xc2\x9b\xe2\x80\xae"), "\\xc2\\x9b\\xe2\\x80\\xae");
}

} // namespace
} // namespace kidi::code
