#include "kidi/cli/interactive.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string_view>

#include <spdlog/fmt/fmt.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace kidi::cli {
namespace {
volatile std::sig_atomic_t interrupted = 0;
auto interrupt(int) -> void { interrupted = 1; }

class InterruptScope {
public:
    InterruptScope() {
        interrupted = 0;
#if defined(_WIN32)
        previous_ = std::signal(SIGINT, interrupt);
#else
        struct sigaction handler{};
        handler.sa_handler = interrupt;
        sigemptyset(&handler.sa_mask);
        installed_ = sigaction(SIGINT, &handler, &previous_) == 0;
#endif
    }
    ~InterruptScope() {
#if defined(_WIN32)
        if (previous_ != SIG_ERR) std::signal(SIGINT, previous_);
#else
        if (installed_) sigaction(SIGINT, &previous_, nullptr);
#endif
    }

private:
#if defined(_WIN32)
    using Handler = void (*)(int);
    Handler previous_;
#else
    struct sigaction previous_{};
    bool installed_ = false;
#endif
};

auto terminal_colors(std::string_view mode) -> bool {
    if (mode != "auto") return mode == "always";
    const auto term = std::getenv("TERM");
    if (std::getenv("NO_COLOR") || (term && std::string_view(term) == "dumb")) return false;
#if defined(_WIN32)
    return _isatty(_fileno(stdout));
#else
    return isatty(fileno(stdout));
#endif
}

auto print_text(std::string_view text) -> void {
    for (unsigned char character : text)
        if ((character >= 32 && character != 127) || character == '\n' || character == '\t')
            std::cout.put(static_cast<char>(character));
}

auto memory_summary() -> std::string {
    std::optional<std::uint64_t> process_bytes, total;
    std::string_view usage_label = "RSS";
#if defined(__APPLE__)
    std::optional<int> headroom;
    usage_label = "footprint";
    task_vm_info_data_t task{};
    mach_msg_type_number_t task_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&task), &task_count) == KERN_SUCCESS &&
        task_count >= TASK_VM_INFO_REV1_COUNT)
        process_bytes = task.phys_footprint;
    std::uint64_t physical = 0;
    auto size = sizeof(physical);
    if (sysctlbyname("hw.memsize", &physical, &size, nullptr, 0) == 0) total = physical;
    int level = 0;
    size = sizeof(level);
    if (sysctlbyname("kern.memorystatus_level", &level, &size, nullptr, 0) == 0 && level >= 0 && level <= 100)
        headroom = level;
#elif defined(__linux__)
    std::optional<std::uint64_t> free;
    struct sysinfo system{};
    if (sysinfo(&system) == 0) {
        total = static_cast<std::uint64_t>(system.totalram) * system.mem_unit;
        free = static_cast<std::uint64_t>(system.freeram) * system.mem_unit;
    }
    std::ifstream statm("/proc/self/statm");
    std::uint64_t virtual_pages = 0, resident_pages = 0;
    const auto page_size = sysconf(_SC_PAGESIZE);
    if (statm >> virtual_pages >> resident_pages && page_size > 0) process_bytes = resident_pages * page_size;
#elif defined(_WIN32)
    std::optional<std::uint64_t> free;
    PROCESS_MEMORY_COUNTERS process{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &process, sizeof(process))) process_bytes = process.WorkingSetSize;
    MEMORYSTATUSEX system{};
    system.dwLength = sizeof(system);
    if (GlobalMemoryStatusEx(&system)) {
        total = system.ullTotalPhys;
        free = system.ullAvailPhys;
    }
#else
    std::optional<std::uint64_t> free;
#endif
    constexpr double GIB = 1024. * 1024. * 1024.;
    const auto used = process_bytes ? fmt::format("{:.2f} GiB", *process_bytes / GIB) : std::string("n/a");
#if defined(__APPLE__)
    const auto remaining = headroom ? fmt::format("{}%", *headroom) : std::string("n/a");
    const auto capacity = total ? fmt::format("{:.2f} GiB total", *total / GIB) : std::string("total n/a");
    return fmt::format("{} {} | RAM headroom {} ({})", usage_label, used, remaining, capacity);
#else
    const auto remaining = free && total && *total && *free <= *total
                               ? fmt::format("{:.2f}/{:.2f} GiB ({:.1f}%)", *free / GIB, *total / GIB,
                                             100. * static_cast<double>(*free) / *total)
                               : std::string("n/a");
    return fmt::format("{} {} | RAM free {}", usage_label, used, remaining);
#endif
}
} // namespace

auto interactive_chat(inference::Generator& generator, inference::GenerationOptions options, const Namespace& arguments,
                      std::uint64_t load_ns) -> int {
    auto configured = generator.configure_serving({.maximum_active = 1,
                                                   .maximum_requests = 1,
                                                   .cache_token_budget = arguments.get<std::size_t>("cache_tokens"),
                                                   .prefill_tokens_per_step = options.prefill_chunk_size});
    if (!configured) {
        std::cerr << configured.error().message << '\n';
        return 2;
    }
    options.stream_text = true;
    const InterruptScope signals;
    const bool color = terminal_colors(arguments.get<std::string>("color"));
    const auto label = [&](std::string_view text, std::string_view escape) {
        if (color) std::cout << escape;
        std::cout << text;
        if (color) std::cout << "\033[0m";
        std::cout.flush();
    };
    std::string system = arguments.get<std::string>("system");
    std::vector<text::ChatMessage> history;
    const auto clear = [&] {
        history.clear();
        if (!system.empty()) history.push_back({"system", system});
    };
    clear();
    label("Kidi chat", "\033[1;36m");
    std::cout << "  /help for commands; Ctrl-C cancels a reply; Ctrl-D exits.\n";
    label(fmt::format("[loaded in {:.2f} s | {}]\n", load_ns / 1e9, memory_summary()), "\033[2m");
    while (std::cout) {
        interrupted = 0;
        label("You> ", "\033[1;32m");
        std::string prompt;
        if (!std::getline(std::cin, prompt)) {
            if (interrupted) {
                std::cin.clear();
                std::cout << '\n';
                continue;
            }
            std::cout << '\n';
            return std::cin.eof() ? 0 : 1;
        }
        if (!prompt.empty() && prompt.back() == '\r') prompt.pop_back();
        if (prompt == "/exit" || prompt == "/quit") return 0;
        if (prompt == "/help") {
            std::cout << "/clear           Reset conversation, keeping the system instruction.\n"
                         "/system TEXT     Set the system instruction and reset history (empty clears it).\n"
                         "/multiline       Enter a prompt ending with /send; /cancel discards it.\n"
                         "/exit, /quit     Exit chat.\n"
                         "Use // to start a message with a literal slash.\n";
            continue;
        }
        if (prompt == "/clear" || prompt == "/system" || prompt.starts_with("/system ")) {
            if (prompt != "/clear") system = prompt.size() > 8 ? prompt.substr(8) : "";
            clear();
            std::cout << "Conversation cleared.\n";
            continue;
        }
        if (prompt == "/multiline") {
            prompt.clear();
            bool first = true;
            for (std::string line;;) {
                label("... ", "\033[2m");
                if (!std::getline(std::cin, line)) {
                    if (!interrupted) return std::cin.eof() ? 0 : 1;
                    std::cin.clear();
                    prompt.clear();
                    std::cout << '\n';
                    break;
                }
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line == "/send") break;
                if (line == "/cancel") {
                    prompt.clear();
                    break;
                }
                if (!first) prompt += '\n';
                first = false;
                prompt += line;
            }
        } else if (prompt.starts_with("//")) {
            prompt.erase(0, 1);
        } else if (prompt.starts_with('/')) {
            std::cout << "Unknown command. Type /help.\n";
            continue;
        }
        if (prompt.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        history.push_back({"user", std::move(prompt)});
        auto queued = generator.enqueue_chat(history, options);
        if (!queued) {
            history.pop_back();
            label("Error: ", "\033[1;31m");
            print_text(queued.error().message);
            std::cout << "\nUse /clear to reset history or submit a shorter message.\n";
            continue;
        }
        label("Assistant> ", "\033[1;36m");
        std::optional<inference::TextGeneration> completed;
        while (generator.pending_requests() && !interrupted) {
            auto step = generator.step();
            if (!step) {
                history.pop_back();
                std::cerr << "\nGeneration failed: " << step.error().message << '\n';
                return 1;
            }
            for (auto& event : step->events) {
                print_text(event.text);
                std::cout.flush();
                if (event.completed) completed = std::move(event.completed);
            }
            if (!std::cout) {
                if (generator.pending_requests()) generator.cancel(*queued);
                return 1;
            }
        }
        std::cout << '\n';
        if (interrupted) {
            if (generator.pending_requests()) {
                auto cancelled = generator.cancel(*queued);
                if (!cancelled) {
                    std::cerr << cancelled.error().message << '\n';
                    return 1;
                }
            }
            history.pop_back();
            std::cout << "[Cancelled; turn discarded.]\n";
        } else if (completed) {
            history.push_back({"assistant", std::move(completed->text)});
            const auto& stats = completed->stats;
            const auto decode_speed = stats.decode_tokens && stats.decode_ns
                                          ? fmt::format("{:.1f} tok/s", stats.decode_tokens * 1e9 / stats.decode_ns)
                                          : std::string("n/a");
            label(fmt::format("[{} tokens | decode {} | first token {:.2f} s | total {:.2f} s | {}]\n",
                              completed->generation.token_ids.size(), decode_speed, stats.time_to_first_token_ns / 1e9,
                              stats.generation_ns / 1e9, memory_summary()),
                  "\033[2m");
            if (arguments.get<bool>("profile"))
                std::cerr << "kidi_chat|request_id=" << *queued << "|prompt_tokens=" << completed->stats.prompt_tokens
                          << "|generated_tokens=" << completed->generation.token_ids.size()
                          << "|decode_tokens=" << stats.decode_tokens << "|decode_ns=" << stats.decode_ns
                          << "|ttft_ns=" << completed->stats.time_to_first_token_ns
                          << "|generation_ns=" << completed->stats.generation_ns << '\n';
        }
    }
    return 1;
}
} // namespace kidi::cli