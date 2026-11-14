#include "kidi/checkpoint/hub.h"
#include "kidi/checkpoint/config.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <crt_externs.h>
#else
extern "C" char** environ;
#endif
#endif

namespace kidi::checkpoint {
namespace {

auto expand_home(std::filesystem::path path) -> std::filesystem::path {
    const auto value = path.string();
    if (value.empty() || value.front() != '~') return path;
    const auto* home = std::getenv(
#if defined(_WIN32)
        "USERPROFILE"
#else
        "HOME"
#endif
    );
    if (!home || !*home) throw std::invalid_argument("cannot expand Hub cache without a home directory");
    if (value == "~") return home;
    if (value.size() < 2 || (value[1] != '/' && value[1] != '\\'))
        throw std::invalid_argument("Hub cache uses an invalid home-relative path");
    return std::filesystem::path(home) / value.substr(2);
}

auto safe_segment(std::string_view part) -> bool {
    return !part.empty() && part != "." && part != ".." && std::ranges::all_of(part, [](unsigned char byte) {
        return std::isalnum(byte) || byte == '_' || byte == '.' || byte == '-';
    });
}

auto safe_revision(std::string_view value) -> bool {
    if (value.empty() || value.front() == '/' || value.back() == '/') return false;
    for (std::size_t start = 0; start <= value.size();) {
        const auto end = value.find('/', start);
        if (!safe_segment(value.substr(start, end - start))) return false;
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return true;
}

auto non_empty_file(const std::filesystem::path& path) -> bool {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error && std::filesystem::file_size(path, error) > 0 &&
           !error;
}

auto validate_downloaded_package(const std::filesystem::path& directory, std::string_view reference) -> Result<void> {
    if (!non_empty_file(directory / "config.json"))
        return std::unexpected(
            Error{ErrorCode::IO, "Missing config.json in @" + std::string(reference) + "; retry online"});
    YAML::Node raw;
    try {
        raw = YAML::LoadFile((directory / "config.json").string());
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid config.json: " + std::string(error.what())});
    }
    auto expected = hub_files(raw);
    if (!expected) return std::unexpected(expected.error());
    auto config = load_config(directory);
    if (!config) return std::unexpected(config.error());
    const bool gemma = (*config)["model"]["type"].as<std::string>("") == "gemma4_text";
    for (const auto& name : *expected) {
        if (gemma && std::filesystem::path(name).filename() == "chat_template.jinja") continue;
        if (!non_empty_file(directory / name))
            return std::unexpected(
                Error{ErrorCode::IO, "Missing " + name + " in @" + std::string(reference) +
                                         "; retry online to complete the download. Sharded and non-Safetensors "
                                         "checkpoints are not supported."});
    }
    if (gemma) {
        const auto tokenizer_dir = std::filesystem::path((*config)["tokenizer_file"].as<std::string>()).parent_path();
        const auto tok_cfg_path = tokenizer_dir / "tokenizer_config.json";
        if (!non_empty_file(tok_cfg_path))
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "Missing tokenizer_config.json"});
        try {
            const auto tok_cfg = YAML::LoadFile(tok_cfg_path.string());
            const auto inline_template =
                tok_cfg.IsMap() && tok_cfg["chat_template"] && tok_cfg["chat_template"].IsScalar()
                    ? tok_cfg["chat_template"].as<std::string>("")
                    : std::string{};
            if (inline_template.empty() && !non_empty_file(tokenizer_dir / "chat_template.jinja"))
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST,
                                             "Missing chat_template.jinja or chat_template in tokenizer_config.json"});
        } catch (const YAML::Exception& error) {
            return std::unexpected(
                Error{ErrorCode::INVALID_MANIFEST, "invalid tokenizer_config.json: " + std::string(error.what())});
        }
    }
    return {};
}

auto run_hf(std::vector<std::string> arguments, std::string_view reference) -> Result<void> {
    const auto missing_hf = [] {
        return std::unexpected(
            Error{ErrorCode::IO,
                  "'hf' was not found in PATH; activate an environment with huggingface-hub installed or download the "
                  "model and pass its local directory path"});
    };
#if defined(__EMSCRIPTEN__)
    (void)arguments;
    (void)reference;
    return std::unexpected(Error{ErrorCode::UNSUPPORTED, "Hub downloads are not available in WebAssembly"});
#elif defined(_WIN32)
    const auto quote = [](std::wstring_view arg) {
        std::wstring out = L"\"";
        std::size_t backslashes = 0;
        for (const wchar_t ch : arg) {
            if (ch == L'\\') {
                ++backslashes;
            } else if (ch == L'"') {
                out.append(backslashes * 2 + 1, L'\\');
                out.push_back(L'"');
                backslashes = 0;
            } else {
                out.append(backslashes, L'\\');
                out.push_back(ch);
                backslashes = 0;
            }
        }
        out.append(backslashes * 2, L'\\');
        out.push_back(L'"');
        return out;
    };
    std::wstring cmdline;
    for (const auto& arg : arguments) {
        if (!cmdline.empty()) cmdline.push_back(L' ');
        cmdline += quote(std::filesystem::path(arg).wstring());
    }
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_ERROR_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(nullptr, cmdline.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process)) {
        if (GetLastError() == ERROR_FILE_NOT_FOUND) return missing_hf();
        return std::unexpected(Error{ErrorCode::IO, "cannot launch 'hf' command"});
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    std::cerr << std::flush;
    if (code != 0)
        return std::unexpected(
            Error{ErrorCode::RUNTIME,
                  "hf download failed for @" + std::string(reference) +
                      "; check the output above or download the model and pass its local directory path"});
    return {};
#else
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (auto& argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    if (const int init = posix_spawn_file_actions_init(&actions); init != 0)
        return std::unexpected(
            Error{ErrorCode::IO, "cannot initialize hf process actions: " + std::system_category().message(init)});
    struct Cleanup {
        posix_spawn_file_actions_t& actions;
        ~Cleanup() { (void)posix_spawn_file_actions_destroy(&actions); }
    } cleanup{actions};
    if (const int dup_status = posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
        dup_status != 0)
        return std::unexpected(
            Error{ErrorCode::IO, "cannot redirect hf output: " + std::system_category().message(dup_status)});
#if defined(__APPLE__)
    char** envp = *_NSGetEnviron();
#else
    char** envp = environ;
#endif
    pid_t pid = 0;
    const int status = posix_spawnp(&pid, argv.front(), &actions, nullptr, argv.data(), envp);
    if (status == ENOENT) return missing_hf();
    if (status != 0)
        return std::unexpected(Error{ErrorCode::IO, "cannot launch 'hf': " + std::system_category().message(status)});
    int result = 0;
    while (waitpid(pid, &result, 0) < 0) {
        if (errno != EINTR)
            return std::unexpected(
                Error{ErrorCode::IO, "cannot wait for 'hf': " + std::system_category().message(errno)});
    }
    std::cerr << std::flush;
    if (WIFEXITED(result) && WEXITSTATUS(result) == 127) return missing_hf();
    if (!WIFEXITED(result) || WEXITSTATUS(result) != 0)
        return std::unexpected(
            Error{ErrorCode::RUNTIME,
                  "hf download failed for @" + std::string(reference) +
                      "; check the output above or download the model and pass its local directory path"});
    return {};
#endif
}

} // namespace

Hub::Hub(std::string remote, std::filesystem::path local)
    : remote_(std::move(remote)), local_(expand_home(std::move(local)).lexically_normal()) {
    while (remote_.ends_with('/')) remote_.pop_back();
    if (!remote_.starts_with("https://") || remote_.size() <= std::string_view{"https://"}.size())
        throw std::invalid_argument("Hub remote must be an HTTPS origin");
    if (local_.empty()) throw std::invalid_argument("Hub local cache must not be empty");
}

auto Hub::repository(std::string_view model_id) const -> Result<std::filesystem::path> {
    const auto slash = model_id.find('/');
    if (slash == std::string_view::npos || !slash || slash + 1 == model_id.size() ||
        model_id.find('/', slash + 1) != std::string_view::npos)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Hub model ID"});
    const auto owner = model_id.substr(0, slash);
    const auto name = model_id.substr(slash + 1);
    if (!safe_segment(owner) || !safe_segment(name))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Hub model ID"});
    return local_ / std::filesystem::path(owner) / std::filesystem::path(name);
}

auto Hub::resolve(std::string_view model_id, std::string_view revision) const -> Result<std::filesystem::path> {
    if (!safe_revision(revision))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Hub model revision"});
    auto directory = repository(model_id);
    if (!directory) return std::unexpected(directory.error());
    std::error_code error;
    if (!std::filesystem::is_directory(*directory, error))
        return std::unexpected(Error{ErrorCode::IO, "Hub model is not cached: " + std::string(model_id)});
    return *directory;
}

auto Hub::get_file(std::string_view model_id, std::string_view file, std::string_view revision) const
    -> Result<std::filesystem::path> {
    const auto relative = std::filesystem::path(file);
    if (relative.empty() || relative.is_absolute() ||
        std::ranges::any_of(relative, [](const auto& part) { return part == "." || part == ".."; }))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Hub model file path"});
    auto directory = resolve(model_id, revision);
    if (!directory) return std::unexpected(directory.error());
    const auto path = *directory / relative;
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error))
        return std::unexpected(Error{ErrorCode::IO, "Hub model file is not cached: " + std::string(file)});
    return path;
}

auto Hub::download(std::string_view reference) const -> Result<std::filesystem::path> {
    const auto at = reference.find('@');
    const auto model_id = reference.substr(0, at);
    const auto revision = at == std::string_view::npos ? std::string_view{} : reference.substr(at + 1);
    if (at != std::string_view::npos && revision.empty())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "A model revision after @ must not be empty"});
    if (!revision.empty() && !safe_revision(revision))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Hub model revision"});
    auto directory = repository(model_id);
    if (!directory) return std::unexpected(directory.error());
    if (revision.empty() && validate_downloaded_package(*directory, reference)) return *directory;

    std::cerr << "[kidi] Resolving @" << reference << " in " << local_.string() << '\n' << std::flush;
    std::vector<std::string> metadata_cmd{"hf",          "download",    std::string(model_id),
                                          "config.json", "--local-dir", directory->string()};
    if (!revision.empty()) {
        metadata_cmd.emplace_back("--revision");
        metadata_cmd.emplace_back(revision);
    }
    if (auto status = run_hf(std::move(metadata_cmd), reference); !status) return std::unexpected(status.error());

    YAML::Node raw;
    try {
        raw = YAML::LoadFile((*directory / "config.json").string());
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid config.json: " + std::string(error.what())});
    }
    auto files = hub_files(raw);
    if (!files) return std::unexpected(files.error());

    std::vector<std::string> payload_cmd{"hf", "download", std::string(model_id), "--local-dir", directory->string()};
    if (!revision.empty()) {
        payload_cmd.emplace_back("--revision");
        payload_cmd.emplace_back(revision);
    }
    bool has_payload = false;
    for (const auto& file : *files) {
        if (file == "config.json") continue;
        payload_cmd.emplace_back("--include");
        payload_cmd.emplace_back(file);
        has_payload = true;
    }
    if (has_payload) {
        if (auto status = run_hf(std::move(payload_cmd), reference); !status) return std::unexpected(status.error());
    }
    if (auto valid = validate_downloaded_package(*directory, reference); !valid) return std::unexpected(valid.error());
    std::cerr << "[kidi] Model ready: " << directory->string() << '\n' << std::flush;
    return *directory;
}

} // namespace kidi::checkpoint
