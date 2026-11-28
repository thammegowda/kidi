#include "kidi/agent/tools/workspace.h"

#include <gtest/gtest.h>

#include <fstream>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace kidi::agent::tools {
namespace {

class AgentWorkspace : public testing::Test {
protected:
    std::filesystem::path root_;
    std::filesystem::path outside_;
    std::vector<Tool> tools_;

    auto SetUp() -> void override {
#if defined(__unix__) || defined(__APPLE__)
        std::string pattern = (std::filesystem::temp_directory_path() / "kidi-agent-test-XXXXXX").string();
        const auto* path = ::mkdtemp(pattern.data());
        ASSERT_NE(path, nullptr);
        root_ = path;
        outside_ = root_ / "outside";
        std::filesystem::create_directory(outside_);
        std::filesystem::create_directory(root_ / "workspace");
        auto tools = workspace_tools(root_ / "workspace", {.max_file_bytes = 1024, .max_directory_entries = 4});
        ASSERT_TRUE(tools) << tools.error().message;
        tools_ = std::move(*tools);
#else
        GTEST_SKIP() << "POSIX workspace APIs required";
#endif
    }

    auto TearDown() -> void override {
        tools_.clear();
        if (!root_.empty()) std::filesystem::remove_all(root_);
    }

    auto invoke(std::string_view name, const Json& args, std::stop_token stop = {}) -> Result<std::string> {
        for (const auto& tool : tools_)
            if (tool.spec.name == name) return tool.invoke(args, stop);
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "missing test tool"});
    }
};

TEST_F(AgentWorkspace, CreateReadList) {
    ASSERT_TRUE(invoke("make_directory", {{"path", "src"}}));
    ASSERT_TRUE(invoke("write_file", {{"path", "src/main.cpp"}, {"content", "int main() {}\n"}}));
    const auto read = invoke("read_file", {{"path", "./src/main.cpp"}});
    ASSERT_TRUE(read) << read.error().message;
    EXPECT_EQ(*read, "int main() {}\n");
    const auto list = invoke("list_files", {{"path", "src"}});
    ASSERT_TRUE(list);
    EXPECT_EQ(Json::parse(*list), Json::array({"main.cpp"}));
}

TEST_F(AgentWorkspace, RunCommand) {
    ASSERT_TRUE(invoke("write_file", {{"path", "a.txt"}, {"content", "hello"}}));
    const auto run = invoke("run_command", {{"command", "cat a.txt; echo err >&2; exit 3"}});
    ASSERT_TRUE(run) << run.error().message;
    const auto result = Json::parse(*run);
    EXPECT_EQ(result["exit_code"], 3);
    EXPECT_EQ(result["output"], "helloerr\n");
    EXPECT_FALSE(invoke("run_command", {{"command", ""}}));

    auto tools = workspace_tools(root_ / "workspace",
                                 {.max_command_output = 4, .command_timeout = std::chrono::milliseconds(300)});
    ASSERT_TRUE(tools);
    const auto& command = *std::ranges::find(*tools, "run_command", [](const Tool& tool) { return tool.spec.name; });
    const auto limited = Json::parse(*command.invoke({{"command", "printf 123456; sleep 5"}}, {}));
    EXPECT_EQ(limited["output"], "1234");
    EXPECT_TRUE(limited["truncated"] && limited["timed_out"]);
}

TEST_F(AgentWorkspace, RegistryFromYaml) {
    std::ofstream(root_ / "agent.yaml") << R"(
version: 1
prompts: {system: "Inspect only"}
tools:
  - name: inspect
    handler: read_file
    description: Inspect a source file.
    parameters:
      type: object
      properties:
        path: {type: string}
      required: [path]
)";
    auto config = load_agent_config(root_ / "agent.yaml");
    ASSERT_TRUE(config) << config.error().message;
    auto tools = workspace_tools(root_ / "workspace", *config);
    ASSERT_TRUE(tools) << tools.error().message;
    ASSERT_EQ(tools->size(), 1);
    EXPECT_EQ(tools->front().spec.name, "inspect");
    EXPECT_EQ(tools->front().capability, Capability::READ);
    std::ofstream(root_ / "workspace" / "file") << "hello";
    auto result = tools->front().invoke({{"path", "file"}}, {});
    ASSERT_TRUE(result);
    EXPECT_EQ(*result, "hello");
    config->tools.front().handler = "unregistered";
    EXPECT_FALSE(workspace_tools(root_ / "workspace", *config));
}

TEST_F(AgentWorkspace, RejectsUnsafePaths) {
    for (const auto& path : {"../outside/secret", "/etc/passwd", "src/../../secret", ".git/config", "./.GIT/config",
                             "src/.git", ".kidi-agent-reserved", ".", ""}) {
        SCOPED_TRACE(path);
        EXPECT_FALSE(invoke("read_file", {{"path", path}}));
        EXPECT_FALSE(invoke("write_file", {{"path", path}, {"content", "bad"}}));
        EXPECT_FALSE(invoke("make_directory", {{"path", path}}));
    }
    EXPECT_FALSE(invoke("read_file", {{"path", std::string("a\0b", 3)}}));
}

TEST_F(AgentWorkspace, RejectsSymlinks) {
    std::ofstream(outside_ / "secret") << "private";
    std::filesystem::create_directory_symlink(outside_, root_ / "workspace" / "link");
    std::filesystem::create_symlink(outside_ / "secret", root_ / "workspace" / "file");
    for (const auto& path : {"link/secret", "file"}) {
        EXPECT_FALSE(invoke("read_file", {{"path", path}}));
        EXPECT_FALSE(invoke("write_file", {{"path", path}, {"content", "bad"}}));
    }
    EXPECT_FALSE(invoke("make_directory", {{"path", "link/new"}}));
    EXPECT_FALSE(invoke("list_files", {{"path", "link"}}));
    std::string content;
    std::ifstream(outside_ / "secret") >> content;
    EXPECT_EQ(content, "private");
}

TEST_F(AgentWorkspace, ReplacesHardLink) {
    std::ofstream(outside_ / "original") << "untouched";
    std::filesystem::create_hard_link(outside_ / "original", root_ / "workspace" / "alias");
    ASSERT_TRUE(invoke("write_file", {{"path", "alias"}, {"content", "new"}}));
    std::string content;
    std::ifstream(outside_ / "original") >> content;
    EXPECT_EQ(content, "untouched");
    EXPECT_EQ(*invoke("read_file", {{"path", "alias"}}), "new");
}

TEST_F(AgentWorkspace, BoundsAndBadArguments) {
    std::ofstream(root_ / "workspace" / "large") << std::string(1025, 'x');
    const auto page = invoke("read_file", {{"path", "large"}});
    ASSERT_TRUE(page);
    EXPECT_EQ(Json::parse(*page)["content"].get<std::string>().size(), 1024);
    EXPECT_EQ(Json::parse(*page)["next_offset"], 1024);
    const auto last = invoke("read_file", {{"path", "large"}, {"offset", 1024}});
    ASSERT_TRUE(last);
    EXPECT_EQ(Json::parse(*last)["content"], "x");
    EXPECT_TRUE(Json::parse(*last)["next_offset"].is_null());
    EXPECT_FALSE(invoke("write_file", {{"path", "small"}, {"content", std::string(1025, 'x')}}));
    EXPECT_FALSE(std::filesystem::exists(root_ / "workspace" / "small"));
    EXPECT_FALSE(invoke("write_file", {{"path", 1}, {"content", "x"}}));
    EXPECT_FALSE(invoke("write_file", {{"path", "file"}}));
    EXPECT_FALSE(invoke("read_file", Json::array()));
    EXPECT_FALSE(invoke("write_file", {{"path", "missing/file"}, {"content", "x"}}));
    EXPECT_FALSE(invoke("read_file", {{"path", "missing"}}));
}

TEST_F(AgentWorkspace, PagedReadPreservesUtf8AndRejectsBadRanges) {
    std::string text;
    for (int i = 0; i < 400; ++i) text += "\xe2\x82\xac\n";
    std::ofstream(root_ / "workspace" / "text") << text;
    std::string actual;
    std::size_t offset = 0;
    for (;;) {
        const auto result = invoke("read_file", {{"path", "text"}, {"offset", offset}, {"max_bytes", 5}});
        ASSERT_TRUE(result) << result.error().message;
        const auto page = Json::parse(*result);
        const auto content = page["content"].get<std::string>();
        EXPECT_LE(content.size(), 5);
        EXPECT_EQ(page["offset"], offset);
        EXPECT_EQ(page["total_bytes"], text.size());
        actual += content;
        if (page["next_offset"].is_null()) break;
        const auto next = page["next_offset"].get<std::size_t>();
        ASSERT_GT(next, offset);
        offset = next;
    }
    EXPECT_EQ(actual, text);
    const auto eof = invoke("read_file", {{"path", "text"}, {"offset", text.size()}});
    ASSERT_TRUE(eof);
    EXPECT_EQ(Json::parse(*eof)["content"], "");
    EXPECT_TRUE(Json::parse(*eof)["next_offset"].is_null());
    for (const auto& args : {Json{{"offset", -1}}, Json{{"offset", 1.5}}, Json{{"offset", text.size() + 1}},
                            Json{{"offset", 1}}, Json{{"max_bytes", 0}}, Json{{"max_bytes", 1025}},
                            Json{{"max_bytes", 1}}, Json{{"max_bytes", "4"}}}) {
        auto request = args;
        request["path"] = "text";
        EXPECT_FALSE(invoke("read_file", request)) << request;
    }
}

TEST_F(AgentWorkspace, DefaultReadPageBoundsLargeToolOutput) {
    std::ofstream(root_ / "workspace" / "large") << std::string(70000, 'x');
    auto tools = workspace_tools(root_ / "workspace");
    ASSERT_TRUE(tools);
    tools_ = std::move(*tools);
    const auto result = invoke("read_file", {{"path", "large"}});
    ASSERT_TRUE(result) << result.error().message;
    const auto page = Json::parse(*result);
    EXPECT_EQ(page["content"].get<std::string>().size(), 4096);
    EXPECT_EQ(page["next_offset"], 4096);
    EXPECT_EQ(page["total_bytes"], 70000);
}

TEST_F(AgentWorkspace, MetadataAndCancellation) {
    std::filesystem::create_directory(root_ / "workspace" / ".git");
    std::ofstream(root_ / "workspace" / ".git" / "config") << "metadata";
    EXPECT_FALSE(invoke("read_file", {{"path", ".git/config"}}));
    EXPECT_FALSE(invoke("write_file", {{"path", ".git/config"}, {"content", "changed"}}));
    const auto list = invoke("list_files", {{"path", "."}});
    ASSERT_TRUE(list);
    EXPECT_TRUE(Json::parse(*list).empty());
    std::ofstream(root_ / "workspace" / "file") << "original";
    std::stop_source source;
    source.request_stop();
    EXPECT_FALSE(invoke("write_file", {{"path", "file"}, {"content", "changed"}}, source.get_token()));
    EXPECT_EQ(*invoke("read_file", {{"path", "file"}}), "original");
    EXPECT_EQ(Json::parse(*invoke("list_files", {{"path", "."}})).size(), 1);
}

TEST_F(AgentWorkspace, RejectsSpecialAndBinary) {
#if defined(__unix__) || defined(__APPLE__)
    ASSERT_EQ(::mkfifo((root_ / "workspace" / "pipe").c_str(), 0600), 0);
    EXPECT_FALSE(invoke("read_file", {{"path", "pipe"}}));
    EXPECT_FALSE(invoke("write_file", {{"path", "pipe"}, {"content", "x"}}));
#endif
    std::ofstream(root_ / "workspace" / "binary") << std::string("\xff\0", 2);
    EXPECT_FALSE(invoke("read_file", {{"path", "binary"}}));
    EXPECT_FALSE(invoke("write_file", {{"path", "new"}, {"content", std::string("\xff", 1)}}));
    EXPECT_FALSE(invoke("write_file", {{"path", "new"}, {"content", std::string("a\0b", 3)}}));
    EXPECT_FALSE(std::filesystem::exists(root_ / "workspace" / "new"));
}

TEST_F(AgentWorkspace, LimitsDirectory) {
    for (int i = 0; i < 5; ++i) std::ofstream(root_ / "workspace" / std::to_string(i)) << "x";
    EXPECT_FALSE(invoke("list_files", {{"path", "."}}));
}

} // namespace
} // namespace kidi::agent::tools
