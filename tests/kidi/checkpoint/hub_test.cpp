#include "kidi/checkpoint/hub.h"

#include <fstream>
#include <iostream>

auto main() -> int {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / "kidi-hub-test";
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root / "owner/model");
    std::ofstream(root / "owner/model/config.json") << "{}\n";
    try {
        const kidi::checkpoint::Hub hub{"https://huggingface.co", root};
        const auto model = hub.resolve("owner/model");
        const auto file = hub.get_file("owner/model", "config.json");
        if (!model || *model != root / "owner/model" || !file || *file != *model / "config.json" ||
            hub.remote() != "https://huggingface.co" || hub.local() != root || hub.resolve("../model") ||
            hub.resolve("owner/..") || hub.resolve("owner/model", "../main") ||
            hub.get_file("owner/model", "../secret") || hub.download("owner/model@"))
            return 1;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
    fs::remove_all(root, error);
    return 0;
}
