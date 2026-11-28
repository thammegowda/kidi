#include "kidi/agent/command.h"

#include <iostream>

auto main(int argc, char** argv) -> int {
    kidi::cli::ArgumentParser parser("kidi-agent");
    kidi::agent::configure_parser(parser);
    try {
        const auto arguments = parser.parse_args(argc, argv);
        if (arguments.should_exit()) {
            std::cout << arguments.output();
            return 0;
        }
        return kidi::agent::agent_command(arguments, kidi::agent::chat::http_provider);
    } catch (const kidi::cli::ParseError& error) {
        std::cerr << error.usage() << "kidi-agent: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "kidi-agent: " << error.what() << '\n';
        return 1;
    }
}
