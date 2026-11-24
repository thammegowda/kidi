#include "kidi/code/command.h"

#include <iostream>

auto main(int argc, char** argv) -> int {
    kidi::cli::ArgumentParser parser("kidi-code");
    kidi::code::configure_parser(parser);
    try {
        const auto arguments = parser.parse_args(argc, argv);
        if (arguments.should_exit()) {
            std::cout << arguments.output();
            return 0;
        }
        return kidi::code::code_command(arguments, kidi::code::http_provider);
    } catch (const kidi::cli::ParseError& error) {
        std::cerr << error.usage() << "kidi-code: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "kidi-code: " << error.what() << '\n';
        return 1;
    }
}
