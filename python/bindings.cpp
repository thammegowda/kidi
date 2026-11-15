#include "kidi/cli/main.h"
#include "kidi/core/version.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

NB_MODULE(_native, module) {
    module.attr("__version__") = std::string(kidi::version());
    module.def(
        "cli",
        [](const std::vector<std::string>& arguments) {
            if (arguments.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("too many CLI arguments");
            std::vector<const char*> argv;
            argv.reserve(arguments.size() + 2);
            argv.push_back("kidi");
            for (const auto& argument : arguments) {
                if (argument.find('\0') != std::string::npos)
                    throw std::invalid_argument("CLI arguments cannot contain NUL bytes");
                argv.push_back(argument.c_str());
            }
            argv.push_back(nullptr);
            return kidi::cli::main(static_cast<int>(arguments.size() + 1), argv.data());
        },
        nanobind::arg("args"));
}