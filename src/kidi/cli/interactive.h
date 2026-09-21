#pragma once

#include "kidi/cli/argparse.h"
#include "kidi/inference/generator.h"

#include <cstdint>

namespace kidi::cli {
auto interactive_chat(inference::Generator& generator, inference::GenerationOptions options, const Namespace& arguments,
                      std::uint64_t load_ns) -> int;
} // namespace kidi::cli