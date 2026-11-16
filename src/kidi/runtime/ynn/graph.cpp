#include "kidi/runtime/ynn/graph.h"

#include <string>

namespace kidi::runtime::ynn {

auto check_status(ynn_status status, const char* operation) -> Result<void> {
    if (status == ynn_status_success) {
        return {};
    }
    return std::unexpected(Error{
        ErrorCode::RUNTIME,
        std::string(operation) + " failed with YNNPACK status " + std::to_string(status),
    });
}

} // namespace kidi::runtime::ynn