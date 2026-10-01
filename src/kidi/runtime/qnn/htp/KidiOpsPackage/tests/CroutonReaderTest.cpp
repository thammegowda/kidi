#include "AttentionCrouton.h"

#include <array>
#include <cstddef>
#include <cstdio>

namespace {

constexpr std::array<std::size_t, 4> SHAPE{{1, 19, 4, 256}};
constexpr std::array<std::size_t, 4> PADDED{{1, 24, 4, 256}};
constexpr std::size_t BLOCK_COUNT = 24;
std::array<std::array<Float16, R4Crouton2Layout::chunk_total>, BLOCK_COUNT> storage{};
std::array<void*, BLOCK_COUNT> blocks{};

} // namespace

int main() {
    for (std::size_t index = 0; index < BLOCK_COUNT; ++index) blocks[index] = storage[index].data();

    for (std::size_t sequence = 0; sequence < SHAPE[1]; ++sequence)
        for (std::size_t head = 0; head < SHAPE[2]; ++head)
            for (std::size_t depth = 0; depth < SHAPE[3]; ++depth) {
                const std::array<std::size_t, 4> coordinates{{0, sequence, head, depth}};
                const auto block = R4Crouton2Layout::chunk_index(coordinates, PADDED);
                const auto offset = R4Crouton2Layout::chunk_offset(coordinates, PADDED);
                storage[block][offset] = Float16(static_cast<float>(sequence * 1000 + head * 256 + depth) / 1024.0f);
            }

    kidi::qnn::htp::Crouton16Reader reader(blocks.data(), PADDED);
    alignas(128) std::array<Float16, 64> actual{};
    for (std::size_t sequence = 0; sequence < SHAPE[1]; ++sequence)
        for (std::size_t head = 0; head < SHAPE[2]; ++head)
            for (std::size_t depth = 0; depth < SHAPE[3]; depth += 64) {
                q6op_vstu_AV(actual.data(), reader.load64(sequence, head, depth));
                for (std::size_t channel = 0; channel < actual.size(); ++channel) {
                    const Float16 expected(static_cast<float>(sequence * 1000 + head * 256 + depth + channel) /
                                           1024.0f);
                    if (actual[channel].raw() != expected.raw()) {
                        std::printf(
                            "Crouton mismatch seq=%zu head=%zu depth=%zu channel=%zu expected=%04x actual=%04x\n",
                            sequence, head, depth, channel, expected.raw(), actual[channel].raw());
                        return 1;
                    }
                }
            }
    std::printf("Crouton reader reconstructed every 64-depth vector\n");
}
