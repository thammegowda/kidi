#include "kidi/audio/wav.h"

#include <array>
#include <cassert>
#include <filesystem>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "kidi-wav-test.wav";
    const std::array samples{-1.F, -0.25F, 0.F, 0.25F, 1.F};
    auto saved = kidi::audio::save_wav(path, samples, 24000);
    assert(saved);
    auto loaded = kidi::audio::load_wav(path);
    assert(loaded);
    assert(loaded->sample_rate == 24000);
    assert(loaded->samples.size() == samples.size());
    for (std::size_t index = 0; index < samples.size(); ++index)
        assert(std::abs(loaded->samples[index] - samples[index]) < 1e-4F);
    std::filesystem::remove(path);
}
