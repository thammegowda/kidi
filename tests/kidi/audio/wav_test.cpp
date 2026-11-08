#include "kidi/audio/wav.h"

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

namespace {
auto write_fixture(const std::filesystem::path& path, std::uint16_t format, std::uint16_t bits,
                   std::span<const float> samples, bool extensible = false) -> void {
    std::ofstream output(path, std::ios::binary);
    const auto write = [&](auto value) {
        for (std::size_t index = 0; index < sizeof(value); ++index)
            output.put(static_cast<char>((value >> (index * 8)) & 0xff));
    };
    const auto bytes = static_cast<std::uint32_t>(samples.size() * (bits / 8));
    output.write("RIFF", 4);
    write(std::uint32_t{46} + (extensible ? 24 : 0) + bytes);
    output.write("WAVEJUNK", 8);
    write(std::uint32_t{1});
    output.write("x\0fmt ", 6);
    write(std::uint32_t{extensible ? 40U : 16U});
    write(static_cast<std::uint16_t>(extensible ? 0xfffe : format));
    write(std::uint16_t{2});
    write(std::uint32_t{16000});
    write(static_cast<std::uint32_t>(16000 * 2 * (bits / 8)));
    write(static_cast<std::uint16_t>(2 * (bits / 8)));
    write(bits);
    if (extensible) {
        write(std::uint16_t{22});
        write(bits);
        write(std::uint32_t{3});
        write(static_cast<std::uint32_t>(format));
        write(std::uint16_t{0});
        write(std::uint16_t{0x10});
        for (const std::uint8_t value : {0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71}) write(value);
    }
    output.write("data", 4);
    write(bytes);
    for (const auto value : samples) {
        const auto raw = format == 3 ? std::bit_cast<std::uint32_t>(value)
                                     : static_cast<std::uint32_t>(static_cast<std::int32_t>(
                                           value * static_cast<double>(std::uint64_t{1} << (bits - 1))));
        for (int index = 0; index < bits / 8; ++index) output.put(static_cast<char>((raw >> (index * 8)) & 0xff));
    }
}
} // namespace

auto main() -> int {
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
    {
        auto reader = kidi::audio::WavReader::open(path);
        assert(reader && reader->sample_count() == samples.size() && reader->sample_rate() == 24000);
        assert(!reader->read(0));
        std::vector<float> streamed;
        for (;;) {
            auto part = reader->read(2);
            assert(part && part->size() <= 2);
            if (part->empty()) break;
            streamed.insert(streamed.end(), part->begin(), part->end());
        }
        assert(streamed == loaded->samples);
        assert(reader->read(1)->empty());
    }

    std::vector<float> long_audio(480003, 0.25F);
    long_audio[480000] = -0.5F;
    assert(kidi::audio::save_wav(path, long_audio, 16000));
    {
        auto reader = kidi::audio::WavReader::open(path);
        assert(reader && reader->sample_count() == long_audio.size());
        assert(reader->read(480000)->size() == 480000);
        const auto tail = reader->read(480000);
        assert(tail && tail->size() == 3 && std::abs((*tail)[0] + 0.5F) < 1e-4F);
        assert(reader->read(480000)->empty());
    }

    const std::array stereo{-0.5F, 0.5F, -1.F, -0.5F, 0.25F, 0.75F};
    for (const auto format : {1, 3}) {
        for (const auto bits : {16, 24, 32}) {
            if (format == 3 && bits != 32) continue;
            for (const bool extensible : {false, true}) {
                write_fixture(path, format, bits, stereo, extensible);
                {
                    auto reader = kidi::audio::WavReader::open(path);
                    assert(reader && reader->sample_count() == 3 && reader->sample_rate() == 16000);
                    const auto first = reader->read(2);
                    const auto last = reader->read(2);
                    assert(first && last && first->size() == 2 && last->size() == 1);
                    assert((*first)[0] == 0.F && (*first)[1] == -0.75F && (*last)[0] == 0.5F);
                }
                std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
                assert(!kidi::audio::WavReader::open(path));
            }
        }
    }
    // Corrupt the extension length, valid-bit count, and subformat GUID.
    for (const auto offset : {46, 48, 69}) {
        write_fixture(path, 1, 16, stereo, true);
        std::fstream altered(path, std::ios::binary | std::ios::in | std::ios::out);
        altered.seekp(offset);
        altered.put('\0');
        altered.close();
        assert(!kidi::audio::WavReader::open(path));
    }
    write_fixture(path, 1, 16, std::span(stereo).first(5));
    assert(!kidi::audio::WavReader::open(path));
    const std::array invalid{std::numeric_limits<float>::quiet_NaN(), 0.F};
    write_fixture(path, 3, 32, invalid);
    assert(!kidi::audio::load_wav(path));
    std::filesystem::remove(path);
}
