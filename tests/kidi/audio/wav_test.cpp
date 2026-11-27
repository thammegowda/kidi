#include "kidi/audio/wav.h"

#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

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

// Per-test file so tests discovered as separate ctest entries can run in parallel.
auto test_path() -> std::filesystem::path {
    return std::filesystem::temp_directory_path() /
           (std::string("kidi-wav-test-") + ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".wav");
}
const std::array STEREO{-0.5F, 0.5F, -1.F, -0.5F, 0.25F, 0.75F};

TEST(WavTest, SaveLoadAndStreamRoundTrip) {
    const auto path = test_path();
    const std::array samples{-1.F, -0.25F, 0.F, 0.25F, 1.F};
    auto saved = kidi::audio::save_wav(path, samples, 24000);
    ASSERT_TRUE(saved);
    auto loaded = kidi::audio::load_wav(path);
    ASSERT_TRUE(loaded);
    EXPECT_EQ(loaded->sample_rate, 24000U);
    ASSERT_EQ(loaded->samples.size(), samples.size());
    for (std::size_t index = 0; index < samples.size(); ++index)
        EXPECT_NEAR(loaded->samples[index], samples[index], 1e-4F) << "sample " << index;
    auto reader = kidi::audio::WavReader::open(path);
    ASSERT_TRUE(reader);
    EXPECT_EQ(reader->sample_count(), samples.size());
    EXPECT_EQ(reader->sample_rate(), 24000U);
    EXPECT_FALSE(reader->read(0));
    std::vector<float> streamed;
    for (;;) {
        auto part = reader->read(2);
        ASSERT_TRUE(part);
        ASSERT_LE(part->size(), 2U);
        if (part->empty()) break;
        streamed.insert(streamed.end(), part->begin(), part->end());
    }
    EXPECT_EQ(streamed, loaded->samples);
    auto end = reader->read(1);
    ASSERT_TRUE(end);
    EXPECT_TRUE(end->empty());
    std::filesystem::remove(path);
}

TEST(WavTest, StreamsLongAudio) {
    const auto path = test_path();
    std::vector<float> long_audio(480003, 0.25F);
    long_audio[480000] = -0.5F;
    ASSERT_TRUE(kidi::audio::save_wav(path, long_audio, 16000));
    auto reader = kidi::audio::WavReader::open(path);
    ASSERT_TRUE(reader);
    EXPECT_EQ(reader->sample_count(), long_audio.size());
    auto head = reader->read(480000);
    ASSERT_TRUE(head);
    EXPECT_EQ(head->size(), 480000U);
    const auto tail = reader->read(480000);
    ASSERT_TRUE(tail);
    ASSERT_EQ(tail->size(), 3U);
    EXPECT_NEAR((*tail)[0], -0.5F, 1e-4F);
    auto end = reader->read(480000);
    ASSERT_TRUE(end);
    EXPECT_TRUE(end->empty());
    std::filesystem::remove(path);
}

TEST(WavTest, ReadsPcmAndFloatFormats) {
    const auto path = test_path();
    for (const auto format : {1, 3}) {
        for (const auto bits : {16, 24, 32}) {
            if (format == 3 && bits != 32) continue;
            for (const bool extensible : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "format=" << format << " bits=" << bits
                                                  << " extensible=" << extensible);
                write_fixture(path, format, bits, STEREO, extensible);
                {
                    auto reader = kidi::audio::WavReader::open(path);
                    ASSERT_TRUE(reader);
                    EXPECT_EQ(reader->sample_count(), 3U);
                    EXPECT_EQ(reader->sample_rate(), 16000U);
                    const auto first = reader->read(2);
                    const auto last = reader->read(2);
                    ASSERT_TRUE(first);
                    ASSERT_TRUE(last);
                    ASSERT_EQ(first->size(), 2U);
                    ASSERT_EQ(last->size(), 1U);
                    EXPECT_EQ((*first)[0], 0.F);
                    EXPECT_EQ((*first)[1], -0.75F);
                    EXPECT_EQ((*last)[0], 0.5F);
                }
                std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
                EXPECT_FALSE(kidi::audio::WavReader::open(path));
            }
        }
    }
    std::filesystem::remove(path);
}

TEST(WavTest, RejectsInvalidFiles) {
    const auto path = test_path();
    // Corrupt the extension length, valid-bit count, and subformat GUID.
    for (const auto offset : {46, 48, 69}) {
        write_fixture(path, 1, 16, STEREO, true);
        std::fstream altered(path, std::ios::binary | std::ios::in | std::ios::out);
        altered.seekp(offset);
        altered.put('\0');
        altered.close();
        EXPECT_FALSE(kidi::audio::WavReader::open(path)) << "offset " << offset;
    }
    write_fixture(path, 1, 16, std::span(STEREO).first(5));
    EXPECT_FALSE(kidi::audio::WavReader::open(path));
    const std::array invalid{std::numeric_limits<float>::quiet_NaN(), 0.F};
    write_fixture(path, 3, 32, invalid);
    EXPECT_FALSE(kidi::audio::load_wav(path));
    std::filesystem::remove(path);
}

} // namespace
