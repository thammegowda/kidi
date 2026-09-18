// Torch-free, zero-copy memory-mapped safetensors reader.
//
// This header owns the low-level, dependency-free parts of the vendored
// safetensors library: limits, the exception type, header metadata parsing,
// endian helpers, and an mmap-backed reader that exposes immutable byte views
// into the file without copying tensor payloads.
//
// `safetensors.hpp` includes this header and adds the LibTorch dtype mapping,
// materialization, and save adapters on top of it. Consumers that must remain
// Torch-free (tahoma-core checkpoint loading) include this header directly.
#ifndef SAFETENSORS_MAPPED_HPP
#define SAFETENSORS_MAPPED_HPP

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Arbitrary but reasonable limits; keep in sync with any consumer expectations.
#ifndef SAFETENSORS_MAX_DIM
#define SAFETENSORS_MAX_DIM 8
#endif
#ifndef SAFETENSORS_MAX_TENSORS
#define SAFETENSORS_MAX_TENSORS 1000000
#endif
#ifndef SAFETENSORS_MAX_FILE_SIZE
#define SAFETENSORS_MAX_FILE_SIZE (2ULL << 40)  // 2 TiB
#endif
#ifndef SAFETENSORS_MAX_STRING_SIZE
#define SAFETENSORS_MAX_STRING_SIZE 8192
#endif
#ifndef SAFETENSORS_MAX_METADATA_SIZE
#define SAFETENSORS_MAX_METADATA_SIZE (2ULL << 22)  // 4 MiB
#endif

namespace safetensors {

class SafetensorsException : public std::runtime_error {
public:
    explicit SafetensorsException(const std::string& message)
        : std::runtime_error(message) {}
};

// Parsed header entry for one tensor. Offsets are relative to the start of the
// data region (immediately after the 8-byte header length and the JSON header).
struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    std::array<size_t, 2> data_offsets;
};

inline bool is_big_endian() {
    union {
        uint32_t i;
        char c[4];
    } probe = {0x01020304};
    return probe.c[0] == 1;
}

template <typename T>
inline T swap_endian(T value) {
    static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");
    union {
        T value;
        unsigned char bytes[sizeof(T)];
    } source, dest;
    source.value = value;
    for (size_t k = 0; k < sizeof(T); k++) {
        dest.bytes[k] = source.bytes[sizeof(T) - k - 1];
    }
    return dest.value;
}

inline void validate_string_length(
        const std::string& value, const std::string& context) {
    if (value.length() > SAFETENSORS_MAX_STRING_SIZE) {
        throw SafetensorsException(context + " exceeds maximum allowed length");
    }
}

// Byte width of a safetensors dtype string. Throws for unknown dtypes.
inline size_t dtype_byte_size(std::string_view dtype) {
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" ||
        dtype == "F8_E4M3" || dtype == "F8_E5M2") {
        return 1;
    }
    if (dtype == "U16" || dtype == "I16" || dtype == "F16" || dtype == "BF16") {
        return 2;
    }
    if (dtype == "U32" || dtype == "I32" || dtype == "F32") {
        return 4;
    }
    if (dtype == "U64" || dtype == "I64" || dtype == "F64") {
        return 8;
    }
    throw SafetensorsException("unknown safetensors dtype: " + std::string{dtype});
}

// Bounds-checked recursive-descent parser for the safetensors JSON header.
// Every read is guarded against the header length, so a malformed or truncated
// header raises SafetensorsException instead of reading past the mapped region.
class HeaderParser {
public:
    HeaderParser(const char* data, size_t length)
        : data_{data}, length_{length}, position_{0} {}

    std::unordered_map<std::string, TensorInfo> parse() {
        std::unordered_map<std::string, TensorInfo> result;
        skip_whitespace();
        expect('{');
        skip_whitespace();
        if (peek() == '}') {
            ++position_;
            return result;
        }
        while (true) {
            skip_whitespace();
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            if (key == "__metadata__") {
                skip_value();
            } else if (!result.emplace(std::move(key), parse_tensor_info())
                            .second) {
                throw SafetensorsException(
                    "duplicate tensor name in safetensors header");
            }
            skip_whitespace();
            const char separator = next();
            if (separator == ',') continue;
            if (separator == '}') break;
            throw SafetensorsException("malformed safetensors header");
        }
        return result;
    }

private:
    const char* data_;
    size_t length_;
    size_t position_;

    [[noreturn]] void fail_truncated() const {
        throw SafetensorsException("truncated safetensors header");
    }

    char peek() const {
        if (position_ >= length_) fail_truncated();
        return data_[position_];
    }

    char next() {
        const char value = peek();
        ++position_;
        return value;
    }

    void expect(char expected) {
        if (next() != expected) {
            throw SafetensorsException("malformed safetensors header");
        }
    }

    void skip_whitespace() {
        while (position_ < length_) {
            const char value = data_[position_];
            if (value != ' ' && value != '\n' && value != '\r' &&
                value != '\t') {
                break;
            }
            ++position_;
        }
    }

    std::string parse_string() {
        expect('"');
        std::string value;
        while (true) {
            if (position_ >= length_) fail_truncated();
            const char character = data_[position_++];
            if (character == '"') break;
            if (character == '\\') {
                if (position_ >= length_) fail_truncated();
                const char escaped = data_[position_++];
                switch (escaped) {
                    case '"': value.push_back('"'); break;
                    case '\\': value.push_back('\\'); break;
                    case '/': value.push_back('/'); break;
                    case 'b': value.push_back('\b'); break;
                    case 'f': value.push_back('\f'); break;
                    case 'n': value.push_back('\n'); break;
                    case 'r': value.push_back('\r'); break;
                    case 't': value.push_back('\t'); break;
                    case 'u':
                        if (position_ + 4 > length_) fail_truncated();
                        position_ += 4;
                        value.push_back('?');
                        break;
                    default: value.push_back(escaped); break;
                }
            } else {
                value.push_back(character);
            }
            if (value.size() > SAFETENSORS_MAX_STRING_SIZE) {
                throw SafetensorsException(
                    "string exceeds maximum allowed length");
            }
        }
        return value;
    }

    uint64_t parse_uint() {
        skip_whitespace();
        if (position_ >= length_ || data_[position_] < '0' ||
            data_[position_] > '9') {
            throw SafetensorsException("expected number in safetensors header");
        }
        uint64_t value = 0;
        while (position_ < length_ && data_[position_] >= '0' &&
               data_[position_] <= '9') {
            const uint64_t digit =
                static_cast<uint64_t>(data_[position_] - '0');
            if (value > (UINT64_MAX - digit) / 10) {
                throw SafetensorsException(
                    "number overflow in safetensors header");
            }
            value = value * 10 + digit;
            ++position_;
        }
        return value;
    }

    std::vector<int64_t> parse_int_array() {
        expect('[');
        std::vector<int64_t> result;
        skip_whitespace();
        if (peek() == ']') {
            ++position_;
            return result;
        }
        while (true) {
            skip_whitespace();
            const uint64_t value = parse_uint();
            if (value > static_cast<uint64_t>(INT64_MAX)) {
                throw SafetensorsException(
                    "dimension overflow in safetensors header");
            }
            result.push_back(static_cast<int64_t>(value));
            if (result.size() > SAFETENSORS_MAX_DIM) {
                throw SafetensorsException(
                    "Tensor dimension exceeds maximum allowed");
            }
            skip_whitespace();
            const char separator = next();
            if (separator == ',') continue;
            if (separator == ']') break;
            throw SafetensorsException("malformed array in safetensors header");
        }
        return result;
    }

    std::array<size_t, 2> parse_offsets() {
        expect('[');
        std::array<size_t, 2> result{0, 0};
        skip_whitespace();
        result[0] = static_cast<size_t>(parse_uint());
        skip_whitespace();
        expect(',');
        skip_whitespace();
        result[1] = static_cast<size_t>(parse_uint());
        skip_whitespace();
        expect(']');
        return result;
    }

    TensorInfo parse_tensor_info() {
        TensorInfo info;
        bool has_dtype = false;
        bool has_shape = false;
        bool has_offsets = false;
        expect('{');
        skip_whitespace();
        if (peek() == '}') {
            throw SafetensorsException(
                "empty tensor entry in safetensors header");
        }
        while (true) {
            skip_whitespace();
            const std::string key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            if (key == "dtype") {
                info.dtype = parse_string();
                has_dtype = true;
            } else if (key == "shape") {
                info.shape = parse_int_array();
                has_shape = true;
            } else if (key == "data_offsets") {
                info.data_offsets = parse_offsets();
                has_offsets = true;
            } else {
                skip_value();
            }
            skip_whitespace();
            const char separator = next();
            if (separator == ',') continue;
            if (separator == '}') break;
            throw SafetensorsException(
                "malformed tensor entry in safetensors header");
        }
        if (!has_dtype || !has_shape || !has_offsets) {
            throw SafetensorsException(
                "incomplete tensor entry in safetensors header");
        }
        return info;
    }

    void skip_value() {
        skip_whitespace();
        const char character = peek();
        if (character == '"') {
            parse_string();
            return;
        }
        if (character == '[') {
            ++position_;
            skip_whitespace();
            if (peek() == ']') {
                ++position_;
                return;
            }
            while (true) {
                skip_value();
                skip_whitespace();
                const char separator = next();
                if (separator == ',') continue;
                if (separator == ']') break;
                throw SafetensorsException(
                    "malformed array in safetensors header");
            }
            return;
        }
        if (character == '{') {
            ++position_;
            skip_whitespace();
            if (peek() == '}') {
                ++position_;
                return;
            }
            while (true) {
                skip_whitespace();
                parse_string();
                skip_whitespace();
                expect(':');
                skip_value();
                skip_whitespace();
                const char separator = next();
                if (separator == ',') continue;
                if (separator == '}') break;
                throw SafetensorsException(
                    "malformed object in safetensors header");
            }
            return;
        }
        while (position_ < length_ && data_[position_] != ',' &&
               data_[position_] != '}' && data_[position_] != ']') {
            ++position_;
        }
    }
};

// RAII owner of a read-only memory mapping. Movable, non-copyable. Payload
// tensor views hold a shared_ptr to this owner so the mapping outlives them.
class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
        const int descriptor = ::open(path.c_str(), O_RDONLY);
        if (descriptor == -1) {
            throw SafetensorsException("failed to open file: " + path);
        }
        struct stat status {};
        if (::fstat(descriptor, &status) == -1) {
            ::close(descriptor);
            throw SafetensorsException("failed to stat file: " + path);
        }
        size_ = static_cast<size_t>(status.st_size);
        if (size_ == 0) {
            ::close(descriptor);
            throw SafetensorsException("empty safetensors file: " + path);
        }
        if (size_ > SAFETENSORS_MAX_FILE_SIZE) {
            ::close(descriptor);
            throw SafetensorsException(
                "file size exceeds maximum allowed size: " + path);
        }
        void* mapped =
            ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor, 0);
        ::close(descriptor);
        if (mapped == MAP_FAILED) {
            throw SafetensorsException("failed to memory map file: " + path);
        }
        data_ = static_cast<const std::byte*>(mapped);
    }

    ~MappedFile() {
        if (data_ != nullptr) {
            ::munmap(const_cast<std::byte*>(data_), size_);
        }
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept
        : data_{other.data_}, size_{other.size_} {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                ::munmap(const_cast<std::byte*>(data_), size_);
            }
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    const std::byte* data() const { return data_; }
    size_t size() const { return size_; }

private:
    const std::byte* data_ = nullptr;
    size_t size_ = 0;
};

// One tensor as an immutable view into the mapped file. `data`/`nbytes` point
// directly at the payload; no bytes are copied. Payload alignment is only what
// the writer produced, so consumers that require alignment must handle it.
struct MappedTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const std::byte* data = nullptr;
    size_t nbytes = 0;

    std::span<const std::byte> bytes() const { return {data, nbytes}; }
    bool is_aligned(size_t alignment) const {
        return alignment != 0 &&
               reinterpret_cast<std::uintptr_t>(data) % alignment == 0;
    }
};

// A validated, memory-mapped safetensors checkpoint. Construction maps the
// file, parses the header, and validates every entry's shape, dtype, and byte
// range before exposing any view.
class MappedCheckpoint {
public:
    explicit MappedCheckpoint(const std::string& path)
        : file_{std::make_shared<MappedFile>(path)} {
        parse_and_validate(path);
    }

    const std::unordered_map<std::string, MappedTensor>& tensors() const {
        return tensors_;
    }

    std::shared_ptr<const MappedFile> owner() const { return file_; }

    bool contains(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

    const MappedTensor& at(const std::string& name) const {
        const auto found = tensors_.find(name);
        if (found == tensors_.end()) {
            throw SafetensorsException("tensor not found: " + name);
        }
        return found->second;
    }

private:
    void parse_and_validate(const std::string& path) {
        const std::byte* base = file_->data();
        const size_t file_size = file_->size();
        if (file_size < 8) {
            throw SafetensorsException("safetensors file too small: " + path);
        }
        uint64_t header_size = 0;
        std::memcpy(&header_size, base, sizeof(uint64_t));
        if (is_big_endian()) {
            header_size = swap_endian(header_size);
        }
        if (header_size > SAFETENSORS_MAX_METADATA_SIZE) {
            throw SafetensorsException(
                "safetensors header exceeds metadata limit: " + path);
        }
        if (header_size > file_size - 8) {
            throw SafetensorsException(
                "safetensors header exceeds file size: " + path);
        }
        const char* header = reinterpret_cast<const char*>(base + 8);
        HeaderParser parser(header, static_cast<size_t>(header_size));
        const auto infos = parser.parse();
        if (infos.size() > SAFETENSORS_MAX_TENSORS) {
            throw SafetensorsException(
                "number of tensors exceeds maximum allowed: " + path);
        }
        const std::byte* data_start = base + 8 + header_size;
        const size_t data_length =
            file_size - 8 - static_cast<size_t>(header_size);
        tensors_.reserve(infos.size());
        for (const auto& [name, info] : infos) {
            validate_string_length(name, "Tensor name");
            if (info.shape.size() > SAFETENSORS_MAX_DIM) {
                throw SafetensorsException(
                    "Tensor dimension exceeds maximum allowed");
            }
            const size_t element_size = dtype_byte_size(info.dtype);
            uint64_t element_count = 1;
            for (const int64_t dimension : info.shape) {
                if (dimension < 0) {
                    throw SafetensorsException(
                        "negative dimension for tensor '" + name + "'");
                }
                const uint64_t extent = static_cast<uint64_t>(dimension);
                if (extent != 0 && element_count > UINT64_MAX / extent) {
                    throw SafetensorsException(
                        "element count overflow for tensor '" + name + "'");
                }
                element_count *= extent;
            }
            const size_t begin = info.data_offsets[0];
            const size_t end = info.data_offsets[1];
            if (end < begin) {
                throw SafetensorsException(
                    "inverted data offsets for tensor '" + name + "'");
            }
            const uint64_t span = static_cast<uint64_t>(end - begin);
            if (element_count > UINT64_MAX / element_size) {
                throw SafetensorsException(
                    "byte length overflow for tensor '" + name + "'");
            }
            if (span != element_count * element_size) {
                throw SafetensorsException(
                    "byte length does not match shape for tensor '" + name +
                    "'");
            }
            if (begin > data_length || span > data_length - begin) {
                throw SafetensorsException(
                    "data range exceeds file for tensor '" + name + "'");
            }
            MappedTensor tensor;
            tensor.dtype = info.dtype;
            tensor.shape = info.shape;
            tensor.data = data_start + begin;
            tensor.nbytes = static_cast<size_t>(span);
            tensors_.emplace(name, std::move(tensor));
        }
    }

    std::shared_ptr<MappedFile> file_;
    std::unordered_map<std::string, MappedTensor> tensors_;
};

inline MappedCheckpoint load_mapped(const std::string& path) {
    return MappedCheckpoint(path);
}

}  // namespace safetensors

#endif  // SAFETENSORS_MAPPED_HPP
