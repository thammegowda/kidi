#include "kidi/cli/main.h"
#include "kidi/core/version.h"

#include <gtest/gtest.h>

#include <array>
#include <iostream>
#include <regex>

namespace {
using kidi::tensor::BackendInfo;
using kidi::tensor::Device;
using kidi::tensor::DeviceKind;
using kidi::tensor::TransformerSupport;

TEST(CliBackend, AutomaticPriorityAndFallback) {
    std::array backends{
        BackendInfo{DeviceKind::A_GPU, "metal", "", true, true, ""},
        BackendInfo{DeviceKind::CUDA, "cuda", "", true, true, ""},
    };
    EXPECT_EQ(kidi::cli::auto_inference_device(backends), Device::cuda());
    backends[1].execution_available = false;
    EXPECT_EQ(kidi::cli::auto_inference_device(backends), Device::apple_gpu());
    backends[1].execution_available = true;
    backends[1].storage_available = false;
    EXPECT_EQ(kidi::cli::auto_inference_device(backends), Device::apple_gpu());
    backends[0].execution_available = false;
    EXPECT_EQ(kidi::cli::auto_inference_device(backends), Device::cpu());
}

TEST(CliBackend, RespectsWorkloadSupport) {
    const std::array backends{
        BackendInfo{DeviceKind::CUDA, "cuda", "", true, true, ""},
        BackendInfo{DeviceKind::A_GPU, "metal", "", true, true, ""},
    };
    EXPECT_EQ(kidi::cli::auto_inference_device(backends, TransformerSupport::GENERAL), Device::apple_gpu());
    EXPECT_EQ(kidi::cli::auto_inference_device(backends, TransformerSupport::QUANTIZED, false), Device::cuda());
    const std::span only_metal{backends.data() + 1, std::size_t{1}};
    EXPECT_EQ(kidi::cli::auto_inference_device(only_metal, TransformerSupport::QUANTIZED, false), Device::cpu());
}

TEST(CliBackend, VersionReportsBackends) {
    testing::internal::CaptureStdout();
    const char* argv[] = {"kidi", "--version"};
    const auto status = kidi::cli::main(2, argv);
    std::cout.flush();
    const auto output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(status, 0);
    EXPECT_TRUE(output.starts_with("kidi " + std::string(kidi::version()) + "\n"));
    EXPECT_NE(output.find("available: ynnpack\n"), std::string::npos);
    std::string unavailable;
    for (const auto& backend : kidi::tensor::BackendRegistry::instance().backends()) {
        const auto available_line = output.find("available: " + backend.name);
        if (backend.execution_available) {
            ASSERT_NE(available_line, std::string::npos) << backend.name;
            const auto unavailable_line = output.find("unavailable: ");
            EXPECT_TRUE(unavailable_line == std::string::npos || available_line < unavailable_line);
#ifdef KIDI_HAS_CUDA
            if (backend.device_kind == DeviceKind::CUDA)
                EXPECT_TRUE(std::regex_search(output, std::regex("available: cuda [0-9]+\\.[0-9]+")));
#endif
        } else {
            EXPECT_EQ(output.find("available: " + backend.name + "\n"), std::string::npos) << backend.name;
            if (!unavailable.empty()) unavailable += ", ";
            unavailable += backend.name;
        }
    }
    if (unavailable.empty())
        EXPECT_EQ(output.find("unavailable: "), std::string::npos);
    else
        EXPECT_TRUE(output.ends_with("unavailable: " + unavailable + "\n")) << output;
}
} // namespace
