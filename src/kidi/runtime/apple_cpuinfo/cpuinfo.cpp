#include "cpuinfo.h"

#include <cstddef>

#include <sys/sysctl.h>

namespace {

bool has_feature(const char* name) {
    int value = 0;
    std::size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && size == sizeof(value) && value != 0;
}

} // namespace

extern "C" {

bool cpuinfo_initialize(void) { return true; }

bool cpuinfo_has_arm_neon(void) { return has_feature("hw.optional.arm.AdvSIMD") || has_feature("hw.optional.neon"); }

bool cpuinfo_has_arm_neon_fma(void) { return cpuinfo_has_arm_neon(); }

bool cpuinfo_has_arm_neon_dot(void) { return has_feature("hw.optional.arm.FEAT_DotProd"); }

bool cpuinfo_has_arm_neon_fp16(void) { return has_feature("hw.optional.neon_fp16"); }

bool cpuinfo_has_arm_neon_fp16_arith(void) { return has_feature("hw.optional.arm.FEAT_FP16"); }

bool cpuinfo_has_arm_neon_bf16(void) { return has_feature("hw.optional.arm.FEAT_BF16"); }

bool cpuinfo_has_arm_i8mm(void) { return has_feature("hw.optional.arm.FEAT_I8MM"); }

bool cpuinfo_has_arm_fp8(void) { return has_feature("hw.optional.arm.FEAT_FP8"); }

bool cpuinfo_has_arm_f8dot(void) { return has_feature("hw.optional.arm.FEAT_F8DOT4"); }

bool cpuinfo_has_arm_sme(void) { return has_feature("hw.optional.arm.FEAT_SME"); }

bool cpuinfo_has_arm_sme2(void) { return has_feature("hw.optional.arm.FEAT_SME2"); }

bool cpuinfo_has_arm_sve(void) { return has_feature("hw.optional.arm.FEAT_SVE"); }

} // extern "C"