#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool cpuinfo_initialize(void);
bool cpuinfo_has_arm_neon(void);
bool cpuinfo_has_arm_neon_fma(void);
bool cpuinfo_has_arm_neon_dot(void);
bool cpuinfo_has_arm_neon_fp16(void);
bool cpuinfo_has_arm_neon_fp16_arith(void);
bool cpuinfo_has_arm_neon_bf16(void);
bool cpuinfo_has_arm_i8mm(void);
bool cpuinfo_has_arm_fp8(void);
bool cpuinfo_has_arm_f8dot(void);
bool cpuinfo_has_arm_sme(void);
bool cpuinfo_has_arm_sme2(void);
bool cpuinfo_has_arm_sve(void);

#ifdef __cplusplus
}
#endif