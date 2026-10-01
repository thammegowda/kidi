QNN_SDK_ROOT ?= $(abspath ../../../../../../benchmarks/android/.cache/qairt/2.50.0.260828)
HEXAGON_SDK_ROOT ?= $(abspath ../../../../../../benchmarks/android/.cache/hexagon-sdk/v6.3.0)
HEXAGON_TOOLS_VERSION ?= 19.0.04
ANDROID_NDK_ROOT ?= $(HOME)/Library/Android/sdk/ndk/28.0.13004108

QNN_INCLUDE := $(QNN_SDK_ROOT)/include/QNN
HEXAGON_CXX := $(HEXAGON_SDK_ROOT)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION)/Tools/bin/hexagon-clang++
HEXAGON_SIM := $(HEXAGON_SDK_ROOT)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION)/Tools/bin/hexagon-sim
QURT_INCLUDE := $(HEXAGON_SDK_ROOT)/rtos/qurt/computev79/include
OUTPUT := build/hexagon-v79/libQnnKidiOpsHtp.so
SIM_TEST := build/hexagon-v79/hvx_kernel_test.elf
SIM_BENCH := build/hexagon-v79/hvx_kernel_bench.elf
SIM_HMX_TEST := build/hexagon-v79/hmx_tile_test.elf
SIM_HMX_PACK_TEST := build/hexagon-v79/hmx_packing_test.elf
SIM_CROUTON_TEST := build/hexagon-v79/crouton_reader_test.elf
SIM_PHASE_BENCH := build/hexagon-v79/prefill_phase_bench.elf
AARCH64_OUTPUT := build/aarch64-android/libQnnKidiOps.so
SOURCES := src/KidiOpsPackageInterface.cpp src/ops/StructuredAttention.cpp src/ops/FusedOps.cpp
OBJECTS := $(patsubst src/%.cpp,build/hexagon-v79/%.o,$(SOURCES))
HOST_CXX ?= c++
HOST_TEST := build/host/attention_math_test

CXXFLAGS := -std=c++17 -fPIC -fvisibility=default -stdlib=libc++ -O3 -ffast-math
CXXFLAGS += -Wall -Werror -Wno-format -Wno-missing-braces -Wno-unused-function
CXXFLAGS += -Wno-unused-command-line-argument -Wno-reorder
CXXFLAGS += -mhvx -mhvx-length=128B -mhmx -mv79 -DUSE_OS_QURT -DPREPARE_DISABLED
CXXFLAGS += -DQNN_API='__attribute__((visibility("default")))'
CXXFLAGS += -D__QAIC_HEADER_EXPORT='__attribute__((visibility("default")))'
CXXFLAGS += -DTHIS_PKG_NAME=KidiOps
CXXFLAGS += $(EXTRA_CXXFLAGS)
CXXFLAGS += -I$(QNN_INCLUDE) -I$(QNN_INCLUDE)/HTP/core
CXXFLAGS += -I$(QURT_INCLUDE)/qurt -I$(QURT_INCLUDE)/posix
CXXFLAGS += -I$(HEXAGON_SDK_ROOT)/incs -I$(HEXAGON_SDK_ROOT)/incs/stddef

.PHONY: all aarch64 host-test sim-test sim-bench sim-hmx-test sim-hmx-pack-test sim-crouton-test sim-phase-bench clean
all: $(OUTPUT)
aarch64: $(AARCH64_OUTPUT)
host-test: $(HOST_TEST)
	$(HOST_TEST)
sim-test: $(SIM_TEST)
	$(HEXAGON_SIM) --simulated_returnval $(SIM_TEST)
sim-bench: $(SIM_BENCH)
	$(HEXAGON_SIM) --simulated_returnval $(SIM_BENCH)
sim-hmx-test: $(SIM_HMX_TEST)
	$(HEXAGON_SIM) --simulated_returnval $(SIM_HMX_TEST)
sim-hmx-pack-test: $(SIM_HMX_PACK_TEST)
	$(HEXAGON_SIM) --simulated_returnval $(SIM_HMX_PACK_TEST)
sim-crouton-test: $(SIM_CROUTON_TEST)
	$(HEXAGON_SIM) --simulated_returnval $(SIM_CROUTON_TEST)
sim-phase-bench: $(SIM_PHASE_BENCH)
	$(HEXAGON_SIM) --timing --simulated_returnval $(SIM_PHASE_BENCH)

$(HOST_TEST): tests/AttentionMathTest.cpp src/ops/AttentionMath.h
	@mkdir -p $(@D)
	$(HOST_CXX) -std=c++17 -O2 -Wall -Wextra -Werror -Isrc/ops $< -o $@

$(SIM_TEST): tests/HvxKernelTest.cpp src/ops/AttentionHvx.h
	@mkdir -p $(@D)
	$(HEXAGON_CXX) -std=c++17 -O3 -fno-vectorize -fno-slp-vectorize -stdlib=libc++ \
		-mhvx -mhvx-length=128B -mv79 -DUSE_OS_QURT \
		-I$(QNN_INCLUDE) -I$(QNN_INCLUDE)/HTP/core -I$(QURT_INCLUDE)/qurt -I$(QURT_INCLUDE)/posix \
		-I$(HEXAGON_SDK_ROOT)/incs -I$(HEXAGON_SDK_ROOT)/incs/stddef -Isrc/ops $< -o $@

$(SIM_BENCH): tests/HvxKernelBench.cpp src/ops/AttentionHvx.h
	@mkdir -p $(@D)
	$(HEXAGON_CXX) -std=c++17 -O3 -ffast-math -stdlib=libc++ -mhvx -mhvx-length=128B -mv79 -DUSE_OS_QURT \
		-I$(QNN_INCLUDE) -I$(QNN_INCLUDE)/HTP/core -I$(QURT_INCLUDE)/qurt -I$(QURT_INCLUDE)/posix \
		-I$(HEXAGON_SDK_ROOT)/incs -I$(HEXAGON_SDK_ROOT)/incs/stddef -Isrc/ops $< -lhexagon -o $@

$(SIM_HMX_TEST): tests/HmxTileTest.cpp
	@mkdir -p $(@D)
	$(HEXAGON_CXX) -std=c++17 -O3 -stdlib=libc++ -mhvx -mhvx-length=128B -mhmx -mv79 -DUSE_OS_QURT \
		-I$(QURT_INCLUDE)/qurt -I$(QURT_INCLUDE)/posix -I$(HEXAGON_SDK_ROOT)/incs \
		-I$(HEXAGON_SDK_ROOT)/incs/stddef $< -o $@

$(SIM_HMX_PACK_TEST): tests/HmxPackingTest.cpp src/ops/AttentionHmx.h
	@mkdir -p $(@D)
	$(HEXAGON_CXX) -std=c++17 -O3 -stdlib=libc++ -mhvx -mhvx-length=128B -mhmx -mv79 -DUSE_OS_QURT \
		-I$(QNN_INCLUDE) -I$(QNN_INCLUDE)/HTP/core -I$(QURT_INCLUDE)/qurt -I$(QURT_INCLUDE)/posix \
		-I$(HEXAGON_SDK_ROOT)/incs -I$(HEXAGON_SDK_ROOT)/incs/stddef -Isrc/ops $< -o $@

$(SIM_CROUTON_TEST): tests/CroutonReaderTest.cpp src/ops/AttentionCrouton.h
	@mkdir -p $(@D)
	$(HEXAGON_CXX) -std=c++17 -O3 -stdlib=libc++ -mhvx -mhvx-length=128B -mv79 -DUSE_OS_QURT \
		-I$(QNN_INCLUDE) -I$(QNN_INCLUDE)/HTP/core -I$(QURT_INCLUDE)/qurt -I$(QURT_INCLUDE)/posix \
		-I$(HEXAGON_SDK_ROOT)/incs -I$(HEXAGON_SDK_ROOT)/incs/stddef -Isrc/ops $< -o $@

$(OUTPUT): $(OBJECTS)
	@mkdir -p $(@D)
	$(HEXAGON_CXX) -mv79 -fPIC -stdlib=libc++ -shared -o $@ $^

build/hexagon-v79/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(HEXAGON_CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

AARCH64_CXX := $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++
AARCH64_SYSROOT := $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/darwin-x86_64/sysroot
AARCH64_FLAGS := --target=aarch64-none-linux-android29 --sysroot=$(AARCH64_SYSROOT)
AARCH64_FLAGS += -std=c++17 -fPIC -fvisibility=default -stdlib=libc++ -O3 -DANDROID -D__HVXDBL__
AARCH64_FLAGS += -Wall -Werror -Wno-format -Wno-missing-braces -Wno-unused-function -Wno-reorder
AARCH64_FLAGS += -Wno-unused-command-line-argument
AARCH64_FLAGS += -DQNN_API='__attribute__((visibility("default")))'
AARCH64_FLAGS += -D__QAIC_HEADER_EXPORT='__attribute__((visibility("default")))'
AARCH64_FLAGS += -DTHIS_PKG_NAME=KidiOps
AARCH64_FLAGS += -I$(QNN_INCLUDE) -I$(QNN_INCLUDE)/HTP/core
AARCH64_FLAGS += -I$(HEXAGON_SDK_ROOT)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION)/Tools/libnative/include

$(AARCH64_OUTPUT): $(patsubst src/%.cpp,build/aarch64-android/%.o,$(SOURCES))
	@mkdir -p $(@D)
	$(AARCH64_CXX) $(AARCH64_FLAGS) -shared -static-libstdc++ -o $@ $^ -L$(QNN_SDK_ROOT)/lib/aarch64-android -lQnnHtp -lQnnHtpPrepare

build/aarch64-android/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(AARCH64_CXX) $(AARCH64_FLAGS) -MMD -MP -c $< -o $@

-include $(OBJECTS:.o=.d)

clean:
	rm -rf build

$(SIM_PHASE_BENCH): tests/PrefillPhaseBench.cpp src/ops/AttentionHvx.h src/ops/AttentionMath.h
	@mkdir -p $(@D)
	$(HEXAGON_CXX) -std=c++17 -O3 -ffast-math -stdlib=libc++ -mhvx -mhvx-length=128B -mv79 -DUSE_OS_QURT \
		-I$(QNN_INCLUDE) -I$(QNN_INCLUDE)/HTP/core -I$(QURT_INCLUDE)/qurt -I$(QURT_INCLUDE)/posix \
		-I$(HEXAGON_SDK_ROOT)/incs -I$(HEXAGON_SDK_ROOT)/incs/stddef -Isrc/ops $< -lhexagon -o $@

SIM_FUSED_TEST := build/hexagon-v79/fused_ops_test.elf
.PHONY: sim-fused-test
sim-fused-test: $(SIM_FUSED_TEST)
	$(HEXAGON_SIM) --simulated_returnval $(SIM_FUSED_TEST)

$(SIM_FUSED_TEST): tests/FusedOpsTest.cpp src/ops/FusedOps.h
	@mkdir -p $(@D)
	$(HEXAGON_CXX) -std=c++17 -O3 -stdlib=libc++ -mhvx -mhvx-length=128B -mv79 -DUSE_OS_QURT \
		-I$(QNN_INCLUDE) -I$(QNN_INCLUDE)/HTP/core -I$(QURT_INCLUDE)/qurt -I$(QURT_INCLUDE)/posix \
		-I$(HEXAGON_SDK_ROOT)/incs -I$(HEXAGON_SDK_ROOT)/incs/stddef -Isrc/ops $< -o $@
