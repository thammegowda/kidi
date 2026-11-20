# KidiOps

KidiOps houses accelerator implementations. Put code for `kidi::<pkg>` in
`src/<pkg>/`, with headers and implementations side by side. Mirror the package
layout in `tests/`. Benchmark sources live with the backend they measure.

| Module | Host adapter | Device code and tools |
|---|---|---|
| [`src/cuda`](src/cuda) | Storage, prepared operators, and graph replay | CUDA runtime and kernels, projection benchmark, and optional cuVS helper |
| [`src/qnn`](src/qnn) | QNN captured-step compiler and runtime loading | Standalone Qualcomm HTP op package and config under [`htp/`](src/qnn/htp) |

Tests live in [`tests/cuda`](tests/cuda) and [`tests/qnn`](tests/qnn), with
SDK-independent math tests and simulator tests under [`tests/qnn/htp`](tests/qnn/htp).

For a new accelerator, gate its subdirectory on its enable option and use
`kidi_ops_add_backend` to attach its host adapter to Kidi. Keep SDK discovery
and SDK-specific types inside that module. Reuse Kidi core's
[`tensor::Backend`](../src/kidi/tensor/backend.h) for device storage,
[`runtime::OperatorBackend`](../src/kidi/runtime/operator.h) for eager ops, and
[`graph::Compiler`](../src/kidi/graph/compiler.h) for captured steps. An attached
accelerator such as QNN needs only the captured-step interface. YNNPACK owns
CPU kernels.

## CUDA

The CUDA module owns native CUDA/CUTLASS kernels, static runtime dispatch,
storage, and graph replay. It is built only with `KIDI_ENABLE_CUDA=ON` and
requires CUDA 13 or later. Run `make release-cuda` from the Kidi root. The
optional `kidi-index-cuda` executable keeps cuVS and DLPack dependencies
separate from inference.

## Qualcomm QNN

The host compiler is built when `KIDI_ENABLE_QNN=ON` and `KIDI_QNN_SDK` points
to a QAIRT SDK. It loads the vendor runtime at execution time. Without the SDK,
Kidi builds with its existing unavailable-backend stub.

The standalone HTP custom-operation package
contains the existing structured-attention, GELU-multiply, and RMS-normalization
kernels together with their SDK-independent math tests and Hexagon simulator
tests.

The package is kept outside the main runtime source tree because it has a
different toolchain and deployment boundary. The host-side test does not need
QAIRT:

```sh
cmake -S kidi-ops -B kidi-ops/build/cmake
cmake --build kidi-ops/build/cmake
ctest --test-dir kidi-ops/build/cmake --output-on-failure
```

Building the Android host library or V79 execution library requires the Android
NDK, Qualcomm AI Runtime SDK, and Hexagon SDK described by the variables in
the [HTP Makefile](src/qnn/htp/Makefile). The original entry points and
artifact paths are unchanged:

```sh
make -C kidi-ops aarch64
make -C kidi-ops
```

The package currently exposes only the original general operations. Experimental
model-specific nodes and hardware probes are not part of this directory.
