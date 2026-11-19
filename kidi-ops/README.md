# KidiOps

KidiOps houses accelerator implementations. Kidi core owns the tensor, operator,
and graph compiler interfaces; YNNPACK owns CPU kernels.

## CUDA

The CUDA module owns kernels, cuBLAS dispatch, storage, and graph replay. It is
built only with `KIDI_ENABLE_CUDA=ON` and requires CUDA 13 or later. Run
`make release-cuda` from the Kidi root. The optional `kidi-index-cuda` executable
keeps cuVS and DLPack dependencies separate from inference.

## Qualcomm QNN HTP

The standalone Qualcomm QNN HTP custom-operation package
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
NDK, Qualcomm AI Runtime SDK, and Hexagon SDK described by the variables at the
top of the `Makefile`:

```sh
make -C kidi-ops aarch64
make -C kidi-ops
```

The package currently exposes only the original general operations. Experimental
model-specific nodes and hardware probes are not part of this directory.
