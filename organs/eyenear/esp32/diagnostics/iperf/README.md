# Official Espressif Wi-Fi Benchmark

An **unmodified** copy of ESP-IDF's Wi-Fi `iperf` example for an independent
TCP/UDP baseline. It is separate from Kidi's camera/audio firmware and has not
been flashed or hardware-tested here.

## Provenance

- Repository: [espressif/esp-idf](https://github.com/espressif/esp-idf).
- Release: **v5.5.1**.
- Commit: `fcae32885b0296b32044cb99ecbdc50d98dddb83`.
- Example: [examples/wifi/iperf](https://github.com/espressif/esp-idf/tree/fcae32885b0296b32044cb99ecbdc50d98dddb83/examples/wifi/iperf).
- Local source: [upstream/](upstream/).
- Checksums and original paths: [source-manifest.json](source-manifest.json).

Each downloaded file was verified against its Git blob hash. The example
identifies itself as public-domain/CC0 code; the included [SDK license](upstream/LICENSE)
and component-specific licenses remain applicable to other SDK dependencies.
Do not rewrite the upstream source to match application formatting.

## Why This Comparison Matters

Kidi's current `make speed-test` measures verified **HTTPS application
throughput**. This official example measures plain TCP/UDP with no TLS, camera,
or microphone workload.

The official example:

- Supports **ESP32-S3**.
- Explicitly disables modem sleep with `esp_wifi_set_ps(WIFI_PS_NONE)`.
- Uses 65535-byte TCP send/receive buffers; Kidi's current prebuilt Arduino
  SDK has 5760-byte buffers.
- Enables throughput-oriented Wi-Fi aggregation/buffer settings and uses
  high-priority traffic/report tasks.
- Requires laptop **iperf 2.x**; `iperf3` is not compatible.

Use it to distinguish radio/network capability from Kidi's TLS, framing,
buffering, and capture-concurrency costs. Do not assume the SDK's lab figures
will be reproduced in this room or on this board.

## Build Prerequisites

Use an installed/activated **ESP-IDF v5.5.1** environment. The example's
component-manager dependencies include Espressif's iperf command, Wi-Fi command,
and ping command components; the SDK also supplies its console system commands.

The upstream CMake files are part of the official example, not a migration of
Kidi's main PlatformIO/Arduino project.

From this directory:

```bash
cd upstream
idf.py -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.defaults.esp32s3;../usb.defaults" \
  set-target esp32s3
idf.py build
```

[usb.defaults](usb.defaults) selects the W11's native USB Serial/JTAG console
and 16 MB flash. It does not modify the upstream example.

These build prerequisites are not currently installed, so no independent
benchmark binary or raw TCP/UDP result has been claimed.

## Flashing and Recovery

**Do not flash this diagnostic over the application without a verified current
backup and explicit approval.**

The official app uses a different SDK and partition layout. It can erase NVS
if initialization detects an incompatible version or insufficient pages.
A saved factory backup alone does not preserve the subsequently provisioned
Kidi identity and Wi-Fi configuration.

Before a diagnostic run:

1. Save and verify the complete **current** flash using the existing backup
   workflow. The backup contains private configuration; keep it Git-ignored.
2. Preserve the laptop's private identity record and `.kidi.wifi.txt`.
3. Build/flash the complete diagnostic image with its own bootloader/partitions.
4. Provision from the already-approved Wi-Fi cache over USB, without Keychain.
   The upstream interactive console can echo commands: credential-bearing
   commands must not be printed, logged, or placed in shell arguments.
5. Compare TCP transmit/receive, UDP throughput/loss, and ping latency using
   the same physical placement and network.
6. Restore the complete pre-test flash and verify normal Kidi operation.

Flashing or automating the official console has not been authorized or
implemented by fetching this source.

## Benchmark-Only Configuration

The upstream defaults disable watchdogs and include experimental PHY settings.
They are not production recommendations. Keep this benchmark standalone rather
than copying its entire configuration into the capture application.
