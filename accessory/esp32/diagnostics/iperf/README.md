# Official Espressif Wi-Fi Benchmark

An **unmodified** copy of ESP-IDF's Wi-Fi `iperf` example for an independent
TCP/UDP baseline. It is separate from Kidi's camera/audio firmware. The official
example was built and hardware-tested on the previous W11 ESP32-S3 on 2026-10-05, then
the complete Kidi flash and private configuration were restored and verified.

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

The build wrapper requires an explicit chip target; it no longer assumes the
old module's 16 MB flash layout or USB console. From this directory:

```bash
uv run --no-project python build.py --target esp32c6 \
  --sdk /path/to/esp-idf-v5.5.1 --tools /path/to/idf-tools \
  --defaults /path/to/your-board-sdkconfig.defaults
```

Use the actual board's target/defaults. Builds are isolated under
`.build/<target>/`; the pinned upstream example and target-specific defaults
remain unchanged. The command builds only and never flashes.
The historical run was ESP32-S3, not C6; this example command does not claim
C6 hardware validation. P4 with a C6 co-processor additionally needs a compatible
hosted Wi-Fi configuration; this wrapper does not implement that integration.

The local isolated SDK/toolchain was installed and the previous configuration built
successfully. SDK, toolchain, generated build files, and backups remain private
local artifacts, not distributed with this source.

## Hardware Results (2026-10-05)

The module joined the already-approved cached LAN. No Keychain, OS credential
retrieval, or password prompt was used. Tests had no camera/audio or TLS load.
TCP used three ten-second trials per direction, reporting receiver throughput:

| Direction | Trials (Mbit/s) | Median (Mbit/s) |
|---|---:|---:|
| Module to laptop | 20.00, 19.20, 20.20 | 20.00 |
| Laptop to module | 14.07, 16.74, 16.67 | 16.67 |

Rates above use decimal Mbit/s. The ESP receiver labels its binary
`bytes * 8 / 1024 / 1024` rates as `Mbits/sec`; those reports were converted
to decimal units. Laptop reports already use decimal units.

UDP used one ten-second trial at each offered rate:

| Offered (Mbit/s) | Module to laptop received | Laptop to module received |
|---|---:|---:|
| 10 | 9.76 Mbit/s | 10.48 Mbit/s |
| 20 | 19.08 Mbit/s | 20.89 Mbit/s |
| 30 | 23.44 Mbit/s | 28.82 Mbit/s |

These are averages of the first five two-second receiver intervals, not the
laptop UDP server's shutdown average, which includes idle time after the module
stops transmitting. Timing/pacing differences can put short receiver-window
rates above the nominal offered rate. No validated UDP packet-loss figure is
claimed; these rates do not establish loss-free real-time media delivery.

Initial ping received all ten replies: 120.549 ms average, with a 1025.917 ms
first-reply outlier; the other nine replies were 10.720-36.707 ms.

This establishes substantially higher raw throughput than the earlier Kidi
HTTPS tests, but does not isolate TLS, TCP configuration, modem sleep, SDK
version, or changed radio conditions as the cause. It does not establish
20 Mbit/s of sustainable encrypted camera/audio throughput.

The complete 16 MB restoration passed a flash digest check. Trusted USB
certificate verification refreshed the changed DHCP address; authenticated
HTTPS status and a 2048x1536 photo then passed. The restored application
reported -51 dBm RSSI. Raw measurements and full-flash backups are local,
not committed.

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
5. Compare TCP transmit/receive, UDP receiver throughput, and ping latency using
   the same physical placement and network. Validate UDP loss-report
   compatibility separately before claiming packet-loss percentages.
6. Restore the complete pre-test flash and verify normal Kidi operation.

Fetching this source alone does not authorize flashing. The 2026-10-05 run
used explicit user approval and a session-local runner with full restoration.
Keep the board in bootloader between final backup verification and diagnostic
flashing: booting Kidi can change NVS and invalidate a byte-for-byte comparison.

## Benchmark-Only Configuration

The upstream defaults disable watchdogs and include experimental PHY settings.
They are not production recommendations. Keep this benchmark standalone rather
than copying its entire configuration into the capture application.
