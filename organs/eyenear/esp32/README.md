# ESP32 media and accessory components

Reusable ESP32 media, USB, authenticated Wi-Fi and laptop tooling for Kidi.
The old W11 board target, pin map, QMI8658/STS35/battery drivers and factory
flash targets have been removed. Its test history is retained under
[history/w11/](history/w11/); private captures, backups and credentials remain
local and ignored.

**There is no deployable ESP32-P4/C6 firmware target yet.** The existing
Arduino/ESP32-S3 SDK build is only a compile check for retained adapters, with
no board pins configured. Do not upload that test image to the new board.

## Retained components

| Component | Purpose |
|---|---|
| [src/esp32.h](src/esp32.h), [src/esp32.cpp](src/esp32.cpp) | SDK/FreeRTOS aliases, explicit error reporting and chip-temperature adapter |
| [src/camera.h](src/camera.h), [src/camera.cpp](src/camera.cpp) | Backend-neutral JPEG frame/profile contract, ownership and capture arbitration |
| [src/parallel_camera.h](src/parallel_camera.h), [src/parallel_camera.cpp](src/parallel_camera.cpp) | Explicit legacy `esp-camera` parallel/JPEG adapter with validated pins |
| [src/audio.h](src/audio.h), [src/audio.cpp](src/audio.cpp) | Backend-neutral 16 kHz mono PCM16 contract and overrun statistics |
| [src/pdm_audio.h](src/pdm_audio.h), [src/pdm_audio.cpp](src/pdm_audio.cpp) | Explicit legacy I2S/PDM microphone adapter with validated pins |
| [src/clip.h](src/clip.h), [src/clip.cpp](src/clip.cpp) | Capture arbitration, configurable task placement and bounded image/video/audio/AV recording |
| [src/stream.h](src/stream.h), [src/stream.cpp](src/stream.cpp) | Paced capture, bounded latest-frame queue, typed packets, diagnostics and timing |
| [src/usb.h](src/usb.h), [src/usb.cpp](src/usb.cpp) | USB commands and media transport; no board-specific application entry point |
| [src/wireless.cpp](src/wireless.cpp) | Persistent ownership, pinned TLS, authenticated setup/media and transfer tests |
| [capture.py](capture.py), [wifi.py](wifi.py), [live.py](live.py) | USB capture, secure Wi-Fi client, browser preview and delivered-rate checks |
| [encode_video.py](encode_video.py), [speed.py](speed.py) | Laptop MP4 encoding and integrity-checked transport measurements |
| [tests/](tests/) | Host regressions and a pin-free SDK compile harness |
| [diagnostics/iperf/](diagnostics/iperf/) | Unmodified official Espressif throughput example with pinned provenance |

The parallel-camera, PDM, USB and Wi-Fi adapters remain framework-specific.
Their SDK types are confined to adapter headers/implementations; clip and stream
code use backend-neutral frames, profiles and audio statistics. A camera backend
must return complete JPEG frames with actual dimensions/timestamps and preserve
borrowed-frame ownership until return. Raw MIPI and H.264 require separate
P4 encoder/codec adapters. No pins are selected implicitly. Unconfigured media
endpoints fail explicitly instead of using old pins.

Media task affinity is also explicit. The reference defaults use no affinity;
a board application may select valid camera/audio cores, priorities and stack
sizes through `MediaExecutionConfig` after measuring its SDK/task layout.

## Proposed P4/C6 direction

- **Candidate board:** [Amazon ASIN B0FKH4KK7M](https://www.amazon.com/dp/B0FKH4KK7M),
  described as an ESP32-P4 plus ESP32-C6 module; verify the delivered revision
  against its manufacturer schematic.
- **ESP32-P4:** camera acquisition, ISP processing and hardware JPEG/H.264
  encoding as supported by the chosen SDK/profile.
- **ESP32-C6:** wireless co-processor, using the board's documented link and
  compatible ESP-Hosted/Wi-Fi Remote firmware.
- **Camera:** the regional listing for ASIN B012V1HEP4 identifies an Arducam
  OV5647 Raspberry Pi CSI module. Espressif lists an OV5647 sensor driver,
  but that does not verify the exact module's cable, power/reset sequence or
  compatibility with the selected board.

The exact board variant behind B0FKH4KK7M could not be independently confirmed;
no GPIO, C6 interconnect, codec or camera connector mapping is assumed.
Verify the manufacturer's schematic and supported camera configuration before
connecting it.

Unlike the previous JPEG-producing sensor, the OV5647 driver exposes raw
8/10-bit image data. The P4 needs a MIPI/ISP/encoder pipeline; injecting its
pins into the retained parallel-JPEG adapter is not a port. Sensor megapixels
also do not guarantee full-resolution ISP output: the current Espressif
camera documentation limits the P4 ISP path to 1920x1080.

The retained live protocol supports JPEG frames and PCM audio, not H.264.
H.264 requires explicit codec metadata, packet/protocol negotiation and client
decoding; do not send H.264 bytes in JPEG packets. Android/BLE integration
remains future work.

Sources: [Espressif camera sensors](https://github.com/espressif/esp-video-components/tree/master/esp_cam_sensor),
[ESP-Hosted](https://github.com/espressif/esp-hosted-mcu),
[candidate board listing](https://www.amazon.com/dp/B0FKH4KK7M),
[camera listing](https://www.amazon.ca/Arducam-Megapixels-Sensor-OV5647-Raspberry/dp/B012V1HEP4).

## Validation and usage

```bash
make test
make compile-check
make format-check
```

`make` defaults to help. `make build` is an alias for the compile check, not a
P4 firmware build. No flash/restore target remains.

Host commands require an independently installed, configured firmware that
implements the Kidi protocol. See [USAGE.md](USAGE.md) and [DESIGN.md](DESIGN.md).

Normal setup/capture remains strictly cache-only for Wi-Fi passwords.
`.kidi.wifi.txt` and `.private/` are owner-only and ignored. Only explicit
`make cache-wifi` authorizes new OS credential retrieval. A new accessory gets
its own identity; do not reuse the returned module's device credential.
