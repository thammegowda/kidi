# ESP32 media and accessory components

Reusable ESP32 media, USB, authenticated Wi-Fi and laptop tooling for Kidi.
The old W11 board target, pin map, QMI8658/STS35/battery drivers and factory
flash targets have been removed. Its test history is retained under
[history/w11/](history/w11/); private captures, backups and credentials remain
local and ignored.

**There is no deployable ESP32-P4/C6 product firmware yet.** Buildable,
non-flashable Network Split/deep-sleep bring-up roots now live under
[firmware/](firmware/). The Arduino/ESP32-S3 SDK build remains only a compile
check for retained adapters, with no board pins configured. Do not upload any
bring-up or compile-check image to the board.

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
| [firmware/components/kidi_protocol/accessory_protocol.h](firmware/components/kidi_protocol/accessory_protocol.h), [protocol/accessory-v1.json](protocol/accessory-v1.json) | Shared C++ constants and machine-readable Android/C6/P4 protocol contract |
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

## Selected P4/C6 accessory

The diagnosed hardware is a Waveshare ESP32-P4-Module-DEV-KIT with ESP32-P4
revision 1.3, its onboard ESP32-C6, an Arducam OV5647 MIPI-CSI module and the
board's onboard ES8311 microphone path. The manufacturer schematic confirms
C6 GPIO2 to P4 GPIO6 for host wake and P4 GPIO54 to C6 `CHIP_PU`.

The selected version-1 design is documented in [DESIGN.md](DESIGN.md):

- C6 remains available for BLE pairing, the accessory SoftAP, authenticated
  control, a maximum of four controllers and P4 wake arbitration.
- P4's low-power side supervises wake/idle transitions; its high-power cores
  initialize the MIPI/ISP/JPEG or ES8311 path only for a media request.
- C6 owns TLS control port 61443 and P4 owns TLS media port 54443 using
  ESP-Hosted Network Split. Only one photo or audio session may run at a time.
- Android pairs from a one-time `kidi://pair/v1` invitation and uses the
  existing photo and dictation controls with Phone/Accessory source selectors.

Kidi-owned P4/C6 product sources use GNU C++23 with colocated headers and
implementations, exceptions/RTTI disabled and warnings treated as errors. The
only C sources under the product tree are the unchanged vendored Espressif
video helper and ESP-IDF internals.

OV5647 supplies RAW8/RAW10 rather than sensor-produced JPEG, so the P4 backend
must explicitly own MIPI capture, ISP and hardware JPEG encoding. Version 1
transports bounded JPEG photos and 16 kHz mono PCM16 audio. H.264 is not part of
this protocol and must not be mislabeled as JPEG if added later.

Sources: [Espressif camera sensors](https://github.com/espressif/esp-video-components/tree/master/esp_cam_sensor),
[ESP-Hosted](https://github.com/espressif/esp-hosted-mcu),
[board listing](https://www.amazon.com/dp/B0FKH4KK7M),
[camera listing](https://www.amazon.ca/Arducam-Megapixels-Sensor-OV5647-Raspberry/dp/B012V1HEP4).

## Validation and usage

```bash
make test
make compile-check
make format-check
make firmware-build  # requires an exported ESP-IDF 5.5.x environment
```

`make` defaults to help. `make build` is an alias for the compile check, not a
P4 firmware build. `make c6-build`, `make p4-build` and `make firmware-build`
only compile; no flash/restore target exists. See [firmware/README.md](firmware/README.md)
for the C6 recovery prerequisite and current bring-up limitations.

Host commands require an independently installed, configured firmware that
implements the Kidi protocol. See [USAGE.md](USAGE.md) and [DESIGN.md](DESIGN.md).

Normal setup/capture remains strictly cache-only for Wi-Fi passwords.
`.kidi.wifi.txt` and `.private/` are owner-only and ignored. Only explicit
`make cache-wifi` authorizes new OS credential retrieval. A new accessory gets
its own identity; do not reuse the returned module's device credential.

## P4/C6 hardware diagnosis (2026-10-06)

The candidate Waveshare board and Arducam OV5647 from the links above were
tested before implementing a Kidi P4 backend:

- ESP32-P4 revision 1.3, running at 360 MHz with the required pre-v3 software
  profile.
- 32 MB physical flash detected. The vendor factory image header exposes
  16 MB; do not infer the physical size from that header.
- 32 MB PSRAM detected at 200 MHz; the factory boot memory test passed.
- Complete 32 MB factory flash saved privately and verified byte-for-byte.
  Secure Boot and flash encryption are disabled; no eFuses were modified.
- ESP32-C6 SDIO transport, association, DHCP, bidirectional HTTP traffic and
  sustained MJPEG transfer passed. The newer diagnostic host reports
  `host=3.0.9`, while the vendor C6 image reports `0.0.0`, then uses compatible
  streaming mode. Do not update one side independently without a matched pair.
- Raw integrity-checked 4 MiB HTTP payload medians: **28.92 Mbit/s module to
  laptop**, **25.32 Mbit/s laptop to module**. The first download was a
  5.50 Mbit/s warm-up outlier. Idle ping had no loss but variable latency.
- OV5647 PID `0x5647` detected. Multiple JPEGs fully decoded and were distinct.
  Initial 800x800 RAW8 streaming delivered 954 distinct frames in 30.09 seconds:
  **31.71 fps** and **10.70 Mbit/s** JPEG payload.
- 1280x960 RAW10 binning improved exposure but produced much larger JPEGs:
  43 distinct frames in 10.86 seconds, **3.96 fps** and **9.29 Mbit/s**.
- The stock OV5647 ISP/IPA output has an unacceptable warm/green cast.
  Waveshare's 800x1280 RAW8 profile was darker. An intentional BGGR metadata
  trial produced blue/purple Bayer artifacts, confirming the driver's GBRG
  phase is the correct of those two; remaining work is ISP/AE/AWB/CCM tuning,
  not cable or MIPI-link bring-up.
- The onboard ES8311 ADC captured exactly five seconds of 16 kHz stereo PCM16.
  Channel 0 was active (range -427..511, AC RMS 71.9, no clipping); channel 1
  was silent under the mono mapping. Speech/claps were recognizable after
  54.5x host normalization, so the microphone works but current gain is low.

Current lab state: the P4 is intentionally running the revision-1.3
1280x960 RAW10 camera/HTTP benchmark diagnostic; the C6 firmware was never
flashed. Factory firmware has already been restored and fully verified during
earlier guarded runs, but was not restored after the final user-requested
diagnostic installation.

Private stills, MP4s, WAVs, full-flash backup, build trees and raw results stay
under ignored/session-local paths. The active Kidi source still has no
deployable P4/C6 target.
