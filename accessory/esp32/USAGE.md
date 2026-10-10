# ESP32 component and client usage

These are retained development components, not installed firmware for the
proposed P4/C6 board. The W11 target and recovery commands were removed.
Archived instructions and measurements are under [history/w11/](history/w11/).

## Validate without hardware

Run from this directory:

```bash
make test
make compile-check
make format-check
make help
```

The PlatformIO environment `esp32s3_compile_check` compiles the current
Arduino/SDK adapters and a test entry point with no camera/microphone pins.
It does not validate P4 compatibility and must not be uploaded. Its configuration
guards are compiled, not hardware-executed by `make compile-check`.

There is no active production board entry point, pin map or flash layout.
`make flash`, `make restore`, `make sensors` and the W11 factory-backup defaults
are no longer available. Do not copy archived flash commands to another chip.

## Board integration contracts

- [install_camera_backend](src/camera.h) installs one backend-neutral JPEG
  camera contract. [install_parallel_camera](src/parallel_camera.h) is the
  retained `esp-camera` adapter; required pins must be valid and distinct,
  while optional reset/power-down pins may be `-1`. Backend context and
  callbacks must remain valid for the firmware lifetime.
- [install_audio_backend](src/audio.h) installs one 16 kHz mono PCM16 backend
  with explicit overrun/DMA-error statistics. Its context and callbacks have
  the same firmware-lifetime requirement.
  [install_pdm_microphone](src/pdm_audio.h) takes explicit clock/data pins and
  an I2S port; it is not an I2S-codec driver.
- [initialize_media](src/clip.h) creates shared capture arbitration and accepts
  explicit task affinity/priorities/stacks. Defaults are unpinned; a board
  application should set cores only after reviewing its SDK task placement.
  [initialize_usb](src/usb.h) initializes media plus the current USB/Wi-Fi
  service; it does not configure a camera or microphone. Do not initialize
  media separately before calling it.
- [acquire_camera_frame](src/camera.h) and
  [return_camera_frame](src/camera.h) form the camera-driver boundary.
  Frames must be complete JPEGs with matching profile dimensions and remain
  borrowed until returned. Writers must consume/copy bytes before returning
  success; an asynchronous DMA transport needs completion/ownership adaptation.

The parallel-camera/PDM adapters, temperature driver, USB and Wi-Fi adapters are
reference implementations. Clip and stream layers no longer expose
`camera_fb_t`, `framesize_t` or I2S event types. A P4 application must add the
appropriate MIPI/ISP/JPEG-or-H.264 backend and board-specific C6 hosted link.
An onboard audio codec requires another `AudioBackend`. Missing backends produce
explicit errors; HTTPS media handlers report HTTP 501 before acquisition.

## Host tools with compatible firmware

All commands use one `UV` setting and `uv run`. No normal command requests
Keychain access, OS Wi-Fi discovery or a password for an already configured
accessory. Only explicit credential caching authorizes OS access.

```bash
make cache-wifi WIFI_SSID=your-network # Explicit one-time credential authorization.
make dev-connect                     # Trusted USB identification/setup; reuse cached network.
make status
make temperature                     # Generic chip-temperature adapter, not board sensors.
make photo TRANSPORT=wifi
make audio TRANSPORT=usb SECONDS=5
make video TRANSPORT=usb
make video-check TRANSPORT=wifi SECONDS=20
make av TRANSPORT=usb                 # Explicit experimental microphone + camera mode.
```

These commands are protocol tools, not proof that the new board supports a
given media profile. Wi-Fi authorization/TLS failures and interrupted captures
never silently retry over USB. `TRANSPORT=wifi` requires wireless;
`TRANSPORT=usb` forces USB. Automatic fallback is permitted only before capture
when Wi-Fi is unavailable.

Native Espressif USB serial products are discovered by vendor ID rather than
a single product ID. With both P4 and C6 ports visible, auto-selection refuses
ambiguity; use `PORT=...` or `--port`. USB-UART bridges require explicit port
selection rather than being assumed to belong to the accessory.

Reference protocol profiles:

| Media | Current client/adapter contract |
|---|---|
| Photo | JPEG, 2048x1536 default or 640x480 |
| Video | JPEG frames, 1280x720 default or 96x96; nominal 10 fps |
| Audio | 16 kHz mono little-endian PCM16; legacy live 48 kHz parsing retained |
| Saved clip | 1-10 seconds |
| Live session | 1-300 seconds; Start/Stop and duration expiry |

The future backend must explicitly support/negotiate appropriate profiles.
The retained photo default is not a promise of P4 ISP resolution. H.264 is
not implemented in this protocol/viewer.

`make video` never initializes microphone acquisition or browser AudioContext.
`video-check` requires at least 9.5 delivered fps (5% tolerance around the
nominal 10 fps target) and reports bytes, receive gaps and device timings.
It divides by the greater of requested duration and actual receiving time.
Old W11 wireless video never met that threshold; the new hardware has not
been tested.

## Saved media and privacy

[capture.py](capture.py) retains USB JPEG/WAV/recorded-video parsing.
[encode_video.py](encode_video.py) converts an existing timestamped capture
directory to H.264/AAC MP4 on the laptop:

```bash
make encode CAPTURE=captures/existing-session OUTPUT=captures/existing-session.mp4
```

Playback frame rate may repeat images; it is not actual acquisition FPS.
The live browser viewer saves no media.

Credentials, captures, backups, SDK/toolchains and build output remain ignored.
`.kidi.wifi.txt` is plaintext JSON containing approved network passwords;
`.private/` holds per-device ownership records. Do not publish either or full
flash backups. Existing local artifacts were preserved during cleanup.
Use `make forget-wifi-cache` only if you intentionally want to remove the
network cache; the next credential retrieval then requires authorization.
