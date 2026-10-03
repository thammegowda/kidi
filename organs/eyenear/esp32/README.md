# W11 ESP32 camera and microphone

USB diagnostics for the Meshnology W11 ESP32-S3 mainboard and its OV3660
camera/PDM microphone expansion board. This is a standalone PlatformIO project,
not yet connected to Kidi's Android app.

## Contents

- [src/main.cpp](src/main.cpp): firmware for photos, microphone recordings,
  silent video, and simultaneous audio/video acquisition.
- [src/esp32.h](src/esp32.h): readable C++ aliases
  for ESP32 SDK camera, I2S, error, and FreeRTOS handle types; no board pin mapping
  or application entry points.
- [src/w11.h](src/w11.h): the W11-specific pin mapping selected by this application.
- [src/sensors.cpp](src/sensors.cpp): motion, temperature, and battery-rail readings.
- [platformio.ini](platformio.ini): pinned ESP32 Arduino build configuration.
- [Makefile](Makefile): common build, flash, capture, backup, and validation goals.
- [capture.py](capture.py): USB capture client with binary framing and timing checks.
- [encode_video.py](encode_video.py): local H.264/AAC MP4 encoding and decode validation.
- [USAGE.md](USAGE.md): build, capture, application update, and factory restoration commands.
- [JOURNAL.md](JOURNAL.md): hardware diagnosis, measured results, and remaining limits.
- `captures/`: local photos, WAVs, MP4s, original video frames, and timing manifests.
- `backups/`: the complete verified factory flash backup.

Captures, backups, and PlatformIO build output are intentionally ignored by Git.
The flash backup can include private configuration, and recordings contain
camera/microphone observations. They remain local; do not publish them without
review and explicit consent.

## Quick start

All Python tools use `uv run`; the Makefile has a single `UV` executable setting.

Run from this directory with the diagnostic already installed:

```bash
make status
make temperature
make sensors
make av SECONDS=10 CAPTURE=captures/new-av-test
```

The default serial port is `/dev/cu.usbmodem1101`; override it with `PORT=...`.
Close other serial monitors first. Durations are limited to 1-10 seconds, and
capture/output paths must not already exist.

Build and flash:

```bash
make build                 # Compile without changing the board; also the default goal.
make flash                 # Build, then update only the application.
make photo                 # Capture the OV3660's full 2048x1536 resolution.
make photo RESOLUTION=640x480 # Capture a smaller image.
make audio SECONDS=5       # Save 48 kHz mono PCM16 WAV files.
make video SECONDS=10      # Capture and encode silent 720p MP4.
make help                  # List goals and configurable variables.
```

See [USAGE.md](USAGE.md) before flashing. Its application-only update command
assumes the board already has the working diagnostic bootloader and partition
table. The verified factory backup is local to this checkout; it is not
distributed with the source.

`make backup` saves and verifies the board's **current** full flash; it does not
recreate the original factory firmware. `make restore CONFIRM_RESTORE=yes`
overwrites the installed firmware using the saved factory backup. Neither
operation accesses the SD card. `make clean` removes build output only.

## Development

```bash
make format-check test
```

The firmware follows Kidi's namespaces, snake_case functions and variables,
SCREAMING_UPPER_CASE constants and enum values, trailing return types, and
repository clang-format conventions. Arduino's required global `setup`/`loop`
entry points delegate to `kidi::esp32`. It uses C++17 rather than Kidi's native
C++23 because the tested ESP32 toolchain uses GCC 8.4. GNU extensions are
required by the bundled FreeRTOS headers.
SDK aliases live in `kidi::esp32` and preserve the original types and
ABI; handle aliases do not add ownership or automatic cleanup.
Headers are colocated with the source. The `kidi::esp32` namespace is chip-level,
not specific to eyes/ears. This application selects `W11Pins`; future boards and
peripherals can supply their own pin layouts, build profiles, and applications
without making camera or microphone support mandatory in the generic aliases.
The Makefile selects `PIO_ENV=w11` and `CHIP=esp32s3` by default; these are
overridable when adding corresponding build profiles and application support.

Host protocol tests require no connected board. Firmware validation uses the
ESP32 cross-compiler through PlatformIO; no CMake project or native Kidi build
is needed.

## Scope

The tested board has 16 MB flash and 8 MB octal PSRAM. The driver acquires
2048x1536 JPEG still photos by default, with 640x480 available on request,
and 48 kHz mono PCM16 audio. Video defaults to 1280x720 (720p); MP4 audio is AAC
at a target 128 kbit/s. The original unprocessed WAV is preserved. The ESP32-S3's camera peripheral
and a separate microphone task operate concurrently; audio is buffered in PSRAM
and transferred over USB after video acquisition.

MP4 encoding happens on the Mac, not on the ESP32. No SD-card access, Wi-Fi
streaming, Bluetooth audio, Android integration, or long-duration streaming is
implemented. The RGB LED shares GPIO48 with the camera and is not driven.

## Sensor Snapshots

`make sensors` returns a `KIDI_SENSORS` JSON record with a schema version,
board-clock timestamp in microseconds, and readings with explicit units:

- ESP32-S3 die temperature in Celsius.
- QMI8658 acceleration in g, angular velocity in degrees/second, and IMU temperature.
- STS35 board temperature when that device is detected and its checksum passes.
- VBAT-rail voltage from GPIO2 and the board's 2:1 divider. This does not establish
  whether a battery is connected or report state of charge.

Unavailable devices have `available: false` and an explicit `error`; the host
also reports them on stderr. The sensor I2C controller is separate from the
camera's SCCB controller. IMU measurement channels are disabled after the
snapshot. This is an on-demand snapshot, not yet a synchronized sensor stream
alongside video.

`make temperature` reads the die sensor without camera/microphone capture.
Its selected range is 20-100 C with a specified error below 2 C. It measures chip
temperature, not ambient temperature.

The CPU currently remains at its configured clock while idle; `delay()` yields
to FreeRTOS but does not put the chip into deep sleep. Camera driver shutdown
also does not physically disconnect the powered camera module. Warmth alone
does not identify a fault. Stop captures and disconnect USB if the board becomes
uncomfortably hot, smells unusual, or shows discoloration.
