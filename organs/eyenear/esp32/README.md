# W11 ESP32 camera and microphone

Unified USB diagnostics, authenticated Wi-Fi photo/audio capture, and an
experimental live preview for the Meshnology
W11 ESP32-S3 and its OV3660 camera/PDM microphone expansion board. This is a
standalone PlatformIO project, not yet connected to Kidi's Android app.

## Contents

- [src/main.cpp](src/main.cpp): firmware for photos, microphone recordings,
  silent video, and simultaneous audio/video acquisition.
- [src/camera.cpp](src/camera.cpp): shared camera acquisition and resource
  arbitration for USB and Wi-Fi.
- [src/audio.cpp](src/audio.cpp): shared 16 kHz speech microphone acquisition.
- [src/stream.cpp](src/stream.cpp): independent camera/audio workers, bounded
  queues, latest-frame delivery, and explicit overrun reporting.
- [src/wireless.cpp](src/wireless.cpp): trusted USB setup, persistent device
  identity, and authenticated HTTPS photo/status endpoints.
- [src/esp32.h](src/esp32.h): readable C++ aliases
  for ESP32 SDK camera, I2S, error, and FreeRTOS handle types; no board pin mapping
  or application entry points.
- [src/w11.h](src/w11.h): the W11-specific pin mapping selected by this application.
- [src/sensors.cpp](src/sensors.cpp): motion, temperature, and battery-rail readings.
- [platformio.ini](platformio.ini): pinned ESP32 Arduino build configuration.
- [Makefile](Makefile): common build, flash, capture, backup, and validation goals.
- [capture.py](capture.py): USB capture client with binary framing and timing checks.
- [wifi.py](wifi.py): automated laptop setup and Wi-Fi-first photo client.
- [live.py](live.py): local browser viewer and headless live-protocol checks.
- [speed.py](speed.py): authenticated, integrity-checked module/laptop throughput
  benchmark, with TLS setup timed separately.
- [diagnostics/iperf/](diagnostics/iperf/): pinned, unmodified official Espressif
  plain TCP/UDP example for an independent Wi-Fi baseline; not flashed yet.
- [tools/live.html](tools/live.html): live image/audio playback; no recordings saved.
- [tools/macos_wifi.swift](tools/macos_wifi.swift): permission-aware current-network
  discovery and single-credential access with an explicit scope prompt.
- [encode_video.py](encode_video.py): local H.264/AAC MP4 encoding and decode validation.
- [USAGE.md](USAGE.md): build, capture, application update, and factory restoration commands.
- [JOURNAL.md](JOURNAL.md): hardware diagnosis, measured results, and remaining limits.
- [DESIGN.md](DESIGN.md): draft wireless architecture and laptop-first prototype
  plan; not implemented.
- `captures/`: local photos, WAVs, MP4s, original video frames, and timing manifests.
- `backups/`: the complete verified factory flash backup.
- `.private/`: local owner credentials and the generated macOS helper; Git-ignored.
- `.kidi.wifi.txt`: selected-network Wi-Fi password cache, owner-only and Git-ignored.

Captures, backups, and PlatformIO build output are intentionally ignored by Git.
The flash backup can include private configuration, and recordings contain
camera/microphone observations. They remain local; do not publish them without
review and explicit consent.

## Quick start

All Python tools use `uv run`; the Makefile has a single `UV` executable setting.

Run from this directory:

```bash
make flash
make dev-connect
make photo
make speed-test
```

`dev-connect` detects the accessory, reuses its configured network or the
private Wi-Fi cache, configures the board over USB when needed, and verifies its
authenticated HTTPS endpoint. The laptop stays on its current Wi-Fi; the module
joins that LAN as a 2.4 GHz station. No module hotspot or Ethernet is required.

Normal setup/capture commands never request Keychain access or a password.
A missing cache produces an actionable error, not an implicit OS credential
lookup. An already-configured device is reused without Wi-Fi-name discovery.
With multiple cached networks, explicitly select `WIFI_SSID=...`.

Only the explicit one-time `make cache-wifi` operation authorizes OS credential
retrieval. On macOS, that operation may request Location Services permission to read the
Wi-Fi name. Credential access uses a named dialog showing the exact network and
logged-in Mac account, then requests only that saved Wi-Fi password. Allow
local-network access for the app/terminal running the client when macOS asks.
On Windows, plaintext saved-key access is subject to native Wi-Fi profile
permissions; managed/enterprise profiles are not automatically copied.

Authorized credentials are cached in `.kidi.wifi.txt` for the selected SSID.
Subsequent setup on that SSID uses the cache instead of Keychain. After a
firmware reboot, setup first waits for the accessory's already-saved profile
to reconnect rather than asking for its password again.

```bash
make cache-wifi             # One-time selected-network authorization and caching.
make forget-wifi-cache      # Delete laptop Wi-Fi password cache, not device identity.
```

If macOS does not authorize current-network name discovery, explicitly select
the intended network once with `make cache-wifi WIFI_SSID=your-network` or
`make dev-connect WIFI_SSID=your-network`. This does not bypass password
authorization and does not collect other profiles.

The cache is JSON text containing plaintext Wi-Fi passwords; never share it.
It is mode 0600 on macOS/POSIX, with a current-user-only ACL on Windows.
Deleting it means the next required credential retrieval needs authorization.

During the explicit cache operation only, if saved-key access is unavailable,
an interactive terminal can use a local
private password prompt. Explicit cancellation/denial of the named credential
request stops setup; it does not trigger another password prompt. Credentials
are never printed, put in command-line arguments, or committed to source.

The serial port is automatically detected when one compatible ESP32 is present;
override it with `PORT=...`. Close serial monitors during setup or USB capture.
Saved Wi-Fi photo capture does not need an open serial connection.

Build and flash:

```bash
make build                 # Compile without changing the board; also the default goal.
make flash                 # Build, then update only the application.
make dev-connect           # Automated USB setup and authenticated Wi-Fi readiness check.
make photo                 # Prefer verified Wi-Fi; explicit USB development fallback.
make photo TRANSPORT=wifi  # Wi-Fi only: never mask a failed wireless test with USB.
make photo TRANSPORT=usb   # Force the existing USB development path.
make photo RESOLUTION=640x480 # Capture a smaller image.
make audio SECONDS=5       # Wi-Fi-first 16 kHz mono PCM16 WAV recording.
make video TRANSPORT=usb   # Verified live 720p browser preview, no recording.
make av TRANSPORT=usb      # Verified live camera + microphone browser preview.
make av TRANSPORT=wifi     # Experimental: Wi-Fi throughput is not yet reliable.
make video TRANSPORT=wifi STREAM_RESOLUTION=96x96 # Lowest-resolution diagnostic.
make av TRANSPORT=wifi STREAM_RESOLUTION=96x96    # Same diagnostic with 16 kHz audio.
make help                  # List goals and configurable variables.
```

`TRANSPORT=auto` is the default. The client first tries a saved authenticated
Wi-Fi endpoint. Before a capture starts, an unavailable Wi-Fi path can fall back
to USB and is clearly reported. Certificate/owner failures, malformed responses,
and interrupted transfers do not fall back or silently start another capture.

Wi-Fi photos and short audio recordings are hardware-verified. A later
ten-second audio download timed out; incomplete audio is rejected rather than
saved, so maximum-length wireless recording is not yet proven reliable. The same
firmware retains all USB diagnostics and adds live video/AV over both
transports. **Wi-Fi live AV is experimental and not yet reliable**: transfer
stalls caused low video delivery and audio queue overruns on the tested LANs.
The identical live protocol over USB delivered 44 frames in five seconds
(8.8 fps) with continuous audio and no reported overruns.

`make video` and `make av` now open a local browser viewer, rather than saving
MP4 files. Click Start/Stop to control acquisition; Ctrl-C stops the viewer.
Sessions have a configurable safety limit: `STREAM_SECONDS=1..300`. No live
media is written to disk or the SD card. The viewer listens only on loopback
and verifies a per-viewer request token; the accessory Wi-Fi link still uses
verified TLS and owner authentication.

`STREAM_RESOLUTION` is `1280x720` by default. Explicitly select `96x96` for the
lowest-resolution diagnostic; this changes neither photo defaults nor the
configured audio format. The host checks the actual JPEG dimensions as well as
stream metadata, and the viewer sizes its canvas accordingly.

On a trusted, non-isolated LAN, a ten-second 96x96 **video-only** test delivered
249 images (24.9 fps). The combined 96x96/48 kHz Wi-Fi test failed with an
audio-queue drop. Lower video resolution therefore does not establish reliable
combined streaming. Guest/IoT client isolation blocks device reachability
regardless of image size.

For reliable live development use `TRANSPORT=usb` for now. `auto` prefers Wi-Fi
and falls back only before capture; it does not conceal mid-stream failures.
No BLE/Android pairing is implemented.

The original saved USB clips remain available through [capture.py](capture.py):

```bash
uv run --no-project --with esptool python capture.py av \
  --seconds 10 --output captures/usb-recording
uv run --no-project --with imageio-ffmpeg python encode_video.py \
  captures/usb-recording captures/usb-recording.mp4
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

## Wi-Fi Speed Test

```bash
make speed-test TRANSFER_BYTES=1048576 BLOCK_BYTES=4096
```

This sends deterministic synthetic bytes in both directions over verified
HTTPS, checks integrity, and reports Mbit/s, KiB/s, RSSI, elapsed transfer time,
and TLS setup time separately. Camera/microphone capture is excluded. It never
falls back to USB or sends media.

Measured with 1 MiB transfers: **2.41 Mbit/s module-to-laptop** and
**2.65 Mbit/s laptop-to-module**. Smaller 128 KiB tests reached 2.60/3.41 Mbit/s.
These are application-stack measurements, not the chip's raw Wi-Fi limit.

Subsequent repetition showed substantial variability: a 1 MiB download fell
to 0.160 Mbit/s and its following upload timed out. Three 64 KiB trials each
way passed integrity checks but had medians of 0.116 Mbit/s download and
0.160 Mbit/s upload. Idle ICMP latency averaged 248 ms in a ten-packet sample.
The earlier higher numbers are not a stable-throughput guarantee.

[Espressif's ESP32-S3 guide](https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/api-guides/wifi.html#esp32-s3-wi-fi-throughput)
reports best lab over-air results up to 20 Mbit/s TCP and 30 Mbit/s UDP.
Its higher shielded-box numbers use controlled conditions and tuned buffers.
Our prebuilt Arduino SDK has 5760-byte TCP buffers; the cited newer default
performance profile uses 32 KiB. That discrepancy warrants profiling, not an
assumption that changing a compiler define retunes the prebuilt stack.

Audio now defaults to **16 kHz mono PCM16**, 256 kbit/s raw rather than the former
768 kbit/s. Live decoding/playback accepts declared 16/48 kHz formats. Reducing
audio did not yet resolve the combined Wi-Fi live stream's queue drops.

## Wireless Identity and Private State

Trusted physical USB setup provisions a unique ECDSA device certificate and
controller credential. The host verifies the pinned certificate and stable
device hostname before sending the credential over TLS. All HTTPS photo/status
routes require authorization; audio/live routes use the same authentication.
There is no plaintext accessory media endpoint.

Keep `.private/` safe: it contains the laptop's owner token. Missing or mismatched
owner credentials are not automatically replaced. An incomplete setup record
is retained if an acknowledgement is lost, allowing a USB retry to recover a
device that committed its configuration.

The current board does not have flash encryption/secure boot enabled. Device
Wi-Fi and TLS configuration is stored in NVS, so physical flash access is outside
this development prototype's protection boundary. New flash backups may contain
network credentials and device keys and must remain private.

## Scope

The tested board has 16 MB flash and 8 MB octal PSRAM. The driver acquires
2048x1536 JPEG still photos by default, with 640x480 available on request,
and 16 kHz mono PCM16 audio. Video defaults to 1280x720 (720p); saved USB clips
can be encoded as AAC at their declared rate with a codec-safe bitrate.
The original unprocessed WAV is preserved. The ESP32-S3's camera peripheral
and a separate microphone task operate concurrently; audio is buffered in PSRAM
and transferred over USB after video acquisition.

Saved USB clips can be encoded into MP4 on the laptop, not on the ESP32. Live
preview does not encode or save MP4. There is no SD-card access, Bluetooth
pairing, or Android integration. Wi-Fi streaming is an experimental prototype,
not validated for sustained real use. The RGB LED shares GPIO48 with the camera
and is not driven.

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
