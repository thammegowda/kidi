# W11 USB and Wi-Fi Capture

The board is running the diagnostic firmware, not the factory firmware. It
supports photos, audio recordings, silent video, and simultaneous audio/video
capture.

The same firmware supports USB capture, authenticated HTTPS photos/audio
recordings, and an experimental live camera/microphone viewer. Wi-Fi uses the
module's station connection to the laptop's LAN. Live USB AV is hardware-verified;
Wi-Fi live AV still stalls and may fail with explicit audio-overrun errors.
Bluetooth pairing and Android integration are not implemented.

## Hardware

| Component | Configuration |
|---|---|
| Processor | ESP32-S3 |
| Flash | 16 MB |
| PSRAM | 8 MB octal PSRAM |
| Camera | OV3660; JPEG photos at 2048 x 1536 by default; 640 x 480 optional |
| Video | 1280 x 720 (720p) |
| Microphone | PDM; 16 kHz speech audio, mono, signed 16-bit PCM |

### Pin Mapping

| Signal | GPIO |
|---|---|
| Camera XCLK | 10 |
| Camera SDA / SCL | 40 / 39 |
| Camera D0-D7 | 15, 17, 18, 16, 14, 12, 11, 48 |
| Camera VSYNC / HREF / PCLK | 38 / 47 / 13 |
| Microphone clock / data | 42 / 41 |

> **Shared pin:** The RGB LED shares GPIO48 with the camera. The diagnostic does
> not drive the LED while the camera is active.

## Common Commands

Run commands from this directory:

```bash
make help
make build
make flash
make dev-connect
make cache-wifi
make speed-test
make status
make temperature
make sensors
make photo
make photo TRANSPORT=wifi
make photo TRANSPORT=usb
make photo RESOLUTION=640x480
make audio SECONDS=5
make video TRANSPORT=usb
make av TRANSPORT=usb
make av TRANSPORT=wifi  # Experimental; current throughput is not reliable.
make format-check test
```

> **Flashing prerequisite:** `make flash` builds first and updates only the
> application at `0x10000`. The diagnostic bootloader and partition table must
> already be installed.

### Make Variables

| Variable | Purpose |
|---|---|
| `PIO_ENV` | PlatformIO build profile |
| `CHIP` | ESP chip type passed to the flasher |
| `PORT` | USB serial port |
| `TRANSPORT` | Media route: `auto` (Wi-Fi first), `wifi`, or `usb` |
| `BAUD` | Flashing baud rate |
| `SECONDS` | Saved audio recording duration, from 1 to 10 seconds |
| `STREAM_SECONDS` | Live session safety limit, from 1 to 300 seconds |
| `STREAM_RESOLUTION` | Live image size: `1280x720` (default) or `96x96` (lowest diagnostic profile) |
| `RESOLUTION` | Photo resolution: `2048x1536` (default) or `640x480` |
| `CAPTURE` | Capture path or filename prefix |
| `OUTPUT` | Encoded MP4 filename |
| `BACKUP` | Destination for a new flash backup |
| `FACTORY_BACKUP` | Factory backup used for restoration or verification |

Saved photo/audio names default to timestamped paths under `captures/`.
The `video` and `av` goals open a live viewer and do not use `CAPTURE` or save
MP4. Direct USB saved-clip commands and `make encode` remain available below.

## Live Preview

```bash
make video TRANSPORT=usb
make av TRANSPORT=usb STREAM_SECONDS=60
```

These commands open a local browser viewer. Click Start to acquire media and
Stop to close the session. Ctrl-C stops the laptop viewer. No recording is saved.
Video is 1280x720 JPEG; live audio is 16 kHz mono PCM16 with browser high-pass
filtering and adjustable playback gain. Browser audio buffering adds latency;
sample-accurate audiovisual synchronization is not claimed.

The device uses independent acquisition workers, a latest-frame queue, and a
bounded audio queue. Slow transmission drops stale preview frames; missing
audio or DMA overruns end the stream explicitly rather than hiding a gap.

**Current limitation:** `TRANSPORT=wifi` is experimental. The tested networks
had TLS transfer stalls, sometimes delivering fewer than one frame per second
or overflowing the audio queue. USB delivered 8.8 fps in a five-second test with
continuous audio. Do not infer frame rate from browser refresh or MP4 playback.

Headless validation, without saving media:

```bash
uv run --no-project --with-requirements requirements-wifi.txt python live.py \
  --mode av --transport usb --seconds 5 --check
```

Lowest-resolution Wi-Fi diagnostic:

```bash
make video TRANSPORT=wifi STREAM_RESOLUTION=96x96 STREAM_SECONDS=10
make av TRANSPORT=wifi STREAM_RESOLUTION=96x96 STREAM_SECONDS=10
```

For a terminal-only test, use `live.py --resolution 96x96 --check` with the same
mode, transport, and duration flags. Current audio is 16 kHz mono PCM16.

The ten-second 96x96 Wi-Fi video-only test delivered 249 distinct images,
24.9 fps. Combined Wi-Fi video/audio still failed with one dropped audio chunk
and zero reported DMA overruns. The host verifies JPEG dimensions rather than
trusting the stream's declared size. A separate low-resolution USB AV comparison
hit a packet-framing error near the end and is not counted as a passing run.

Use a trusted LAN that allows peer traffic. An isolated guest/IoT SSID prevents
this test even when both devices are associated. The laptop can use the router's
5 GHz band while the ESP32 uses its 2.4 GHz band; they do not need matching bands.

### Capture Requirements

- The default serial port is auto-detected. Override it with `PORT=...`
  for Make commands or `--port` for the capture script.
- Capture durations must be between **1 and 10 seconds**.
- Full-resolution still photos use `RESOLUTION=2048x1536`; video and simultaneous
  AV use 1280x720. Audio uses 16 kHz mono PCM16. Install the updated firmware
  with `make flash` first.

## Temperature and Other Sensors

```bash
make temperature
make sensors
```

`temperature` reports the ESP32-S3's die temperature and CPU clock. The sensor
uses its 20-100 C range, with a specified error below 2 C; it does not measure
ambient temperature.

`sensors` emits a machine-readable `KIDI_SENSORS` JSON snapshot:

| Field | Contents |
|---|---|
| `schema_version` | Snapshot format version |
| `timestamp_us` | Snapshot completion time on the board's microsecond clock |
| `cpu_mhz` | Current CPU clock |
| `chip_temperature` | Die temperature in `celsius` |
| `imu` | QMI8658 `acceleration_g`, `gyroscope_dps`, and `temperature_c` |
| `board_temperature` | STS35 `celsius`, if detected |
| `battery_rail` | GPIO2-derived `voltage_v`; `battery_presence` remains `unknown` |

Sensor objects include an `available` flag. An unavailable sensor reports an
explicit `error`, not a zero-valued substitute. Sensor I2C uses a different
controller from the camera, and the IMU channels are disabled after each
snapshot. These readings are not yet streamed alongside audio/video.

The VBAT rail can have voltage without an attached battery. Do not interpret it
as proof of battery presence, a charging state, or a charge percentage.
- Output files and capture directories must not already exist.
- Close other serial monitors before capturing or flashing.
- [capture.py](capture.py) transfers binary data. Do not send `photo`, `audio`,
  `video`, or `av` commands from a plain-text terminal expecting readable output.

## Automated Wi-Fi Setup and Photo/Audio Capture

```bash
make dev-connect
make photo
make audio TRANSPORT=wifi SECONDS=5
```

Short Wi-Fi audio recordings are verified. A ten-second recording timed out
in later testing; incomplete transfers fail without saving a success-shaped
WAV or retrying a second capture over USB.

Setup uses USB to identify the board, reuse the existing device network or a
cached personal-network credential, provision the device identity
and network, and verify the HTTPS endpoint. The laptop keeps its current
internet connection. The network must have a reachable 2.4 GHz LAN for the
ESP32-S3; guest isolation and managed authentication may prevent this.

The selected network's authorized credential is cached in `.kidi.wifi.txt`.
It is plaintext JSON, Git-ignored, and restricted to the current account
(0600 on POSIX; owner-only Windows ACL). Matching SSIDs reuse it without another
Keychain request. After reboot, setup first waits for the device's stored
network to reconnect.

Normal setup and capture are strictly cache-only for passwords: they never
open Keychain or request password entry. New-network credential access requires
the explicit `make cache-wifi` goal. The direct equivalent is
`wifi.py cache-network --authorize-credentials`; that authorization flag is
rejected for all other commands.

```bash
make cache-wifi         # Populate the current-network cache once.
make forget-wifi-cache  # Remove laptop cached passwords, not device identity.
```

Use `WIFI_SSID=your-network` for an explicit network selection if automatic
name discovery is unavailable. Cached credentials apply only to matching SSIDs.

Do not publish or share this cache. Cancelled credential access stops the
operation and does not populate it.

On macOS, approve the named Wi-Fi helper's scoped request only if its network
and Mac account match the intended setup. Wi-Fi Personal has no separate
username; the displayed Mac account is the local authorization context.
Location/local-network permissions may also be needed. No password is logged,
written to source, or requested in chat.

```bash
make photo TRANSPORT=wifi  # Verified Wi-Fi only.
make photo TRANSPORT=usb   # Physical USB development capture.
```

`auto` prefers a saved, verified Wi-Fi connection without opening USB. It can
use an explicitly reported USB fallback for pre-request Wi-Fi unavailability.
It never falls back after certificate/authentication failure or an interrupted
capture whose outcome is unknown.

Owner credentials and the generated helper are stored under Git-ignored
`.private/` with restrictive permissions. Preserve that directory. The prototype
does not silently replace another owner and does not yet implement production
BLE ownership transfer.

macOS “No route to host” for a valid LAN endpoint can indicate denied
local-network permission. Allow the app/terminal running the client under
System Settings > Privacy & Security > Local Network; do not disable TLS to
work around a routing problem.

## Authenticated Wi-Fi Speed Test

```bash
make speed-test TRANSFER_BYTES=1048576 BLOCK_BYTES=4096
```

Transfers a known pattern in both directions over certificate-verified HTTPS
with owner authentication, checking every byte. Camera and microphone are
excluded, and there is no USB fallback. Each direction reports bytes, transfer
time, payload Mbit/s and KiB/s, RSSI, and TLS setup time separately.

Limits: 4096-1048576 payload bytes; 256-16384 bytes per block; a 60-second
device transfer deadline. A timeout is a failed measurement, not a zero-speed
success. For one direction, run `speed.py --direction download` or `upload`
with `uv run --no-project --with-requirements requirements-wifi.txt python`.

The measured 1 MiB results were 2.41 Mbit/s download and 2.65 Mbit/s upload.
These include HTTPS/application overhead; they are not raw radio capacity.

## Direct USB Python Commands

These commands are equivalent to the Make workflow. All Python tools use
`uv run`. These examples deliberately use USB; `make photo` now prefers Wi-Fi.

### Status, Photos, and Audio

```bash
mkdir -p captures

uv run --no-project --with esptool python capture.py status

uv run --no-project --with esptool python capture.py photo \
  --output captures/another-photo.jpg

uv run --no-project --with esptool python capture.py photo \
  --resolution 2048x1536 --output captures/full-resolution-photo.jpg

uv run --no-project --with esptool python capture.py audio \
  --seconds 5 --output captures/another-recording.wav
```

### Silent Video

```bash
uv run --no-project --with esptool python capture.py video \
  --seconds 10 --output captures/another-video-frames

uv run --no-project --with imageio-ffmpeg python encode_video.py \
  captures/another-video-frames captures/another-video.mp4
```

### Simultaneous Audio and Video

```bash
uv run --no-project --with esptool python capture.py av \
  --seconds 10 --output captures/another-av-capture

uv run --no-project --with imageio-ffmpeg python encode_video.py \
  captures/another-av-capture captures/another-av-video.mp4
```

## Capture Behavior

### Audio

The original WAV is preserved without gain or filtering.
[capture.py](capture.py) also saves a DC-removed copy.
New recordings use 16 kHz mono PCM16; this is not 24-bit audio.

The confirmed speech playback file, `captures/w11-mic-speech-playback.wav`, was
high-pass filtered at 80 Hz and amplified 16x for listening. This processing was
performed on the Mac, not on the board.

### Video

Each JPEG and its board timestamp is saved alongside `capture.json`.
[encode_video.py](encode_video.py) uses these timestamps to produce H.264 MP4
at a 10 fps playback rate.

An earlier VGA test produced **92 distinct frames in 10.000999 seconds**,
approximately **9.2 fps**. Its MP4 contains 100 output frames over exactly
10.000 seconds, with repeated frames where needed for constant-rate playback.

Current video defaults to 1280x720. H.264 encoding happens on the Mac; the board
transfers JPEG frames over USB. The 10 fps playback rate is not a guarantee of
ten distinct source frames per second.

### Simultaneous Audio and Video

A separate FreeRTOS task drains microphone DMA while the main task captures
and transfers JPEG frames. Audio is buffered in PSRAM and transferred after
video acquisition: **capture is simultaneous, but USB transmission is not**.

- Camera acquisition timestamps and audio timing share one board-clock origin.
- `capture.json` records sample count, elapsed audio time, and DMA overflow count.
- Missing samples or a reported overflow fail the capture rather than hiding gaps.
- The host requires measured audio duration to be within 200 ms of nominal.
- The original microphone recording is saved as `audio.wav`.
- MP4 playback applies an 80 Hz high-pass and up to 16x audio gain, limited by
  the measured peak to leave 1 dB headroom. Saved USB recordings can be encoded
  as AAC at the original declared sample rate with a codec-safe bitrate.
- The RGB LED is not driven while the camera is active.

> **Synchronization limit:** The timing check is not a sample-accurate
> audio/video synchronization guarantee. DMA buffering and camera frame
> intervals introduce timing uncertainty.

An earlier verified 720p/48 kHz 10-second concurrent test produced:

| Measurement | Result |
|---|---|
| Camera images | 36 distinct 1280 x 720 images; 3.6 fps over the requested interval |
| Microphone samples | 480,000 at 48 kHz |
| Reported DMA overruns | 0 |
| Measured audio acquisition time | 10.000039 seconds |
| User playback check | Speech audible and understandable |

The resulting MP4 is exactly 10 seconds at a 10 fps playback rate. It repeats
frames to fill that playback timeline; the source camera delivered 3.6 fps in
this test. Results are saved in `captures/av-720p-48khz-10s.mp4`.

## Build and Flash

### Compile Without Flashing

```bash
uv run --no-project --with platformio pio run --project-dir .
```

### Update the Diagnostic Application

This command updates only the application on the already-configured board:

```bash
uv run --no-project --with esptool esptool \
  --chip esp32s3 --port /dev/cu.usbmodem1101 --baud 460800 \
  --after hard-reset write-flash \
  --flash-mode keep --flash-freq keep --flash-size keep \
  0x10000 .pio/build/w11/firmware.bin
```

## Backup and Restoration

### Original Factory Backup

| Property | Value |
|---|---|
| File | `backups/w11-factory-backup-20261003.bin` |
| Size | 16,777,216 bytes |
| Verification | Matched the board's flash before firmware replacement |

SHA-256:

```text
f6ea91cf9b8aca76944f649572efc3ca2a2192a6d346e867ef8999e34a8e7f74
```

> **Private data:** The backup contains all original flash contents. Treat it
> as private.

### Back Up the Current Firmware

`make backup` saves and verifies the **current** flash to a new timestamped path.
It refuses to overwrite an existing backup and does not recreate factory data.

`make verify-backup` compares the factory backup with the installed flash.
A mismatch is expected while diagnostic firmware is installed.

### Restore Factory Firmware

> **Warning:** Restoration overwrites the installed diagnostic firmware.

Using Make:

```bash
make restore CONFIRM_RESTORE=yes
```

Equivalent direct command:

```bash
uv run --no-project --with esptool esptool \
  --chip esp32s3 --port /dev/cu.usbmodem1101 --baud 460800 \
  --after hard-reset write-flash \
  --flash-mode keep --flash-freq keep --flash-size keep \
  0 backups/w11-factory-backup-20261003.bin
```

### Verify Restored Flash

The following direct command leaves the board in the bootloader. Press **R**
after verification to run the restored firmware.

```bash
uv run --no-project --with esptool esptool \
  --chip esp32s3 --port /dev/cu.usbmodem1101 --baud 460800 \
  --after no-reset verify-flash \
  0 backups/w11-factory-backup-20261003.bin
```

## Troubleshooting and Safety

Warmth can result from the CPU, powered camera, PSRAM, and active captures.
The CPU remains at its configured clock while idle; yielding to FreeRTOS is
not the same as deep sleep. Use `make temperature` to obtain a reading rather
than estimating the chip temperature by touch.

If the board is uncomfortably hot, smells unusual, or shows discoloration,
stop testing, disconnect USB, and let it cool.

If flashing cannot connect, enter the ROM bootloader:

1. Hold **B**.
2. Press and release **R**, keeping **B** held.
3. Release **B**.

Preserve the image's DIO bootloader header; do not force QIO in `write-flash`.

**Unplug the board before attaching or removing the expansion board.**
