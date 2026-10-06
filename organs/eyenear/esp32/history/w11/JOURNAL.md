# W11 diagnosis and capture journal

Archived hardware history. The W11 target and board drivers have been removed.
These measurements and recovery notes are not instructions for the next board.

Date: 2026-10-03. Host: Apple Silicon macOS. All tests were local; no camera,
audio, or flash contents were sent to remote services. Kidi's application source
and the SD card were not changed during the diagnosis.

## Board discovery

The board initially did not enumerate on USB. Reconnection and BOOT/reset
attempts eventually produced an Espressif USB Serial/JTAG device and
`/dev/cu.usbmodem1101`. The exact initial connection fault was not established.
macOS's built-in USB CDC driver worked; no CH340/CP210x driver was installed.

ROM diagnostics identified:

- ESP32-S3, silicon revision v0.2, 240 MHz capability, 40 MHz crystal.
- 8 MB embedded PSRAM and 16 MB SPI flash.
- Secure Boot and flash encryption disabled.
- An existing bootloader, partition table, and application image.

The manufacturer layout shows tiny R/reset and B/BOOT buttons flanking USB-C
on the lower board. Hold B, press and release R, then release B to enter the ROM
bootloader. Press R alone to run installed firmware. Unplug before attaching or
removing the expansion board.

## Indicators

The published mainboard schematic connects the dedicated red LED to the
SGM40567 charger's status output, not to a generic "board running" signal.
Its datasheet specifies slow blinking during charging and a continuous
charge-completion indication for 51.2 seconds, followed by the LED turning off.
Without a connected battery, that sequence is not proof a battery was charged.
USB remained operational with the LED off.

The separate RGB LED is programmable and shares GPIO48 with camera data.
Diagnostic firmware does not drive it while using the camera.

The published schematic is labeled v0.1; these circuit details are not an
independent electrical verification of every shipped revision.

## Factory firmware

USB output identified the preloaded application as
`W11 Factory Test + Data Output`.

| Component | Factory result | Limit of evidence |
|---|---|---|
| QMI8658C IMU | PASS | Self-test and ongoing sensor data |
| Digital microphone | PASS | Nonzero, varying samples |
| Camera | PASS | Driver initialization only |
| Wi-Fi | PASS | AP initialization, not a connected-client test |
| BLE | PASS | Initialization, not a connected-client test |
| ADC | PASS | Basic sampling, not calibrated voltage accuracy |
| SD card | FAIL / no card | Excluded from subsequent tests |

The example's `OV5640` test label is hardcoded; it did not identify the actual
sensor. Our diagnostic later read sensor PID `0x3660`, confirming OV3660.
The published factory example briefly initializes and then disconnects its AP;
its printed `192.168.4.1` is not proof a preview webpage is available. The shipped
binary's behavior is not identical in every detail to the downloadable example.

## Backup and diagnostic installation

With the user's approval, the complete 16 MB flash was read and verified
against the board by a matching digest before firmware replacement.

- Local backup: `backups/w11-factory-backup-20261003.bin`.
- Size: 16,777,216 bytes.
- SHA-256: `f6ea91cf9b8aca76944f649572efc3ca2a2192a6d346e867ef8999e34a8e7f74`.
- The backup is private and Git-ignored; preserve it separately before removing
  this checkout.

A first flashing command forced QIO into a bootloader image that required a DIO
ROM-startup header. ROM boot failed. Reflashing the generated bootloader with
`--flash-mode keep --flash-freq keep --flash-size keep` corrected it.
Subsequent application updates preserved that working bootloader.
This was a flashing-configuration issue, not evidence of a defective board.

## Sequential camera and microphone tests

- Real JPEG capture identified OV3660 and produced valid 640x480 images.
- Two user-positioned photos had visibly different content.
- Microphone recordings used 16 kHz mono signed 16-bit PCM.
- A 10-second spoken test returned all 160,000 samples without raw clipping.
- The user confirmed understandable speech in local playback. A playback-only
  copy used an 80 Hz high-pass and 16x gain; the original WAV was preserved.
- The silent-video test captured 92 distinct JPEG frames over 10.000999 seconds,
  approximately 9.2 fps. Frame intervals had median 108.00 ms and maximum
  207.38 ms.
- The resulting MP4 decoded fully: 640x480, exactly 10.000 seconds, 100 output
  frames at 10 fps. Some source frames were repeated for uniform playback.

Relevant local outputs include:

- `captures/w11-photo-1.jpg` and `captures/w11-photo-2.jpg`.
- `captures/w11-mic-speech.wav` and `captures/w11-mic-speech-playback.wav`.
- `captures/w11-video-10s.mp4`.
- `captures/w11-video-frames/capture.json` and the original JPEGs.

## Simultaneous audio/video test

Firmware v3 records microphone data on a separate FreeRTOS task while the main
task captures and transfers JPEG frames. Microphone samples are buffered in
PSRAM, then transferred after video acquisition. Camera acquisition timestamps
and microphone timing use one board-clock origin.

A preliminary three-second test produced 26 frames, all 48,000 audio samples,
and zero reported DMA overruns.

The user then spoke, moved the camera, and clapped during a ten-second test:

| Measurement | Result |
|---|---|
| Resolution | 640x480 JPEG |
| Camera frames | 72 distinct frames, approximately 7.2 fps over the requested interval |
| First/last camera acquisition | 0.004577 s / 9.801467 s |
| Camera frame intervals | Median 108.06 ms; maximum 396.20 ms |
| Video capture loop elapsed | 10.054 s |
| Audio | 160,000 samples, 16 kHz mono PCM16 |
| Measured audio acquisition time | 10.000002 s |
| Reported audio DMA overruns | 0 |
| Longest audio block read | 64.026 ms |
| Raw clipped samples | 0 |

All camera acquisition timestamps were inside the microphone recording
interval. The MP4 contained both H.264 video and mono AAC audio and decoded
successfully. Its 10 fps playback includes repeated frames; it is not evidence
that the camera delivered 10 distinct frames per second.

Playback audio used an 80 Hz high-pass and approximately 5.50x gain, capped by
the measured peak to leave 1 dB headroom. The user confirmed clear speech and an
apparently synchronized visible clap. DMA buffering and frame intervals still
limit synchronization precision; this is not a sample-accurate sync guarantee.

Outputs:

- `captures/w11-av-10s.mp4`: combined playback.
- `captures/w11-av-10s-capture/audio.wav`: original unprocessed recording.
- `captures/w11-av-10s-capture/capture.json`: timings and audio integrity metadata.
- `captures/w11-av-10s-capture/frame-*.jpg`: original camera images.

Photo, standalone audio, and silent-video commands were retested successfully
after concurrent capture. Firmware v3 remains installed at the user's request;
the factory backup remains available.

## Sensor specifications versus measured throughput

The OV3660's maximum active image is 2048x1536, approximately 3.15 megapixels.
Its published sensor transfer limits are:

| Image size | Sensor-rated maximum |
|---|---:|
| 2048x1536 | 15 fps |
| 1920x1080 | 20 fps |
| 1280x720 | 45 fps |
| 1024x768 | 45 fps |
| 640x480 | 60 fps |
| 320x240 | 120 fps |

These are sensor limits, not measured end-to-end ESP32 JPEG, USB, or Wi-Fi
streaming performance. The diagnostics targeted only 10 fps and used one
camera frame buffer; they were not maximum-throughput benchmarks.
JPEG quality, lighting/exposure, buffering, transport, and concurrent work
affect actual rates.

## Reusable project refactor

The project now has a Makefile for building, flashing, capturing, encoding,
backing up, formatting, and testing. Firmware uses Kidi-style snake_case names,
trailing return types, and repository formatting under `kidi::esp32`.
Headers remain beside the source: `esp32.h` provides generic SDK type aliases,
and `w11.h` isolates the selected board's pin mapping. The namespace is not
limited to camera/microphone peripherals.

The tested GCC 8.4 toolchain supports GNU C++17 for this project. Strict C++17
failed because the bundled FreeRTOS headers require GNU variadic-macro
extensions; GNU C++17 built successfully. Native Kidi remains C++23.

Validation after the refactor:

- Formatting and the firmware cross-build passed.
- Seven host protocol tests passed, including rejection of missing samples,
  DMA overruns, timing errors, and malformed binary boundaries.
- Unconfirmed factory restoration and backup overwrites were rejected before
  accessing the board.
- The new `make flash` goal updated only the application and verified its hash.
- `make status`, `make photo`, and a one-second `make audio` capture passed.
- A one-second `make video` captured ten frames and encoded a fully decodable MP4.
- A three-second `make av` captured 28 frames and all 48,000 audio samples, with
  zero reported DMA overruns; the H.264/AAC MP4 decoded successfully.
- The USB command and binary-transfer protocol remained unchanged.

The resulting local AV regression recording is `captures/make-refactor-av.mp4`.

## Full-resolution photos and higher-quality defaults

Firmware v5 adds opt-in full-resolution stills:

```bash
make photo RESOLUTION=2048x1536
```

The board returned a 116,801-byte JPEG, and the decoded image dimensions were
confirmed as 2048x1536 (approximately 3.15 MP). A subsequent default photo still
returned 640x480, confirming that the still-resolution selection does not
silently change the default.

At the user's request, video and concurrent AV now default to 1280x720.
Microphone capture uses 48 kHz mono PCM16, not 24-bit audio. New capture metadata
declares the sample rate; the host rejects missing or mismatched audio formats
rather than labeling recordings with an assumed rate. MP4 audio targets
128 kbit/s AAC, with a codec-limit cap for older lower-rate recordings.
Original raw WAVs remain unprocessed.

Short tests verified both silent 720p video and concurrent 720p/48 kHz capture.
The user then performed a full ten-second concurrent recording:

| Measurement | Result |
|---|---|
| Video | 1280x720 |
| Source camera frames | 36 distinct frames; 3.6 fps over the requested interval |
| Camera acquisition times | 0.028310 s to 9.963207 s |
| Camera frame intervals | Median 287.14 ms; maximum 344.59 ms |
| Capture loop elapsed | 10.236 s |
| Raw audio | 480,000 samples; 48 kHz mono PCM16 |
| Audio payload | 960,000 bytes |
| Measured audio acquisition time | 10.000039 s |
| Reported DMA overruns | 0 |
| Longest microphone read | 21.380 ms |
| Raw clipped samples | 0 |
| Encoded output | H.264/AAC MP4, exactly 10 s, 10 fps playback |

All camera timestamps lay inside the audio acquisition interval. The MP4 fully
decoded, and the user confirmed audible, understandable speech. The 10 fps
playback repeats source images and does not imply ten distinct captures per
second. This was a resolution/format test, not a maximum-throughput benchmark
or a sample-accurate synchronization test.

The encoder was also retested against the original 16 kHz/VGA recording and
preserved its original sample rate. Factory backup and SD-card contents were
unchanged.

Local results:

- `captures/full-resolution-2048x1536.jpg`.
- `captures/audio-48khz-check.wav`.
- `captures/av-720p-48khz-10s.mp4`.
- `captures/av-720p-48khz-10s/capture.json` and original JPEG/audio data.

## Laptop Wi-Fi Photo Slice

The implementation now uses one firmware for USB and Wi-Fi. The initial archive
plan was superseded by the user: USB capture remains active, and both photo
transports share camera acquisition with a mutex. Wi-Fi support in this slice
is limited to photographs; audio/video and sensor commands remain USB.

The laptop setup command detects the board and provisions the selected saved
Wi-Fi profile over trusted USB. A named macOS helper obtains the current SSID
with Location Services authorization and scopes credential consent to the exact
network and logged-in Mac account. The password is returned through a private
process pipe, not printed or written to the project.

Setup binds a unique ECDSA certificate/hostname and owner credential to the
device. HTTPS photo/status endpoints require authorization. The laptop pins
the device certificate; wrong certificate or owner authentication fails closed
without a USB fallback. Device configuration persists in NVS, which is not
encrypted on this prototype board.

Integration issues found and corrected:

- The SDK's constant-time comparison header lacked C++ linkage guards.
- The native USB receive queue's 256-byte default truncated configuration
  packets; the unified firmware uses an 8192-byte receive queue.
- macOS redacted SSID data from ordinary CLI tools; the authorized native
  helper obtained it without bypassing the OS permission boundary.
- Initial local TCP access was denied by macOS Local Network privacy. Both the
  accessory and gateway returned "No route to host" despite valid routing/ARP;
  enabling the app/terminal's local-network access resolved it.
- The client initially retained the five-second TCP-connect timeout for TLS
  handshakes/response reads. It now applies the intended 25-second timeout.
- USB AV exposed a too-small/short-wait TX path under concurrent Wi-Fi load.
  The firmware now uses a 4096-byte TX queue and a one-second TX wait. The host
  rejects unexpected binary control data without printing raw media.

Hardware validation on macOS:

- Automated selected-network provisioning completed without manual SSID or
  password entry.
- Authenticated full-resolution HTTPS photo saved as
  `captures/wifi-photo-first.jpg`; decoded dimensions were 2048x1536.
- Default automatic routing preferred Wi-Fi.
- Wi-Fi-only capture completed without opening any serial connection:
  `captures/wifi-no-serial-photo.jpg`, also 2048x1536.
- Forced USB photo and one-second 48 kHz USB audio still worked.
- USB video and a three-second USB AV regression passed with Wi-Fi enabled;
  AV delivered 18 frames, all 144,000 audio samples, and zero reported overruns.
- Automatic USB fallback was exercised against a deliberately unavailable TCP
  endpoint while retaining the real hardware USB path. Saved credentials were
  unchanged. This was a controlled outage test, not a router disconnection.
- The firmware rejected a wrong owner before capture; the host rejected a
  wrong device certificate before an HTTP request.
- The laptop's original Wi-Fi internet route remained unchanged.
- The saved station configuration, device identity, and owner authentication
  survived reboot without another credential request.

The final automated suite contains 34 passing USB/Wi-Fi tests, including local
TLS integration, permission/cancellation behavior, private-file restrictions,
transport selection, authentication rejection, and interrupted-transfer
handling.

Local owner credentials and generated helper files remain under Git-ignored
`.private/`; capture results remain Git-ignored. Windows network/profile code
and encrypted-key rejection have automated coverage, but Windows hardware
validation is still pending.

## Audio Recording and Live Preview Work

Authenticated Wi-Fi audio recording was added, preserving 48 kHz mono PCM16
and the original USB recording commands. The user confirmed understandable
audio in an initial simultaneous Wi-Fi clip. Slow video delivery led the user
to change scope from saved Wi-Fi video to live preview without recording.

The firmware now shares microphone setup/readout in `audio.cpp`, and uses
independent camera and microphone workers for live acquisition. Camera frame
buffers and a one-slot preview queue discard stale frames. A bounded audio
queue has priority in delivery; overruns or sequence gaps fail explicitly.
USB and Wi-Fi carry the same typed packet protocol with sequence numbers,
timestamps, declared formats, telemetry, and a successful end marker.

`make video` and `make av` now launch a loopback-only browser viewer. Start/Stop
controls acquisition; live media is not saved on the laptop or SD card.
Sessions have a 1-300 second safety limit. TLS/device/owner verification is
unchanged. Browser access to the loopback stream requires a per-viewer token.

Verified:

- Three-second Wi-Fi audio: all 144,000 samples, 48 kHz mono PCM16.
- Wi-Fi full-resolution photo still decodes to 2048x1536.
- An initial ten-second Wi-Fi AV recording returned all 480,000 audio samples
  without DMA overruns, but only ten source images; its apparent MP4 frame rate
  was not a real camera frame rate.
- USB live AV delivered 44 frames in five seconds (8.8 fps), and 238,080 audio
  samples in a later test, with zero reported audio drops or DMA overruns.
  Live sessions stop at a wall-clock limit rather than promising the exact
  sample count of a saved recording.
- Automated tests cover typed packet framing, bounded payloads, sample
  continuity, authentication, end markers, and route selection.
- The browser viewer was exercised over USB: Start produced image/audio
  packets and approximately 8.4 received frames/second in the UI; the session
  safety limit returned it to the ready state, and Stop aborted the preview.
- USB Stop now uses an explicit command and waits for the firmware's
  post-cleanup acknowledgement before closing the port. CRLF line endings
  are handled. An early Stop of a 60-second session was acknowledged, and
  the camera was immediately reused successfully for a USB photo.
- Fifty automated tests, the cross-build, and formatting checks passed.

A ten-second saved Wi-Fi audio transfer also timed out before its complete
payload arrived during the later network tests. The client refused to save
that incomplete recording. Short three-second Wi-Fi audio and full-resolution
Wi-Fi photographs were subsequently verified after the live refactor, but
maximum-length wireless recording should not be presented as reliable.

**Not resolved:** Wi-Fi live AV repeatedly suffered transmission stalls and
audio queue overruns, even with strong signal. Camera acquisition continued
producing frames while network delivery lagged. The laptop was switched to a
new 2.4 GHz network and automated USB provisioning succeeded, but the streaming
failure persisted. This was a new selected network; it does not prove a test
against independent router infrastructure.

Experiments with TCP batching, modem sleep, TLS cipher selection, chunk
coalescing, PSRAM clip caching, and HTTP-task affinity did not establish a
reliable improvement. Unsuccessful transport tuning was removed. The retained
code uses the SDK response API and standard certificate-verified TLS defaults.
No plaintext media workaround, silent audio loss, or quality reduction was used.

The live viewer is a development prototype: USB streaming is verified, but
Wi-Fi live AV must not be described as completed or production-ready. The
remaining blocker needs focused transport profiling rather than another
nominal-FPS claim.

## Lowest-Resolution Wireless Live Test

Added an explicit `STREAM_RESOLUTION=96x96` diagnostic profile, the smallest
size listed by the tested camera driver. The normal live default remains
1280x720, photo defaults are unchanged, and audio stays 48 kHz mono PCM16.
The setting is wired through Make, laptop CLI, HTTPS requests, USB stream
commands, firmware metadata, and the browser canvas. The host checks JPEG SOF
dimensions against the requested resolution.

Initial wireless testing could not reach the accessory's TCP endpoint.
The laptop could reach the router, but not the module; both reported the same
selected Wi-Fi profile. The user confirmed that network was an isolated
guest/IoT network. This was a connectivity restriction, not a valid stream
performance result.

The user then selected a trusted dual-band network. Automated USB setup
configured the accessory without changing its identity. The laptop did not
need to be forced onto 2.4 GHz.

Results on that reachable LAN:

| Test | Result |
|---|---|
| Wi-Fi AV, requested 10 seconds, 96x96 + 48 kHz | Failed: one dropped audio chunk, zero reported DMA overruns |
| Wi-Fi video only, 10 seconds, 96x96 | Passed: 249 frames, 24.9 fps |
| Video acquisition timestamps | 0.030 to 9.921 seconds |
| Video-only receive time | 10.241 seconds |
| USB AV comparison, requested 10 seconds, 96x96 + 48 kHz | Framing error near the end; not a passing run |

The failing Wi-Fi AV telemetry reported 32 acquired camera frames and one
delivered frame before the audio error. The successful video-only run verifies
that low-resolution wireless camera delivery works, but it does not resolve the
combined stream's audio/transport issue.

Firmware cross-build, formatting, and 54 automated tests passed. Tests cover
resolution forwarding, actual JPEG size, mismatched declarations/payloads, and
the lowest-resolution protocol over verified TLS. No SD-card writes or saved
live recordings were used.

A second five-second video-only run received 93 frames (18.6 fps). A received
JPEG was fully decoded in memory to 96x96 RGB, confirming more than header
metadata alone. No image or recording file was saved for this decode check.

## Throughput Benchmark and Speech-Rate Audio

Added `make speed-test`, using owner-authenticated, certificate-verified HTTPS
in both directions with synthetic, deterministic bytes and full integrity
checks. Capture resources are locked out during the test; there are no
camera/microphone operations or media files. TLS setup is timed separately.
This measures the existing application stack, not raw PHY or plaintext iperf.

| Payload | Block size | Module to laptop | Laptop to module |
|---|---:|---:|---:|
| 128 KiB | 4096 bytes | 2.60 Mbit/s | 3.41 Mbit/s |
| 1 MiB | 4096 bytes | 2.41 Mbit/s | 2.65 Mbit/s |
| 128 KiB | 640 bytes | 1.87 Mbit/s | Not measured |
| 128 KiB | 1920 bytes | 2.47 Mbit/s | Not measured |
| 128 KiB | 16384 bytes | 2.06 Mbit/s | Not measured |

The 1 MiB transfers took 3.475 seconds download and 3.164 seconds upload,
excluding approximately one-second TLS setup. Device-reported upload receive
time was 3.121 seconds. Every successful transfer passed integrity validation.

[Espressif's target-specific guide](https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/api-guides/wifi.html#esp32-s3-wi-fi-throughput)
publishes best over-air lab results up to 20 Mbit/s TCP and 30 Mbit/s UDP.
Shielded-box figures are substantially higher and use the specified iperf
configuration. They are not a promise for this camera application.
The current prebuilt Arduino SDK uses 5760-byte TCP send/receive buffers,
whereas the newer cited default-performance profile uses 32 KiB, and its iperf
profile uses 64 KiB. This is a concrete difference to investigate, not a proven
explanation or something safely changed by defining a compiler macro against
prebuilt libraries.

With the user's approval, microphone capture now defaults to 16 kHz mono
PCM16, directly matching Whisper's input rate. Raw audio payload drops from
768 to 256 kbit/s. Saved recordings, metadata validation, live framing, tests,
and browser playback were updated consistently. Live decoding still accepts
explicitly declared legacy 48 kHz input; playback uses the declared rate.

The subsequent 96x96/16 kHz Wi-Fi AV test still failed: six audio-queue drops
and zero reported DMA overruns. Its last telemetry reported 136 acquired
camera frames and eight delivered frames. Lowering audio alone did not resolve
the simultaneous streaming issue.

## Private Selected-Network Credential Cache

At the user's request, `.kidi.wifi.txt` stores authorized selected-network
credentials as JSON text. It is Git-ignored (including its atomic temporary
file), mode 0600 on macOS/POSIX, and given a current-user-only ACL on Windows.
Writing the cache does not change the project directory's permissions.
The cache is plaintext and must not be shared or committed.

Matching SSIDs reuse the cache without another Keychain request. Unknown
networks require their own selected-credential authorization; cancellation
does not populate the cache. Invalid, symlinked, or permissive cache files
fail explicitly rather than causing hidden credential retrieval.

Setup now waits for a configured accessory to rejoin its saved network after
reboot before requesting credentials. This addresses an unnecessary request
when Wi-Fi was still connecting.

macOS denied automatic SSID discovery after the native helper was updated.
The user explicitly selected the already-provided network name instead.
One scoped, approved credential read populated the real cache; a subsequent
reuse was verified with Keychain access disabled in the test. The password
was not logged. `make forget-wifi-cache` removes only the laptop's cached
network passwords, not device identity or firmware.

Firmware build/formatting and 63 automated tests passed, including cache
scoping, owner-only permissions, directory-permission preservation, cancellation,
and reboot reconnection without a credential request.

## No Implicit Credential Prompts

Following the user's objection to repeated Keychain prompts, normal setup and
capture now use only existing device configuration or cached passwords.
Missing credentials fail explicitly without OS lookup or password entry.
Only `make cache-wifi` / `cache-network --authorize-credentials` may retrieve
a new OS credential. That flag is rejected on other commands.

An already-configured accessory is reused without automatic SSID discovery.
A fresh accessory uses a single cached profile or requires explicit selection
if the cache is ambiguous. No new Keychain authorization was requested during
verification of this change.

## Repeated Throughput Test Without Credential Access

The user requested more testing with no Keychain/password interaction.
Measurements used only the saved device identity and authenticated TLS
endpoint. Keychain, OS Wi-Fi discovery, Windows credential retrieval, and
interactive password APIs were disabled in the test process. No network
configuration or firmware changes were made.

The repeated 1 MiB batch did not complete:

- Module-to-laptop download: 52.461 seconds, 0.160 Mbit/s; integrity passed.
- The following upload timed out before its response; no successful upload
  speed or three-run aggregate was claimed.

The smaller 64 KiB, 4096-byte-block batch completed:

| Run | Module to laptop | Laptop to module |
|---|---:|---:|
| 1 | 0.307 Mbit/s | 0.309 Mbit/s |
| 2 | 0.116 Mbit/s | 0.160 Mbit/s |
| 3 | 0.102 Mbit/s | 0.147 Mbit/s |
| Median | 0.116 Mbit/s | 0.160 Mbit/s |

All six smaller transfers passed byte integrity checks. Signal reports ranged
from -64 to -65 dBm; TLS setup took approximately 1.3-2.6 seconds and was excluded
from the payload rates.

ICMP during the batch: 10/10 replies, 13.081-314.828 ms, average 140.808 ms.
A separate idle sample, with no concurrent transfer/capture: 10/10 replies,
59.646-432.564 ms, average 248.053 ms.

These results are materially worse than the earlier 2.4-3.4 Mbit/s samples.
The link/application transport is highly variable even without camera or
microphone load. At the current medians, the 0.256 Mbit/s raw 16 kHz PCM16
audio requirement alone exceeds the measured available throughput.
This explains why streaming cannot be assumed reliable under these conditions,
but does not identify whether power saving, radio conditions, routing/AP
behavior, or TCP/TLS configuration is the root cause.

No credential request was made, no media was captured, and no secrets were
included in benchmark output.

## Elevated-Module Reception Retest

The user held the module up in midair for better reception. No firmware or
network settings were changed, and all credential/discovery/password APIs
remained disabled in the benchmark process.

Three 128 KiB trials each way passed all integrity checks:

| Direction | Median | Range |
|---|---:|---:|
| Module to laptop | 0.739 Mbit/s | 0.705-0.823 Mbit/s |
| Laptop to module | 0.672 Mbit/s | 0.603-0.771 Mbit/s |

For a matched comparison with the previous 64 KiB test, another three trials
each way used exactly 64 KiB and 4096-byte blocks:

| Direction | Previous median | Elevated median | Elevated range |
|---|---:|---:|---:|
| Module to laptop | 0.116 Mbit/s | 0.641 Mbit/s | 0.494-0.694 Mbit/s |
| Laptop to module | 0.160 Mbit/s | 0.406 Mbit/s | 0.388-0.851 Mbit/s |

All twelve elevated transfers passed integrity verification. Reported signal
improved from approximately -64/-65 dBm to -57..-61 dBm across the retests.
The matched median improvements were approximately 5.5x download and 2.5x
upload. Placement correlates with a meaningful improvement, but this is not a
controlled RF study and does not isolate it from changing interference.

Idle ICMP while elevated: 10/10 replies, 15.739-453.230 ms, average 170.767 ms.
Latency remains variable, and the improved throughput is still below the
earlier highest application-stack samples and the published TCP lab figures.
No camera/microphone capture or Keychain access was performed.

## Official Espressif Benchmark Source

Fetched an unmodified ESP-IDF v5.5.1 `examples/wifi/iperf` snapshot into
`diagnostics/iperf/upstream`, pinned to commit
`fcae32885b0296b32044cb99ecbdc50d98dddb83`.
Each file was verified against its immutable upstream Git blob; provenance,
original paths, and SHA-256 checksums are recorded in `source-manifest.json`.
The source and license are available locally, but it has not been built/flashed.

The official code explicitly disables modem sleep and uses 65535-byte TCP
buffers, plus throughput-oriented aggregation and task settings. This
contrasts with the current prebuilt SDK's 5760-byte TCP buffers and Kidi's
HTTPS benchmark. It requires iperf 2.x, not iperf3.

The example's watchdog/experimental PHY settings are benchmark-only. Before
any replacement of the camera firmware, a full current-flash backup must be
saved and verified so the provisioned identity and credentials can be restored.
The official app can erase NVS on initialization errors; no such operation was
performed while fetching the source.

## Official Espressif Hardware Benchmark (2026-10-05)

The reconnected board was detected as the same ESP32-S3 with 16 MB flash and
8 MB PSRAM. A fresh complete backup was saved. Resetting after verification
changed only NVS; refreshing that partition produced another full image that
passed a complete-flash digest check without booting the application.

The unmodified ESP-IDF v5.5.1 example was temporarily flashed and joined the
cached approved LAN, with all credential/discovery/password APIs disabled.
No camera/audio or TLS workload was present.

Three ten-second TCP receiver trials produced:

| Direction | Trials (decimal Mbit/s) | Median |
|---|---:|---:|
| Module to laptop | 20.00, 19.20, 20.20 | 20.00 Mbit/s |
| Laptop to module | 14.07, 16.74, 16.67 | 16.67 Mbit/s |

The ESP receiver's binary rates were converted to decimal Mbit/s. Six UDP
trials also completed; active-window receiver rates and measurement caveats
are recorded in [the diagnostic results](../../diagnostics/iperf/README.md#hardware-results-2026-10-05).
No validated UDP loss percentage is claimed.

All ten initial pings returned. Average latency was 120.549 ms, dominated by
a 1025.917 ms first reply; the other replies ranged from 10.720 to 36.707 ms.

The session interruption happened before restoration was confirmed. A
restore-only operation then wrote the complete 16 MB backup and verified its
flash digest. A stale saved DHCP address initially caused TLS verification
to fail; trusted USB verified the original device certificate and refreshed
the address without changing credentials or bypassing certificate checks.
Authenticated HTTPS status and a 2048x1536 JPEG (214277 bytes) passed afterward.
Restored RSSI was -51 dBm. No additional benchmarks were run after the user
asked to stop.

Raw throughput is substantially better than previous application-stack
measurements. This does not by itself identify the bottleneck or prove
sustained encrypted audiovisual streaming. SDK/TCP configuration, modem
sleep, TLS/application costs, and radio conditions still differ.

## Video-Only Optimization (2026-10-05)

The user selected video-only streaming as the primary use case, with a
1280x720, nominal 10 fps target. Microphone support was retained as an explicit
diagnostic option, not initialized during video preview. The browser likewise
does not create an AudioContext for video-only sessions.

Changes made:

- Paced camera acquisition on core 1, two framebuffers and bounded frame
  credits. Under backpressure, stale queued frames are recycled so the next
  send starts with a recent image.
- Sensor JPEG quality 16 for video-only, retaining quality 12 for photographs.
- Removed the 1 MiB PSRAM packet staging buffer/full-frame memcpy. Only a
  4 KiB header/prefix is staged in internal RAM; the remainder uses the
  framebuffer directly. USB writes are bounded; HTTPS keeps bulk writes.
- TCP_NODELAY and active-stream-only modem-sleep suppression with restoration.
- All-channel association scan for the saved SSID, sorted by signal, and
  BSSID/channel/power-policy diagnostics. No credential APIs were called.
- Send duration, bytes, pre-send frame age and capture-wait counters; host
  delivered-FPS, receive-gap and byte-rate reporting with an explicit threshold.

Single-buffer capture serialized acquisition/transmission and had shutdown
ownership problems; it was not retained. Sending whole USB frames without
bounded writes produced framing/timeouts, so the final USB path uses 4 KiB
writes. Applying those small chunks to HTTPS added excessive record overhead;
HTTPS bulk writes were retained instead. Pinning the HTTPS handler to core 0
and a test-only AES-128-GCM cipher choice did not establish an improvement;
neither experiment was retained.

The laptop was suspended during some early checks. Those checks were not
treated as clean optimization evidence.

An intermittent USB end-of-stream failure was traced to raw SDK text:
`gdma_disconnect(299): no peripheral is connected to the channel`.
The camera deinit API returned success, but its stdout text appeared before
the binary END packet. Cleanup logs are now captured in a bounded buffer and
forwarded as typed diagnostic status, with explicit truncation reporting.
The message is still surfaced, not silently suppressed or mistaken for JPEG.

Final sustained USB video validation:

- 198 frames in 20 requested seconds; 20.098 seconds receiving time.
- 9.85 delivered fps; 12,306,150 JPEG bytes, 4.90 Mbit/s payload.
- Zero audio samples and zero dropped frames.
- Maximum receive gap 103.7 ms; maximum JPEG send 69.4 ms.
- Real browser JPEG decoding, no AudioContext, Stop/restart and duration expiry
  passed. No media files were saved.

Final Wi-Fi video validation:

- Three JPEGs in ten requested seconds: 0.30 delivered fps.
- Maximum JPEG send 6.42 seconds; maximum pre-send frame age 92.7 ms.
- Zero audio samples; stale images were dropped rather than building a queue.
- The 9.5 fps acceptance check failed. The 720p/10 fps wireless goal remains
  unresolved; no wireless optimization success is claimed.

An explicit Wi-Fi cancellation check verified power-save mode 1 before,
0 while streaming, and 1 afterward. Cleanup completed in 1.64 seconds.
A following authenticated 2048x1536 photo passed. Device identity and cached
credentials were preserved; no Keychain/password prompts were used.

The final firmware builds, formatting passes, and 70 host tests pass.
Application-only flashing required making the esptool Make command explicitly
use Python 3.12 and esptool 5.x; the unpinned command selected an older package
without the expected executable/CLI options.

Core separation is not complete: the prebuilt SDK pins Wi-Fi/TCP-IP and the
camera driver to core 0 even though the application capture worker runs on
core 1. A configurable SDK build is a candidate next step for camera-driver
affinity and larger TCP buffers, not a proven cure. CPU usage/power were not
measured, and differences in SDK, radio conditions, camera load and TLS prevent
treating the official raw iperf result as an HTTPS-video guarantee.

## Factory Restoration for Return (2026-10-05)

The user decided to return the W11 and obtain different hardware. No further
streaming optimization or SDK migration was performed.

The original 16,777,216-byte factory backup was checked against its known
SHA-256:
`f6ea91cf9b8aca76944f649572efc3ca2a2192a6d346e867ef8999e34a8e7f74`.
The complete flash was overwritten at offset zero using `keep` mode/frequency/
size settings, preserving the original bootloader header. A separate full-flash
verification passed its digest check before reset.

This replaced Kidi firmware and the provisioned NVS identity, certificate/private
key, owner-token hash and network configuration with the original factory
contents. No efuses were modified, no credential APIs were accessed, and laptop
backups/captures/credential caches were not deleted.

Normal boot was confirmed with the `W11 Factory Test + Data Output` banner.
IMU, digital microphone, camera, Wi-Fi AP initialization, BLE and ADC passed.
SD-card mounting failed, as in the original diagnosis. The factory camera test
still prints the hardcoded OV5640 label and a GDMA cleanup diagnostic, then
reports PASS; the sensor was previously confirmed as OV3660.

The live preview server was already stopped, and its browser page was cleared.
The module is ready for power disconnection and packing. Personal SD storage
must be removed or separately reviewed: restoring internal flash does not wipe
an SD card. Software source and private local artifacts remain on the laptop;
neither should be included in the seller's return package.

## Remaining work

- Resolve Wi-Fi live AV stalls and validate sustained throughput, stop/reconnect,
  and browser audiovisual playback before claiming usable wireless streaming.
- Production BLE pairing and Android/Kidi integration.
- Windows hardware validation of the laptop setup and Wi-Fi photo workflow.
- Long-running capture, recovery after disconnects, and sustained-load tests.
- Frame-rate benchmarks and throughput improvements at the selected resolutions.
- Tighter AV synchronization measurement if needed.
- SD-card diagnosis or formatting only if separately requested.

## References

- [Meshnology W11 manual, examples, and schematics](https://wiki.meshnology.com/W11/W11_ESP32S3_Mini_Module/).
- [Manufacturer documentation repository](https://github.com/Meshnology/meshnology_doc).
- [OV3660 specification, manufacturer datasheet mirror](https://media.ic-find.com/datasheets/4666f79f51fd79b7cecae867005c0723.pdf).
- [SGM40567 charger datasheet](https://www.sg-micro.com/rect/assets/e95e555a-a8a2-4761-b86a-6554dff2d8ed/SGM40567.pdf).
- [Espressif USB Serial/JTAG driver documentation](https://docs.espressif.com/projects/esp-iot-solution/en/latest/usb/usb_overview/usb_serial_jtag.html).
- [Espressif camera driver and buffering guidance](https://github.com/espressif/esp32-camera).
