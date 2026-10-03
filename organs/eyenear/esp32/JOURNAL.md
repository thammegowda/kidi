# W11 diagnosis and capture journal

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

## Remaining work

- Wireless simultaneous image/audio streaming and Android/Kidi integration.
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
