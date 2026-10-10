# Kidi ESP32-P4/C6 firmware

These ESP-IDF projects implement the selected two-chip accessory design from
[`../DESIGN.md`](../DESIGN.md). They are separate from the retained Arduino
compile harness in [`../src/`](../src/).

Kidi-owned product firmware is GNU C++23, built with exceptions and RTTI
disabled and warnings treated as errors. Component headers and implementations
are colocated in one directory. C linkage is limited to ESP-IDF entry points;
the vendored Espressif video helper retains its upstream C layout. The retained
Arduino adapter harness remains C++17 because its GCC 8 toolchain cannot parse
C++23.

## Safety state

Do not flash the C6 project yet. The board's complete P4 factory flash is backed
up and verified, but the separate C6 factory flash is not. First obtain a
restorable C6 image through its UART header or demonstrate and verify a
host-assisted recovery path. Flashing always requires explicit approval.

## Current implementation stage

- `c6/` pins ESP-IDF 5.5 and ESP-Hosted 3.0.9, enables Network Split and host
  deep-sleep wake on C6 GPIO2, starts a per-device WPA2 SoftAP, and exposes the
  version-1 BLE UUIDs with bounded two-byte message framing. It generates and
  persists separate P-256 pairing and control-TLS identities.
- C6 accepts a short-lived physical-USB invitation from P4, validates the
  invitation HMAC, stores at most four controllers in NVS, encrypts the profile
  with HKDF-SHA256/AES-256-GCM and signs it with P-256. Its pinned-TLS control
  service authenticates controller bearer tokens, issues one-use media tickets,
  reports status and supports authenticated self-unpair.
- `p4/` pins the matching Hosted version, validates the Network Split port
  ownership, records normal versus C6 wake boot, and arms the official Hosted
  deep-sleep timer using P4 GPIO6 and C6 reset GPIO54. It targets the verified
  pre-v3/revision-1.3 silicon profile.
- P4 compiles the diagnosed OV5647 RAW10 1280x960 MIPI/ISP/hardware-JPEG path,
  the ES8311 16 kHz mono path, a persistent media-TLS identity and strict
  `/v1/photo` and `/v1/audio` HTTPS handlers. The handlers enforce ticket MAC,
  kind, parameter, expiry and replay checks and one active media session.
- C6 and P4 synchronize the media certificate pin, ticket key, trusted epoch and
  media-session completion over ESP-Hosted peer data. P4 remains fail-closed
  until that synchronization succeeds.
- P4 LP-core supervision, controller-issued invitations/owner revocation and
  hardware integration tests remain incomplete. The board itself continues to
  run the temporary camera diagnostic firmware; none of these images were
  flashed.

For the initial physical-USB bootstrap, the host helper supplies a trusted
current timestamp, prints the deep link and writes an owner-only local SVG QR:

```bash
make pair-invite PORT=auto
```

The underlying P4 UART command is `pair-invite <current-unix-seconds> 300`.
Treat the resulting URI and QR as a short-lived secret.

Builds are intentionally separate and never flash:

```bash
idf.py -C firmware/c6 set-target esp32c6
idf.py -C firmware/c6 build
idf.py -C firmware/p4 set-target esp32p4
idf.py -C firmware/p4 build
```
