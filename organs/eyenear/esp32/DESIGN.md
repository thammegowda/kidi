# Wireless Accessory Design

Status: **design draft with the laptop Wi-Fi-photo development slice implemented**.
Live-preview update: shared acquisition workers and a local viewer are now
implemented. USB live AV is verified; Wi-Fi live AV remains experimental due
to transfer stalls and audio queue overruns. Saved Wi-Fi audio is verified.
The user superseded bounded Wi-Fi video recording with live preview, without
saved video or SD-card caching. This does not establish production streaming
readiness.
Implementation update: the laptop development slice now uses one firmware for
USB diagnostics and authenticated Wi-Fi photos. Automatic photo routing
prefers a saved verified Wi-Fi endpoint and explicitly falls back to USB only
for pre-request unavailability. Certificate/owner failures and ambiguous
interrupted captures fail closed. Production BLE/Android and live Wi-Fi AV
remain design work; the prototype does not imply those are implemented.

This design uses the W11 ESP32-S3 as the first accessory, but the protocol and
`kidi::esp32` firmware structure must support other boards and peripherals.
The first product client is Android on explicitly qualified dual-Wi-Fi devices.
iPhone support is deferred. The first implementation and validation will use
macOS and Windows development clients; Android integration remains out of scope
for that prototype phase, but is the real-use client for version 1.

### Client Scope

| Client | Purpose | Required connection behavior | Current status |
|---|---|---|---|
| macOS laptop | Development, protocol validation, and recovery testing | Automated USB setup; accessory joins the laptop's existing LAN | USB setup and authenticated Wi-Fi photos hardware-tested |
| Windows laptop | The same development workflow | Automated USB setup; accessory joins the laptop's existing LAN | Native adapter implemented and parser-tested; Windows hardware validation pending |
| Qualified Android phone | Real use in version 1 | BLE setup plus accessory Wi-Fi coexisting with internet/Android Auto | Designed; integration and device qualification remain pending |

All three use the same accessory firmware, identity, ownership records,
command schema, TLS authentication, and media framing. Development uses trusted
USB setup and the module's station interface; Android uses BLE setup and the
module's SoftAP. These are connection/setup adapters, not separate media
implementations. Moving ownership between clients must not require reflashing.

## 1. Requirements

- Work at home, in a car, and away from existing Wi-Fi infrastructure.
- Connect to the module's Wi-Fi without replacing the user's internet/car
  connection.
- Use BLE for production Android setup and application-level pairing.
- Automate development setup over USB on macOS and Windows. Do not require
  Ethernet, a second Wi-Fi adapter, manual SSID entry, or manual IP discovery
  for the normal supported development path.
- Allow ownership transfer between qualified clients without reflashing the
  module; keep the ownership protocol platform-neutral for future clients.
- Keep pairing, commands, and recordings local; no cloud account or relay.
- Authenticate the accessory and controller, not just their network addresses.
- Preserve one application protocol across transports rather than implementing
  separate home/car/outdoors workflows.
- Default media profiles remain 2048x1536 stills, 1280x720 video, and 16 kHz
  mono PCM16 speech audio. The user explicitly approved this reduction from
  the earlier 48 kHz diagnostic profile.
- Do not start capture merely because a paired client reconnects.
- Do not integrate Android or change firmware during this design task.

## 2. Android-First Connection Model

Use two complementary transport roles:

1. **BLE:** discovery, pairing, presence, low-bandwidth control and telemetry.
2. **Accessory Wi-Fi plus authenticated TLS:** high-bandwidth media on the
   module's own local-only network.

The ESP32 runs a protected 2.4 GHz SoftAP. The Android client requests a
local-only connection using `WifiNetworkSpecifier`. On a qualified dual-STA
configuration, this coexists with the phone's existing connection instead of
replacing it.

Production Android does not provision home Wi-Fi credentials to the module.
There is no need for it to join the car's network or run an internet gateway.
Development can explicitly provision an optional station profile over USB,
allowing the module to join the laptop's existing LAN.

```text
Home:
  Android -- primary Wi-Fi --> router --> internet
          -- accessory Wi-Fi --> Kidi module
          -- BLE ------------> pairing/control

Car:
  Android -- car Wi-Fi ------> wireless Android Auto
          -- accessory Wi-Fi --> Kidi module
          -- cellular -------> internet
          -- BLE ------------> pairing/control

On the move:
  Android -- accessory Wi-Fi --> Kidi module
          -- cellular -------> internet, if available
          -- BLE ------------> pairing/control
```

"Primary" and "accessory" are logical roles, not interface numbers the app can
assign. With cellular internet and no other Wi-Fi connection, the accessory may
be the phone's only Wi-Fi connection; the same connection workflow still applies.

For Android, the projected car interface is **Android Auto**, not CarPlay.
CarPlay is part of the deferred iPhone scope.

### Qualification, Not an All-Modern-Phones Assumption

Offline AI capability does not prove radio/driver concurrency support.
Android's STA/STA feature is optional even on otherwise powerful phones.

The supported accessory configuration requires:

- Android 12/API 31 or newer for this feature. Kidi's unrelated local-AI
  functions need not lose their existing Android 10 support.
- A positive `WifiManager.isStaConcurrencyForLocalOnlyConnectionsSupported()`
  capability check and the required permissions/user approval.
- An eligible foreground connection request. Background persistence and
  reconnection need separate platform validation; companion association is
  not an exemption from Wi-Fi or background-execution rules.
- Coexistence on the actual channel/band combinations: the ESP32-S3 accessory
  uses 2.4 GHz, while the router/car link may use another band.
- Successful operation with the actual phone, OS build, and head unit.

The published local-only concurrency use case is **internet-providing primary
Wi-Fi plus local-only accessory Wi-Fi**. A wireless Android Auto link is often
itself local-only, with internet on cellular. Two local links plus cellular
therefore need an explicit coexistence test; the capability flag alone is not
a guarantee for the car case.

Do not request an accessory connection on an unsupported configuration and hope
the OS preserves the existing network. Refuse it with
`UNSUPPORTED_WIFI_CONFIGURATION` before an operation that might displace the
primary connection. Unknown car configurations remain unqualified until tested.

If the radio becomes unavailable, return a clear connection/media error.
Retain authenticated BLE control where possible, but do not silently change
networks, downgrade media, or treat BLE as the full video transport.

### Why BLE Alone Is Not the Media Solution

The previous ten-second 720p test produced 1,837,655 bytes of JPEG data:

| Payload | Measured or configured rate |
|---|---:|
| JPEG images, at 3.6 distinct frames/second in that test | 1.470 Mbit/s |
| 48 kHz mono PCM16 | 0.768 Mbit/s |
| Combined media payload | 2.238 Mbit/s |

These figures exclude transport overhead and are scene-dependent. Reliable
cross-platform BLE throughput cannot be assumed to carry this load. Short
recordings or still photos could later use a bounded, slower BLE file-transfer
profile, but that is not equivalent to live video or live 48 kHz audio.

### Scenario Qualification Matrix

| Situation | Control/pairing path | Full media path | Important limit |
|---|---|---|---|
| Home with Wi-Fi internet | BLE | Accessory SoftAP as concurrent local-only link | Primary Wi-Fi and internet must remain unchanged |
| Home/on the move with cellular internet | BLE | Accessory SoftAP | Internet must remain on cellular, not the accessory network |
| Wired Android Auto | BLE | Accessory SoftAP | Validate coexistence with projection over USB |
| Wireless Android Auto | BLE | Accessory SoftAP alongside the car link | Must qualify this two-local-links configuration on real hardware |

Bluetooth control also needs coexistence testing while the car uses Bluetooth
for its own functions. Android Auto is not a network relay for the accessory.
If the phone already uses both STA slots, a third Wi-Fi connection cannot be
assumed available.

### Laptop Development: One Laptop Wi-Fi Connection

```text
                    Existing router / reachable LAN
                      /                       \
             Laptop Wi-Fi                 Module STA
                  |                           |
               internet              Same authenticated media API

Laptop USB ----------------------> setup, power, serial diagnostics
Module SoftAP -------------------> remains the Android connection path
```

The laptop stays on its existing Wi-Fi. The module joins a compatible 2.4 GHz
network on the same reachable LAN. The laptop may use a different router band;
both do not have to share a radio channel if the router bridges those networks.
Ethernet and a second laptop Wi-Fi interface are not prerequisites.

ESP32-S3 supports AP+STA mode. One TLS/media service can serve both module
interfaces with the same identity, credentials, and command handlers. This is
not an internet router or a bridge between the accessory subnet and the LAN.

The module has one radio: its AP and STA share the station's channel when the
station is connected. Do not connect, roam, or change this profile in the middle
of an active media session without an explicit coordinated transition.
Android production does not need the development station profile.

The development LAN must provide a compatible 2.4 GHz network and allow peer
traffic. Guest/client isolation, 5/6-GHz-only networks, and managed enterprise
authentication are not automatically solvable by the setup helper. Report these
limits instead of disconnecting the laptop or trying arbitrary other networks.

## 3. Deliberately Deferred

- iPhone and CarPlay integration.
- Wi-Fi Aware, Wi-Fi Direct, and automatic phone-hotspot creation.
- Simultaneous Wi-Fi media from multiple accessories. Multiple paired devices
  do not imply enough radio interfaces for multiple concurrent SoftAP links.
- Background or unattended capture without a separately approved UX/policy.

The earlier Wi-Fi Aware research remains relevant to future platforms, not to
this baseline. The current Espressif component lists ESP32-S31, not the W11's
ESP32-S3, and secured iPhone compatibility requires a newer SDK. Its
component-wide peer-platform configuration also needs single-firmware
interoperability validation. Do not assume a reflash alone solves future
iPhone connectivity.

## 4. Architecture Without Scenario-Specific Branches

```text
macOS / Windows development client or Android version-1 client
  Accessory client
    Identity and credential store
    Connection coordinator
      BLE transport
      IP/TLS transport
    Shared commands, events, and media decoder

ESP32 accessory
  Identity and ownership manager
  Authenticated session / capture lease
  Command dispatcher
    Camera
    Audio
    Other advertised capabilities
  Shared acquisition workers
  BLE transport / IP transport
```

Transport adapters report capabilities and availability. The coordinator
checks the qualified accessory connection, rather than branching on
"home", "car", "Android", or "iPhone".

For example:

- A telemetry read needs an authenticated low-bandwidth route.
- A 720p stream needs an authenticated media-capable route.
- If no suitable route exists, return a specific error and retain the control
  connection.

There is one media implementation and two configured network attachments:
the accessory AP for Android, and an optional station interface for development.
Endpoints are returned by the authorized setup path. Connection/setup adapters
do not duplicate camera, audio, authentication, or session logic.

Pairing and connections are separate state machines:

```text
UNOWNED -> PAIRING_WINDOW -> OWNED
OWNED -> TRANSFER_PENDING -> UNOWNED

DISCONNECTED -> DISCOVERING -> AUTHENTICATING
  -> CONTROL_READY -> MEDIA_READY -> CAPTURING
```

An unavailable media route does not imply the accessory is unpaired. A
connection failure must not erase ownership or request a firmware reflash.

## 5. Identity, Pairing, and Ownership Transfer

### Stable Identity

- Generate a device identity and device TLS key once; do not identify trust by
  MAC address, IP address, SSID, or a friendly name.
- Bind the trusted device certificate/public key through physically authorized
  BLE enrollment for Android or the trusted development USB connection.
- Keep accessory AP credentials, development station profiles, and client ownership records in mutable configuration,
  separate from firmware.
- Discover minimally before authentication; expose full capabilities afterward.

### Controller Credentials

For the IP prototype, use a unique, random, per-controller bearer credential
over certificate-verified TLS. Store only its hash on the device where
practical. Never place the credential in URLs, advertisements, source files,
or logs.

The laptop stores its credential in a private, Git-ignored file with restrictive
permissions. The future Android client protects credentials using platform
secure storage. The ownership format remains platform-neutral for later clients.

Initially permit one owner/controller. Multiple simultaneous owners are not a
requirement for the first prototype.

### Production Android Pairing Ceremony

1. Require physical authorization and open a bounded pairing window.
2. Establish trust in the accessory identity and proof of possession.
3. Issue/register a fresh controller credential.
4. Deliver the accessory's SSID, strong random WPA2-PSK, endpoint information,
   and verified TLS identity through the authenticated enrollment session.
5. Close the pairing window and reject further unauthenticated enrollment.

SSID/BSSID identify which network to request, not which device to trust.
Validate the enrolled TLS identity before sending application credentials or
accepting readings. First Wi-Fi connection still requires Android's system
approval; BLE setup cannot bypass that consent.

For wireless provisioning, use a vetted SDK security implementation with
unique proof of possession and authenticated encryption. Do not invent a
cryptographic handshake, use a universal password, or treat Bluetooth
"Just Works" association as sufficient application authentication.

**SDK gate:** the current ESP-IDF 4.4 protocomm exposes Security 0 and Security 1,
not Security 2. Security 1 uses X25519 and AES-CTR; it is not an AEAD replacement
for Security 2. Select and validate the production BLE provisioning library/SDK
before implementing wireless enrollment. Do not silently downgrade the security
requirement to fit the current framework.

### Unpair and Transfer

- Authorized unpair revokes device-side controller credentials and terminates
  that client's sessions, not merely its phone-side association.
- Lost-phone recovery requires physical authorization on the module.
- Ownership reset removes old client credentials and rotates accessory AP credentials;
  preserve the device identity and firmware.
- Re-pairing the replacement phone issues a new credential. The old phone must
  fail authentication afterward.
- Rebooting must not undo revocation or reopen pairing.

The W11's B button could provide physical authorization while firmware is
already running. GPIO0 is a boot strap: holding it across reset enters the ROM
bootloader, so pairing instructions must not use a BOOT/reset sequence.

Development USB enrollment is an explicit privileged path, not a wireless
authentication bypass. It must not silently replace an existing owner.
Reconfiguration by the same trusted development client is allowed; replacing
another owner requires explicit authorization. Ownership reset also clears
development Wi-Fi profiles so a subsequent user does not inherit those secrets.

## 6. Application Protocol

Use one versioned command/event schema across USB diagnostics, BLE control,
and the IP transport. The existing USB text format remains a diagnostic adapter;
it is not a public unauthenticated network API.

Initial command families:

- `DEVICE_INFO`, `CAPABILITIES`, `STATUS`.
- `SENSOR_SNAPSHOT`.
- `CAPTURE_PHOTO`, `CAPTURE_AUDIO`, `START_AV`, `STOP_CAPTURE`.
- `PAIR`, `UNPAIR`, and network configuration through authorized setup paths.

Every command has a request ID, protocol version, explicit parameters, and a
structured success or error response. Retries of state-changing commands must
not accidentally start another recording.

Authenticated capabilities describe the board/model, supported operations,
image sizes, audio formats, and available sensors. Clients must not assume
every ESP32 accessory has a camera and microphone.

### IP Transport

Proposed baseline:

- Protected accessory SoftAP, using a strong random per-ownership WPA2-PSK.
- Optional station interface for the development LAN, using credentials
  provisioned over trusted USB.
- Endpoint information delivered over authenticated BLE for Android or USB for
  development; LAN discovery is optional convenience, not a prerequisite.
- TLS 1.2 or newer, with device certificate/public-key verification.
- One authenticated WebSocket session for commands, events, and typed binary
  payloads; bounded transfer sizes and one capture owner.

The current SDK includes HTTPS-server and WebSocket support. That is build
capability evidence, not a completed secure-server implementation.

No plaintext media endpoint, global certificate-verification bypass, automatic
trust-on-first-use, or redirect to an unverified host.

Every exposed AP/STA endpoint enforces the same authentication. Being on the
development LAN grants no additional authority.

### Android Routing

Request the exact enrolled accessory network through `WifiNetworkSpecifier`,
without `NET_CAPABILITY_INTERNET`, and retain its `NetworkCallback`.
Create accessory sockets through the returned `Network`/socket factory.
Do not call `bindProcessToNetwork` to redirect the whole application.

Internet/model-download traffic must continue to use the normal default
network. Hold the accessory request while the session needs it, then release
it cleanly. Treat permission denial, radio unavailability, connection loss, and
OS replacement of a network as explicit state transitions.

### Media Framing and Acquisition

Each binary payload identifies its protocol version, request/session ID, stream
kind, sequence number, board-clock timestamp, format, and bounded byte length.

- Camera payload: native JPEG. The ESP32 does not currently encode H.264.
- Audio payload: little-endian mono PCM16 with declared sample rate/count.
- Telemetry payload: typed readings with units and explicit availability/errors.
- Include a boot/session identifier because the board clock resets on reboot.

The laptop may create MP4 locally using the existing encoder. A high MP4
playback frame rate must not be reported as the source camera frame rate.

**Acquisition change required for live use:** the current diagnostic records
audio concurrently but transfers it after video. A live implementation needs
separate acquisition workers and bounded queues:

- Audio chunks of approximately 20-40 ms, with priority over preview frames.
- A small JPEG queue; drop stale preview frames rather than grow memory without
  bound.
- Explicitly report audio gaps/overruns; do not hide them with fabricated samples.
- Keep command handling responsive while acquisition and transfers run.
- Stop cleanly when the capture lease expires or the owner disconnects.

Do not buffer an entire continuous video in PSRAM or hold a server request
handler for the whole recording.

## 7. Privacy and Security Boundaries

- Capture requires an explicit authenticated request, never automatic reconnect.
- Use a short capture lease renewed by the active client. Losing the owner must
  eventually stop acquisition even if a socket cannot close cleanly.
- Separate connection readiness from recording state in the client UI.
- Default to RAM-only device buffering; save recordings only when requested.
- Do not log credentials, raw media, or full private network profiles.
- Bound clients, handshakes, queues, payloads, and pairing attempts to protect
  the MCU from resource exhaustion.
- Discovery proves reachability, not identity. Authenticate before sending
  credentials or accepting device data.
- Do not require internet access, external analytics, or a relay service.

The W11 RGB LED shares GPIO48 with camera data. It cannot be treated as a
reliable independent recording indicator while camera DMA owns that pin.
Use client indicators in the prototype; a dependable product-level hardware
recording indicator needs a non-conflicting pin/hardware design.

### At-Rest Limitation

Secure Boot and flash encryption were disabled in the diagnosed board.
Preferences/NVS persistence alone is not encrypted secret storage. Physical
flash access is outside the first network prototype's protection boundary.

Production flash/NVS encryption, secure boot, manufacturing credentials, and
recovery must be a separate reviewed hardening step. Do not burn irreversible
eFuses during an exploratory prototype. Once Wi-Fi/ownership credentials are
provisioned, flash backups must continue to be treated as private.

## 8. macOS/Windows Development and Android Version-1 Plan

The laptop USB-setup/HTTPS-photo slice is implemented. The following milestones
also describe broader work not yet implemented, including production BLE,
live media delivery, and Android integration.

### Shared Desktop Client

Use one Python client invoked with `uv run` on both macOS and Windows, sharing
USB setup, TLS/media handling, commands, decoding, and tests. A cross-platform
BLE backend such as Bleak can later validate the Android-style setup separately.

Keep the unavoidable platform details in small adapters:

- **macOS:** current-network discovery and authorized Keychain access, plus
  local-network permission handling.
- **Windows:** Native Wi-Fi profile/credential access with the required OS
  authorization. A later BLE test uses native BLE/WinRT, not assumed WSL
  Bluetooth passthrough.
- **USB diagnostics:** macOS device paths and Windows COM ports are configurable;
  USB is the automated development setup/debug path, as well as flashing/power.
- **Build tooling:** preserve PlatformIO and the `uv` workflow. The existing
  Makefile uses POSIX shell tools, so Windows development needs a documented
  compatible shell/Make environment or the equivalent direct Python commands.

Start validation on a current supported macOS release and Windows 11. Pin exact
minimum OS/library versions when selecting the BLE implementation. Do not claim
desktop support solely because a Python library advertises both backends.

### P0: Automated USB Development Setup

Proposed single command, not yet implemented:

```bash
make dev-connect
```

After the developer plugs in the accessory, the helper should:

1. Detect the compatible USB accessory and query its identity/capabilities.
   Do not configure an arbitrary serial device; multiple candidates require
   explicit selection.
2. Discover the laptop's active Wi-Fi profile using an OS adapter.
3. Obtain the selected profile's credential through authorized OS credential
   access. Never collect unrelated profiles or print the credential.
4. Establish/reuse the development client trust binding over physical USB.
   Do not silently replace an existing owner or reflash for credential changes.
5. Send the development station configuration over USB, with request IDs and
   redacted diagnostic output.
6. Wait for an explicit join result and obtain the module's station endpoint.
7. Verify the expected TLS identity and authenticate a wireless session.
8. Save the private client record with restrictive permissions and report
   readiness without asking the developer to copy IP addresses or Wi-Fi details.

Firmware installation is a separate, explicit build/flash operation. The
helper must not silently overwrite unknown firmware to obtain a setup interface.

OS permission/Keychain/UAC prompts may be unavoidable. Automation is not
authorization to bypass protected credential access. Windows profile APIs can
return an encrypted key without reporting an error when plaintext access is
denied; detect that state rather than transmitting it as a Wi-Fi password.

For supported saved personal-network profiles, setup should be automatic after
any OS consent. If credential retrieval is denied/unavailable, provide one
actionable error and, if the developer chooses, a local private password prompt.
Never ask for the password in chat, put it in shell arguments, or guess it.
Managed/captive/unsupported networks need an approved compatible network, not
an insecure automatic workaround.

This validates the device protocol, not Android dual-STA behavior. The laptop
does not switch Wi-Fi networks or need Ethernet. A phone hotspot may be a
developer-chosen shared network when no LAN is available, provided it supports
2.4 GHz and peer traffic; the helper does not enable or reconfigure it.

### P1: Station Endpoint and Authenticated Media

1. Keep the laptop on its existing Wi-Fi and internet route.
2. Run automated USB setup and obtain the station endpoint.
3. Connect to that endpoint and verify the accessory's TLS identity.
4. Authenticate the controller from a Python client using `uv run`.
5. Exercise status, capabilities, photos, audio, and bounded AV wirelessly.
6. Emulate two controllers to test unpair/re-pair without reflashing.

USB may carry development setup and serial debugging. Capture commands and
media must use Wi-Fi during wireless acceptance tests; a connected USB power/
debug cable must not mask an accidental data-path fallback.

### P2: Production BLE Setup and Accessory AP

Validate the provisioning security SDK, then prototype BLE discovery, physical
authorization, proof of possession, enrollment, AP credential delivery, and
reconnection. Test the same TLS/media service through the AP interface.
Production pairing remains BLE, not the USB development exception.

### P3: Live Media and Failure Recovery

Implement concurrent audio/video delivery, leases, cancellation, backpressure,
and reconnect behavior. Benchmark throughput, latency, power, and temperature
without lowering the requested profile silently.

### Version 1: Qualified Android Client

Only after the laptop validates the protocol:

- Companion-device association, BLE/Wi-Fi permissions, and per-network socket
  binding. OS association does not itself authenticate the application owner.
- Feature-gate on API level and dual-STA support.
- Qualify the actual home and wired/wireless Android Auto configurations,
  including channel combinations and concurrent Bluetooth use.
- Test foreground connection establishment and the chosen background lifecycle
  explicitly rather than assuming an always-running process.

Kidi currently imports image URIs through `ImageStore` and captures 16 kHz
waveforms through `SpeechRecorder`. A future remote input adapter should reuse
those image and speech pipelines; current 16 kHz module audio directly matches
Whisper's sample rate. Higher-rate inputs need proper anti-aliasing resampling.
Transport callbacks must
not directly invoke model inference or duplicate its preprocessing.

## 9. Acceptance Tests

- Run the desktop enrollment, reconnection, capture, and ownership-transfer
  suites on both macOS and Windows.
- Enroll on one desktop client, unpair, and enroll on the other without changing
  the accessory firmware. Old credentials must fail afterward.
- Verify Android version-1 behavior independently; passing desktop tests does
  not prove Android dual-STA or Android Auto coexistence.
- Laptop internet remains usable during discovery, connection, and capture.
- A saved supported Wi-Fi profile is provisioned over USB with one development
  command, without manual SSID/IP copying or another network adapter.
- Credential access denial, encrypted profile keys, incompatible bands, and
  isolated networks produce explicit errors without exposing secrets.
- No secrets appear in console output, command lines, logs, or source files.
- AP and station endpoints use the same identity and reject the same
  unauthorized/revoked clients.
- On Android, accessory SoftAP coexists with the intended internet/car connection.
- Unsupported concurrency is rejected without replacing an existing network.
- The car case proves Android Auto and audio remain operational; a capability
  flag alone does not pass this test.
- Home internet remains on its intended Wi-Fi network, not an unnoticed cellular
  fallback.
- Wrong device certificate or missing/revoked credentials fails closed before
  capture or disclosure of readings.
- Pairing is unavailable outside the authorized window.
- Owner transfer survives reboot; the previous owner cannot reconnect.
- Device identity is stable while IP/BLE addresses may change.
- Default photo decodes to 2048x1536; audio declares and delivers 16 kHz PCM16;
  video frames decode to 1280x720.
- Report actual distinct frames, timestamps, payload throughput, and audio
  sample continuity rather than nominal playback FPS.
- Disconnects, permission denial, client process death, occupied radio interfaces,
  duplicate commands, malformed lengths, and queue overflow produce explicit
  state/error transitions.
- Device RAM usage remains bounded during prolonged streaming.
- Test unavailable media routes without losing BLE control or changing the
  phone's existing network.
- USB diagnostic behavior remains available for recovery.

## 10. Decisions Still Needed

1. Which phone/OS/head-unit combinations pass the coexistence qualification?
2. Which vetted BLE enrollment SDK/security implementation will we ship?
3. Which saved-network credential APIs/permissions can we support on the chosen
   macOS and Windows releases?
4. How will the product provide a non-conflicting recording indicator?

These are product and feasibility decisions, not branches to conceal inside
the transport implementation.

## References

- [Android companion device pairing](https://developer.android.com/develop/connectivity/bluetooth/companion-device-pairing).
- [Android BLE background guidance](https://developer.android.com/develop/connectivity/bluetooth/ble/background).
- [Android Wi-Fi network requests](https://developer.android.com/develop/connectivity/wifi/wifi-bootstrap).
- [Android STA/STA concurrency and hardware requirements](https://source.android.com/docs/core/connect/wifi-sta-sta-concurrency).
- [Android Wi-Fi Aware availability and requirements](https://developer.android.com/develop/connectivity/wifi/wifi-aware).
- [Apple Core Bluetooth](https://developer.apple.com/documentation/corebluetooth).
- [Apple local-network privacy permission](https://developer.apple.com/documentation/bundleresources/information-property-list/nslocalnetworkusagedescription).
- [Apple Wi-Fi Aware](https://developer.apple.com/documentation/wifiaware).
- [Apple WWDC25: Wi-Fi Aware](https://developer.apple.com/videos/play/wwdc2025/228/).
- [Espressif's August 2026 iPhone Wi-Fi Aware guide](https://developer.espressif.com/blog/2026/08/wifi-aware-esp-to-iphone/).
- [Espressif Wi-Fi Aware component README, verified revision](https://github.com/espressif/esp-wifi-apps/blob/ec06a61dd570c6805d817e70aec98011352765a0/components/wifi_aware/README.md).
- [ESP-IDF 4.4 protocomm security capabilities](https://docs.espressif.com/projects/esp-idf/en/v4.4.8/esp32s3/api-reference/provisioning/protocomm.html).
- [ESP32-S3 station/AP coexistence](https://docs.espressif.com/projects/esp-idf/en/v4.4.8/esp32s3/api-guides/wifi.html).
- [Windows Wi-Fi profile credential access and permissions](https://learn.microsoft.com/en-us/windows/win32/api/wlanapi/nf-wlanapi-wlangetprofile).
