# Wireless Accessory Design

Status: **P4/C6 hardware qualified; first Android prototype architecture selected**.
The old module has been returned and its board target/drivers removed.
Historical measurements are under [history/w11/](history/w11/); they are not
validation of the proposed hardware. Current SDK adapters have a pin-free
ESP32-S3 compile check, not a production application.

The selected board is the Waveshare ESP32-P4-Module-DEV-KIT with ESP32-P4 v1.3
and ESP32-C6. Hardware diagnosis confirmed the OV5647 MIPI camera, ES8311
microphone path, C6 SDIO/Wi-Fi and the full 32 MB flash/PSRAM configuration.
The complete factory flash is privately backed up and verified.

The official module schematic connects C6 GPIO2 to P4 GPIO6 and P4 GPIO54 to
C6 `CHIP_PU`. The first pair is exactly Espressif Hosted's default deep-sleep
wake wiring; the second lets P4 reset C6. Therefore Network Split plus Host
Power Save is physically viable without an external jumper.

The current diagnostic firmware is not a Kidi product image. A matched C6/P4
firmware pair and Android integration remain to be implemented.

The retained protocol and `kidi::esp32` components must support other boards
and peripherals. Certificate/owner failures and interrupted captures fail
closed. The first product client is Android; macOS/Windows remain development
and recovery clients. iPhone support is deferred.

## Selected prototype architecture

### Two-chip power and network split

```text
Android
  ├─ BLE pairing/control ──────────────────────────────┐
  └─ local-only Wi-Fi + TLS ──────────────────────────┤
                                                       ▼
ESP32-C6 (always-on controller)
  - BLE GATT pairing and invitations
  - protected SoftAP
  - controller registry (maximum 4)
  - low-port TLS control service: 61443
  - Network Split, DHCP/ARP/ICMP while P4 sleeps
  - one-active-media-session arbitration
  - GPIO2 host wake output
                         │ GPIO2 -> GPIO6
                         ▼
ESP32-P4 LP core (sleep supervisor)
  - wake debounce/reason/counters in retained memory
  - HP-core wake request and idle policy
                         │
                         ▼
ESP32-P4 HP cores (on-demand media host)
  - host-owned TLS media service: 54443
  - OV5647 MIPI/ISP/JPEG photo capture
  - ES8311 16 kHz mono PCM16 audio
  - sleep after bounded idle timeout
```

Espressif Hosted v3 Network Split assigns CP-owned ports `61440..65535` and
host-owned ports `49152..61439`. The selected ports intentionally fit those
ranges. Traffic to the media port causes C6 to wake P4 over GPIO2/GPIO6. The
C6 and P4 Hosted components and C6 slave image are one pinned compatibility
unit; never update only one side.

The LP core supplements the hardware GPIO wake source. It records wake cause,
debounces the line, retains counters and supervises the HP idle timeout. Camera,
ISP, JPEG and audio capture always run on HP resources.

### Controller and ownership model

- At most **four paired controllers** are stored on C6.
- Exactly one media session can be active.
- The first controller becomes the owner/administrator.
- The owner can issue one-time invitations and revoke other controllers.
- Any controller can unpair itself.
- Deleting the owner requires owner authorization and explicit successor
  selection, or a trusted USB recovery reset.
- Controller switching does not require reflashing or replacing the device
  identity.

### Pairing bootstrap

The displayless accessory does not use BLE "Just Works" as its trust anchor.

1. Trusted USB setup or an existing owner creates a random, single-use,
   short-lived invitation.
2. The invitation is represented as
   `kidi://pair/v1#<base64url-canonical-json>` and displayed as a QR code.
   The fragment contains protocol version, device ID, device P-256 public key,
   256-bit invitation secret, expiry and BLE service UUID.
3. Android's system QR scanner opens Kidi through an app link; Kidi does not
   require an embedded QR library.
4. Android connects to the matching BLE advertisement and sends a controller
   public key, nonce and HMAC-authenticated request.
5. C6 consumes the invitation and returns AES-256-GCM-encrypted SoftAP
   credentials, controller token, TLS pins and media/control ports.
6. Android stores controller secrets encrypted by an Android Keystore key.

Invitation secrets are never logged, placed in command-line arguments or reused.
BLE transport encryption is defense-in-depth; the application transcript is
authenticated independently.

BLE identifiers are UUIDv5 values derived directly from their semantic tags
using the standard DNS namespace
`6ba7b810-9dad-11d1-80b4-00c04fd430c8`:

| Surface | Tag | UUID |
|---|---|---|
| Service | `ai.gowda.kidi.accessory.v1.service` | `9a1f0dbc-a6fe-536d-94d8-5ca19ed84493` |
| Pair request (write) | `ai.gowda.kidi.accessory.v1.pair-request` | `8e68995e-6448-552c-9376-168454089158` |
| Pair response (indicate) | `ai.gowda.kidi.accessory.v1.pair-response` | `e9c06854-b576-5a5a-876d-79fa31eb1e52` |
| Status (read/notify) | `ai.gowda.kidi.accessory.v1.status` | `d7105927-5e98-5292-8fe4-1fb2cab546b9` |

Pair messages use a two-byte big-endian total length followed by canonical JSON
and are fragmented across GATT writes/indications at `MTU - 3`. Messages are
bounded to 2048 bytes. Invalid, out-of-order, expired or replayed messages fail
without changing ownership.

### Media request flow

1. Android requests the accessory SoftAP with `WifiNetworkSpecifier`.
2. Accessory sockets are created through the returned Android `Network`; the
   process default network is not rebound, so normal internet remains unchanged.
3. Android authenticates to C6 `:61443` and requests a single-use media ticket.
4. C6 checks controller authorization/session availability, arms the wake flow
   and returns a short-lived HMAC-authenticated ticket.
5. Android connects to P4 `:54443`; Network Split wakes P4 if necessary.
6. P4 receives the current controller registry/ticket key over the Hosted
   control channel, validates the ticket, captures media and responds.
7. After the response and idle grace period, P4 shuts down media peripherals
   and returns HP cores to deep sleep.

First prototype media endpoints:

| Endpoint | Request | Response |
|---|---|---|
| `POST /v1/photo` | JSON profile, maximum bytes | `image/jpeg` with actual dimensions and SHA-256 headers |
| `POST /v1/audio` | sample rate 16000, mono PCM16, maximum 30 seconds | `application/vnd.kidi.pcm16`, chunked little-endian samples; closing the socket stops capture |
| `POST /v1/media-ticket` (C6) | media kind/profile | one-time ticket, wake/session state and P4 readiness |
| `GET /v1/status` (C6) | none | paired controller/session/radio/P4 power state; no secrets |

Photo and audio endpoints are mutually exclusive in version 1. Responses never
silently retry a second capture after an ambiguous disconnect.

Controller requests use `Authorization: Bearer <controller-token>`, where the
token is the 32-byte per-controller secret encoded as unpadded Base64URL. Media
requests use `Authorization: Ticket <ticket>`. Tickets expire after 10 seconds,
are valid for exactly one request, and bind the controller, media kind, request
parameters and expiry under HMAC-SHA256. A photo response must include its exact
`Content-Length`, lowercase hexadecimal `X-Kidi-SHA256`, `X-Kidi-Width` and
`X-Kidi-Height`; audio declares its sample rate, channel count and sample width
in `X-Kidi-*` headers before sending little-endian samples.

### Android composer integration

Existing photo and microphone actions gain a small remembered source selector:
**Phone** or **Accessory**.

- Accessory JPEGs pass through the existing `ImageStore` import/size/color
  normalization path and become ordinary pending chat attachments.
- Accessory PCM chunks feed the existing speech activity/draft/final
  transcription pipeline; they never request Android's microphone permission.
- Phone paths retain their current behavior.
- Pairing/network/media errors are surfaced in the existing snackbar/status
  state and never fall back to a phone sensor after accessory capture starts.

Android 12+ is required for qualified concurrent local-only Wi-Fi use.
`BLUETOOTH_SCAN`, `BLUETOOTH_CONNECT`, `NEARBY_WIFI_DEVICES`,
`ACCESS_WIFI_STATE` and `CHANGE_WIFI_STATE` are requested as applicable.
Android 17/target SDK 37 adds runtime `ACCESS_LOCAL_NETWORK`; target 36 remains
in Android 16's opt-in transition but the client design must be ready for it.

### Client Scope

| Client | Purpose | Required connection behavior | Current status |
|---|---|---|---|
| macOS laptop | Development, protocol validation, and recovery testing | Automated USB setup; accessory joins the laptop's existing LAN | Retained tools tested on previous hardware; new board pending |
| Windows laptop | The same development workflow | Automated USB setup; accessory joins the laptop's existing LAN | Native adapter implemented and parser-tested; Windows hardware validation pending |
| Qualified Android phone | Real use in version 1 | BLE setup plus accessory Wi-Fi coexisting with internet/Android Auto | Designed; integration and device qualification remain pending |

All three should use the same host application protocol, identity, ownership
records and TLS authentication. The P4 application and C6 co-processor firmware
are separate chip images, not separate firmware branches for each client.
Development uses trusted USB plus station mode; Android uses BLE setup and
SoftAP. Moving ownership between clients must not require reflashing.

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
- The retained reference profiles are JPEG stills, 1280x720 JPEG video and
  16 kHz mono PCM16. New hardware must explicitly advertise supported sizes
  and codecs; the old 2048x1536 still default is not a P4 ISP guarantee.
- H.264 requires negotiated codec metadata, framing and client decoding.
  Preserve JPEG/PCM interoperability; never relabel raw/H.264 bytes as JPEG.
- Do not start capture merely because a paired client reconnects.
- Android integration and production BLE pairing remain separate tasks.

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
- Coexistence on the actual channel/band combinations: the ESP32-C6 radio
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

The C6 Wi-Fi/hosted stack must be configured and qualified for AP+STA mode.
One host TLS/media service should serve both interfaces with the same identity
and handlers. This is not an internet router or a subnet bridge.

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

The earlier Wi-Fi Aware research is not a capability claim for the proposed
C6 radio. Verify the exact SoC, hosted stack, SDK and peer-platform requirements
before adding it. Do not assume a reflash alone solves iPhone connectivity.

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

A board-specific physical action should authorize recovery while firmware is
running. Determine boot-strap pins from the selected board's schematic; pairing
instructions must not repurpose a BOOT/reset sequence.

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

A dependable recording indicator needs a non-conflicting board pin/hardware
design. Do not assume an onboard LED is independent of camera, SDIO, USB or
audio signals. Use client indicators until the physical design is verified.

### At-Rest Limitation

Do not assume Secure Boot, flash or NVS encryption is enabled on a new board.
Preferences/NVS persistence alone is not encrypted secret storage. Physical
flash access remains outside the reference network prototype's boundary.

Production flash/NVS encryption, secure boot, manufacturing credentials, and
recovery must be a separate reviewed hardening step. Do not burn irreversible
eFuses during an exploratory prototype. Once Wi-Fi/ownership credentials are
provisioned, flash backups must continue to be treated as private.

## 8. macOS/Windows Development and Android Version-1 Plan

Laptop setup/HTTPS/media components are retained from the previous prototype.
The following milestones require revalidation on the P4/C6 hardware; production
BLE, the new camera/audio backends and Android integration are not implemented.

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
- Photos/video decode to the requested advertised size and codec; audio
  declares and delivers the negotiated sample format. Unsupported modes fail
  before capture rather than silently using a different resolution or codec.
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
5. Which exact P4/C6 board revision, hosted link, camera cable/power setup and
   audio codec will be used, and which SDK components support them?

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
- [ESP-Hosted co-processor integration](https://github.com/espressif/esp-hosted-mcu).
- [Espressif camera/ISP sensor support](https://github.com/espressif/esp-video-components/tree/master/esp_cam_sensor).
- [Windows Wi-Fi profile credential access and permissions](https://learn.microsoft.com/en-us/windows/win32/api/wlanapi/nf-wlanapi-wlangetprofile).
