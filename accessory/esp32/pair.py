import argparse
import base64
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import segno

import wifi


PAIRING_PREFIX = b"KIDI_PAIRING_URI "
MAXIMUM_INVITATION_SECONDS = 15 * 60
BLE_SERVICE_TAG = "ai.gowda.kidi.accessory.v1.service"
BLE_SERVICE_UUID = "9a1f0dbc-a6fe-536d-94d8-5ca19ed84493"


def decode_unpadded_base64url(value):
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid unpadded Base64URL")
    return base64.urlsafe_b64decode(value + "=" * ((4 - len(value) % 4) % 4))


def validate_uri(uri, now):
    parsed = urlsplit(uri)
    if (parsed.scheme, parsed.netloc, parsed.path, parsed.query) != (
        "kidi",
        "pair",
        "/v1",
        "",
    ):
        raise ValueError("unexpected pairing URI")
    payload = json.loads(decode_unpadded_base64url(parsed.fragment))
    expected = {
        "v",
        "device_id",
        "device_public_key",
        "invitation",
        "expires_at",
        "ble_service",
        "name",
    }
    if set(payload) != expected or payload["v"] != 1:
        raise ValueError("unexpected pairing payload")
    if not re.fullmatch(r"[0-9a-f]{32}", payload["device_id"]):
        raise ValueError("invalid device ID")
    public_key = decode_unpadded_base64url(payload["device_public_key"])
    if len(public_key) != 65 or public_key[0] != 4:
        raise ValueError("invalid device public key")
    if len(decode_unpadded_base64url(payload["invitation"])) != 32:
        raise ValueError("invalid invitation secret")
    if (
        payload["ble_service"] != BLE_SERVICE_UUID
        or not isinstance(payload["name"], str)
        or not 1 <= len(payload["name"]) <= 64
        or any(ord(character) < 32 or ord(character) == 127 for character in payload["name"])
    ):
        raise ValueError("invalid accessory metadata")
    if (
        not isinstance(payload["expires_at"], int)
        or not now <= payload["expires_at"] <= now + MAXIMUM_INVITATION_SECONDS
    ):
        raise ValueError("invalid invitation expiry")
    return payload


def request_uri(port, lifetime, timeout):
    device = wifi.open_serial(wifi.choose_port(port))
    try:
        deadline = time.monotonic() + timeout
        next_request = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() >= next_request:
                now = int(time.time())
                device.write(f"pair-invite {now} {lifetime}\n".encode())
                device.flush()
                next_request = time.monotonic() + 5
            line = device.readline()
            marker = line.find(PAIRING_PREFIX)
            if marker >= 0:
                uri = line[marker + len(PAIRING_PREFIX) :].strip().decode()
                validate_uri(uri, int(time.time()))
                return uri
        raise TimeoutError("Timed out waiting for the P4 pairing URI")
    finally:
        device.close()


def save_qr(uri, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = os.umask(0o077)
    try:
        segno.make(uri, error="m").save(output, kind="svg", scale=8)
    finally:
        os.umask(previous)
    output.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(
        description="Request a one-time Kidi accessory invitation over physical USB",
    )
    parser.add_argument("--port", default="auto")
    parser.add_argument("--lifetime", type=int, default=300, choices=range(1, 901))
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    uri = request_uri(arguments.port, arguments.lifetime, arguments.timeout)
    output = arguments.output or (
        f"captures/pairing-invitation-{time.strftime('%Y%m%d-%H%M%S')}.svg"
    )
    save_qr(uri, output)
    print(uri)
    print(f"QR saved owner-only at {Path(output).resolve()}")


if __name__ == "__main__":
    main()
