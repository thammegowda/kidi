import base64
import json
import time
import unittest
import uuid

import pair


def invitation_uri(expires_at):
    payload = {
        "v": 1,
        "device_id": "00112233445566778899aabbccddeeff",
        "device_public_key": base64.urlsafe_b64encode(b"\x04" + bytes(64))
        .rstrip(b"=")
        .decode(),
        "invitation": base64.urlsafe_b64encode(bytes(32)).rstrip(b"=").decode(),
        "expires_at": expires_at,
        "ble_service": pair.BLE_SERVICE_UUID,
        "name": "Kidi accessory",
    }
    encoded = (
        base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode(),
        )
        .rstrip(b"=")
        .decode()
    )
    return "kidi://pair/v1#" + encoded


class PairTest(unittest.TestCase):
    def test_service_uuid_is_derived_from_its_tag(self):
        self.assertEqual(
            uuid.uuid5(uuid.NAMESPACE_DNS, pair.BLE_SERVICE_TAG),
            uuid.UUID(pair.BLE_SERVICE_UUID),
        )

    def test_accepts_bounded_invitation(self):
        now = int(time.time())
        payload = pair.validate_uri(invitation_uri(now + 300), now)
        self.assertEqual(payload["name"], "Kidi accessory")

    def test_rejects_expired_and_overlong_invitations(self):
        now = int(time.time())
        for expiry in (now - 1, now + pair.MAXIMUM_INVITATION_SECONDS + 1):
            with self.subTest(expiry=expiry), self.assertRaises(ValueError):
                pair.validate_uri(invitation_uri(expiry), now)


if __name__ == "__main__":
    unittest.main()
