import base64
import hashlib
import hmac
import json
import re
import struct
from pathlib import Path
import unittest
import uuid

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "protocol/accessory-v1.json").read_text())
VECTOR = json.loads((ROOT / "protocol/accessory-v1-test-vectors.json").read_text())
HEADER = (ROOT / "firmware/components/kidi_protocol/accessory_protocol.h").read_text()


def macro(name):
    match = re.search(rf"^#define {re.escape(name)} (.+)$", HEADER, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing protocol macro {name}")
    return match.group(1)


def unbase64url(value):
    return base64.urlsafe_b64decode(value + "=" * ((4 - len(value) % 4) % 4))


class AccessoryProtocolTest(unittest.TestCase):
    def test_network_split_ports_and_paths_match_header(self):
        self.assertEqual(
            SPEC["network"]["control_port"],
            int(macro("KIDI_ACCESSORY_CONTROL_PORT").rstrip("U")),
        )
        self.assertEqual(SPEC["network"]["media_port"], int(macro("KIDI_ACCESSORY_MEDIA_PORT").rstrip("U")))
        self.assertIn(SPEC["network"]["control_port"], range(61440, 65536))
        self.assertIn(SPEC["network"]["media_port"], range(49152, 61440))
        self.assertEqual(
            f'"{SPEC["media"]["ticket_path"]}"',
            macro("KIDI_ACCESSORY_MEDIA_TICKET_PATH"),
        )
        self.assertEqual(f'"{SPEC["media"]["photo_path"]}"', macro("KIDI_ACCESSORY_PHOTO_PATH"))
        self.assertEqual(f'"{SPEC["media"]["audio_path"]}"', macro("KIDI_ACCESSORY_AUDIO_PATH"))

    def test_ble_contract_matches_header(self):
        self.assertEqual(SPEC["ble"]["maximum_message_bytes"], int(
            macro("KIDI_ACCESSORY_MAX_BLE_MESSAGE_BYTES").rstrip("U")
        ))
        derivation = SPEC["ble"]["uuid_derivation"]
        self.assertEqual(derivation["algorithm"], "UUIDv5")
        self.assertEqual(derivation["namespace"], "DNS")
        self.assertEqual(uuid.UUID(derivation["namespace_uuid"]), uuid.NAMESPACE_DNS)
        self.assertEqual(
            f'"{derivation["namespace"]}"',
            macro("KIDI_ACCESSORY_BLE_UUID_NAMESPACE"),
        )
        self.assertEqual(
            f'"{derivation["namespace_uuid"]}"',
            macro("KIDI_ACCESSORY_BLE_UUID_NAMESPACE_UUID"),
        )
        for field, uuid_name, tag_name in (
            ("service", "KIDI_ACCESSORY_BLE_SERVICE_UUID", "KIDI_ACCESSORY_BLE_SERVICE_TAG"),
            (
                "pair_request",
                "KIDI_ACCESSORY_BLE_PAIR_REQUEST_UUID",
                "KIDI_ACCESSORY_BLE_PAIR_REQUEST_TAG",
            ),
            (
                "pair_response",
                "KIDI_ACCESSORY_BLE_PAIR_RESPONSE_UUID",
                "KIDI_ACCESSORY_BLE_PAIR_RESPONSE_TAG",
            ),
            ("status", "KIDI_ACCESSORY_BLE_STATUS_UUID", "KIDI_ACCESSORY_BLE_STATUS_TAG"),
        ):
            entry = SPEC["ble"][field]
            self.assertEqual(uuid.uuid5(uuid.NAMESPACE_DNS, entry["tag"]), uuid.UUID(entry["uuid"]))
            self.assertEqual(f'"{entry["tag"]}"', macro(tag_name))
            self.assertEqual(f'"{entry["uuid"]}"', macro(uuid_name))

    def test_media_bounds_match_header(self):
        self.assertEqual(SPEC["media"]["photo_maximum_bytes"], 4 * 1024 * 1024)
        self.assertEqual(macro("KIDI_ACCESSORY_MAX_PHOTO_BYTES"), "(4U * 1024U * 1024U)")
        self.assertEqual(SPEC["media"]["audio_sample_rate"], int(
            macro("KIDI_ACCESSORY_AUDIO_SAMPLE_RATE").rstrip("U")
        ))
        self.assertEqual(SPEC["media"]["audio_channels"], 1)
        self.assertEqual(SPEC["media"]["audio_bits_per_sample"], 16)
        self.assertEqual(SPEC["media"]["maximum_audio_seconds"], int(
            macro("KIDI_ACCESSORY_MAX_AUDIO_SECONDS").rstrip("U")
        ))
        self.assertEqual(SPEC["media"]["ticket_lifetime_seconds"], int(
            macro("KIDI_ACCESSORY_TICKET_LIFETIME_SECONDS").rstrip("U")
        ))

    def test_pairing_and_controller_limits_are_bounded(self):
        self.assertEqual(SPEC["version"], int(macro("KIDI_ACCESSORY_PROTOCOL_VERSION").rstrip("U")))
        self.assertEqual(SPEC["controllers"]["maximum"], int(
            macro("KIDI_ACCESSORY_MAX_CONTROLLERS").rstrip("U")
        ))
        self.assertEqual(SPEC["controllers"]["maximum_active_media_sessions"], 1)
        self.assertEqual(SPEC["controllers"]["controller_token_bytes"], 32)
        self.assertEqual(SPEC["pairing_uri"]["maximum_lifetime_seconds"], 15 * 60)

    def test_pairing_vector_verifies_and_decrypts(self):
        secret = unbase64url(VECTOR["invitation_secret_base64url"])
        nonce = unbase64url(VECTOR["client_nonce_base64url"])
        transcript = bytes.fromhex(VECTOR["request_transcript_hex"])
        self.assertEqual(
            hmac.new(secret, transcript, hashlib.sha256).digest(),
            unbase64url(VECTOR["request_proof_base64url"]),
        )
        request = VECTOR["request_utf8"].encode()
        self.assertEqual(hashlib.sha256(request).hexdigest(), VECTOR["request_sha256_hex"])

        extracted = hmac.new(nonce, secret, hashlib.sha256).digest()
        response_key = hmac.new(
            extracted,
            b"kidi-pair-response-v1\x01",
            hashlib.sha256,
        ).digest()
        self.assertEqual(response_key.hex(), VECTOR["response_key_hex"])
        cleartext = AESGCM(response_key).decrypt(
            unbase64url(VECTOR["response_iv_base64url"]),
            unbase64url(VECTOR["response_ciphertext_base64url"]),
            bytes.fromhex(VECTOR["profile_aad_hex"]),
        )
        self.assertEqual(cleartext.decode(), VECTOR["profile_utf8"])

        public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(),
            unbase64url(VECTOR["device_public_key_base64url"]),
        )
        signed = (
            hashlib.sha256(request).digest()
            + unbase64url(VECTOR["response_iv_base64url"])
            + unbase64url(VECTOR["response_ciphertext_base64url"])
        )
        public_key.verify(
            unbase64url(VECTOR["response_signature_der_base64url"]),
            signed,
            ec.ECDSA(hashes.SHA256()),
        )
        derived = ec.derive_private_key(
            int(VECTOR["device_private_scalar_hex"], 16),
            ec.SECP256R1(),
        ).public_key().public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint,
        )
        self.assertEqual(derived, unbase64url(VECTOR["device_public_key_base64url"]))

    def test_ticket_vector_has_exact_layout_and_hmac(self):
        ticket = unbase64url(VECTOR["ticket_base64url"])
        body, received_mac = ticket[:-32], ticket[-32:]
        self.assertEqual(len(body), SPEC["media"]["ticket_encoding"]["body_bytes"])
        self.assertEqual(len(ticket), SPEC["media"]["ticket_encoding"]["authenticated_bytes"])
        self.assertEqual(body.hex(), VECTOR["ticket_body_hex"])
        self.assertEqual(
            hmac.new(bytes.fromhex(VECTOR["ticket_key_hex"]), body, hashlib.sha256).digest(),
            received_mac,
        )
        self.assertEqual(received_mac.hex(), VECTOR["ticket_hmac_hex"])
        (
            version,
            kind,
            issued_at,
            expires_at,
            nonce,
            controller_id,
            parameter_hash,
        ) = struct.unpack(">BBQQ16s16s32s", body)
        self.assertEqual((version, kind), (1, 1))
        self.assertEqual(issued_at, VECTOR["ticket_issued_at"])
        self.assertEqual(expires_at, VECTOR["ticket_expires_at"])
        self.assertEqual(nonce.hex(), VECTOR["ticket_nonce_hex"])
        self.assertEqual(controller_id.hex(), VECTOR["controller_id_hex"])
        self.assertEqual(
            parameter_hash,
            hashlib.sha256(VECTOR["ticket_parameters_utf8"].encode()).digest(),
        )


if __name__ == "__main__":
    unittest.main()
