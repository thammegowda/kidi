package ai.gowda.kidi

import org.json.JSONObject
import java.net.URI
import java.util.Base64
import java.util.UUID

internal enum class CaptureSource {
    PHONE,
    ACCESSORY,
}

internal data class PairingInvitation(
    val deviceId: String,
    val deviceName: String,
    val devicePublicKey: ByteArray,
    val secret: ByteArray,
    val expiresAtSeconds: Long,
)

internal object AccessoryProtocol {
    const val VERSION = 1
    const val MAXIMUM_CONTROLLERS = 4
    const val MAXIMUM_MEDIA_SESSIONS = 1
    const val MAXIMUM_BLE_MESSAGE_BYTES = 2048
    const val CONTROL_PORT = 61443
    const val MEDIA_PORT = 54443
    const val AUDIO_SAMPLE_RATE = 16000
    const val AUDIO_CHANNELS = 1
    const val AUDIO_BITS_PER_SAMPLE = 16
    const val MAXIMUM_AUDIO_SECONDS = 30

    const val PHOTO_PATH = "/v1/photo"
    const val AUDIO_PATH = "/v1/audio"
    const val MEDIA_TICKET_PATH = "/v1/media-ticket"
    const val STATUS_PATH = "/v1/status"
    const val CONTROLLERS_SELF_PATH = "/v1/controllers/self"
    const val PHOTO_DIGEST_HEADER = "X-Kidi-SHA256"
    const val PHOTO_WIDTH_HEADER = "X-Kidi-Width"
    const val PHOTO_HEIGHT_HEADER = "X-Kidi-Height"
    const val AUDIO_SAMPLE_RATE_HEADER = "X-Kidi-Sample-Rate"
    const val AUDIO_CHANNELS_HEADER = "X-Kidi-Channels"
    const val AUDIO_BITS_PER_SAMPLE_HEADER = "X-Kidi-Bits-Per-Sample"

    const val BLE_UUID_NAMESPACE = "DNS"
    const val BLE_UUID_NAMESPACE_UUID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    const val BLE_SERVICE_TAG = "ai.gowda.kidi.accessory.v1.service"
    const val BLE_PAIR_REQUEST_TAG = "ai.gowda.kidi.accessory.v1.pair-request"
    const val BLE_PAIR_RESPONSE_TAG = "ai.gowda.kidi.accessory.v1.pair-response"
    const val BLE_STATUS_TAG = "ai.gowda.kidi.accessory.v1.status"
    val BLE_SERVICE_UUID: UUID = UUID.fromString("9a1f0dbc-a6fe-536d-94d8-5ca19ed84493")
    val BLE_PAIR_REQUEST_UUID: UUID = UUID.fromString("8e68995e-6448-552c-9376-168454089158")
    val BLE_PAIR_RESPONSE_UUID: UUID = UUID.fromString("e9c06854-b576-5a5a-876d-79fa31eb1e52")
    val BLE_STATUS_UUID: UUID = UUID.fromString("d7105927-5e98-5292-8fe4-1fb2cab546b9")

    private val invitationKeys = setOf(
        "v",
        "device_id",
        "device_public_key",
        "invitation",
        "expires_at",
        "ble_service",
        "name",
    )
    private const val maximumClockSkewSeconds = 30L
    private const val maximumInvitationLifetimeSeconds = 15 * 60L

    fun parsePairingUri(value: String, nowSeconds: Long): PairingInvitation {
        val uri = runCatching { URI(value) }.getOrElse { throw IllegalArgumentException("Invalid pairing URI", it) }
        require(uri.scheme == "kidi" && uri.host == "pair" && uri.path == "/v1") {
            "Unsupported pairing URI"
        }
        require(uri.rawQuery == null && uri.rawFragment?.isNotBlank() == true) {
            "Pairing data must be in the URI fragment"
        }
        val payload = decodeBase64Url(uri.rawFragment)
        require(payload.size <= MAXIMUM_BLE_MESSAGE_BYTES) { "Pairing invitation is too large" }
        val objectValue = runCatching { JSONObject(payload.decodeToString()) }
            .getOrElse { throw IllegalArgumentException("Pairing invitation is not JSON", it) }
        require(objectValue.keys().asSequence().toSet() == invitationKeys) {
            "Pairing invitation fields are invalid"
        }
        require(objectValue.getInt("v") == VERSION) { "Unsupported pairing protocol version" }
        require(UUID.fromString(objectValue.getString("ble_service")) == BLE_SERVICE_UUID) {
            "Unexpected pairing BLE service"
        }
        val deviceId = objectValue.getString("device_id")
        require(deviceId.matches(Regex("[0-9a-f]{32}"))) { "Invalid accessory ID" }
        val publicKey = decodeBase64Url(objectValue.getString("device_public_key"))
        require(publicKey.size == 65 && publicKey.first() == 0x04.toByte()) {
            "Accessory identity must be an uncompressed P-256 public key"
        }
        val secret = decodeBase64Url(objectValue.getString("invitation"))
        require(secret.size == 32) { "Pairing invitation secret must be 256 bits" }
        val expiresAt = objectValue.getLong("expires_at")
        require(expiresAt >= nowSeconds - maximumClockSkewSeconds) { "Pairing invitation has expired" }
        require(expiresAt <= nowSeconds + maximumInvitationLifetimeSeconds) {
            "Pairing invitation lifetime is too long"
        }
        val name = objectValue.getString("name")
        require(name.isNotBlank() && name.length <= 64 && name.none(Char::isISOControl)) {
            "Invalid accessory name"
        }
        return PairingInvitation(deviceId, name, publicKey, secret, expiresAt)
    }

    private fun decodeBase64Url(value: String): ByteArray {
        require(value.isNotEmpty() && value.none(Char::isWhitespace)) { "Invalid Base64URL value" }
        require(value.all { it.isLetterOrDigit() || it == '-' || it == '_' }) {
            "Base64URL value must be unpadded"
        }
        val padding = "=".repeat((4 - value.length % 4) % 4)
        return runCatching { Base64.getUrlDecoder().decode(value + padding) }
            .getOrElse { throw IllegalArgumentException("Invalid Base64URL value", it) }
    }
}
