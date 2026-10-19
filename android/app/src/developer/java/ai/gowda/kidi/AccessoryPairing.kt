package ai.gowda.kidi

import org.json.JSONObject
import java.math.BigInteger
import java.nio.charset.StandardCharsets
import java.security.AlgorithmParameters
import java.security.KeyFactory
import java.security.MessageDigest
import java.security.SecureRandom
import java.security.Signature
import java.security.spec.ECGenParameterSpec
import java.security.spec.ECParameterSpec
import java.security.spec.ECPoint
import java.security.spec.ECPublicKeySpec
import java.util.Base64
import javax.crypto.Cipher
import javax.crypto.Mac
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

internal class AccessoryPairingSession(
    private val invitation: PairingInvitation,
    val controllerName: String,
    val controllerId: ByteArray = ByteArray(16).also(SecureRandom()::nextBytes),
    private val clientNonce: ByteArray = ByteArray(32).also(SecureRandom()::nextBytes),
) {
    val request: ByteArray

    init {
        require(controllerName.isNotBlank() && controllerName.length <= 64 && controllerName.none(Char::isISOControl)) {
            "Invalid controller name"
        }
        require(controllerId.size == 16 && clientNonce.size == 32) { "Invalid pairing session entropy" }
        val proof = hmac(invitation.secret, requestTranscript())
        request = buildString {
            append("{\"v\":")
            append(AccessoryProtocol.VERSION)
            append(",\"device_id\":")
            append(JSONObject.quote(invitation.deviceId))
            append(",\"controller_id\":")
            append(JSONObject.quote(controllerId.toHex()))
            append(",\"name\":")
            append(JSONObject.quote(controllerName))
            append(",\"nonce\":")
            append(JSONObject.quote(clientNonce.base64Url()))
            append(",\"proof\":")
            append(JSONObject.quote(proof.base64Url()))
            append('}')
        }.encodeToByteArray()
        require(request.size <= AccessoryProtocol.MAXIMUM_BLE_MESSAGE_BYTES) { "Pairing request exceeds BLE bound" }
    }

    fun acceptResponse(response: ByteArray): AccessoryProfile {
        require(response.size in 1..AccessoryProtocol.MAXIMUM_BLE_MESSAGE_BYTES) {
            "Pairing response exceeds BLE bound"
        }
        val envelope = JSONObject(response.decodeToString())
        require(
            envelope.keys().asSequence().toSet() == setOf("v", "iv", "ciphertext", "signature") &&
                envelope.getInt("v") == AccessoryProtocol.VERSION,
        ) { "Invalid pairing response envelope" }
        val iv = envelope.getString("iv").unbase64Url()
        val ciphertext = envelope.getString("ciphertext").unbase64Url()
        val signature = envelope.getString("signature").unbase64Url()
        require(iv.size == 12 && ciphertext.size in 17..MAXIMUM_ENCRYPTED_PROFILE_BYTES) {
            "Invalid encrypted pairing profile"
        }
        val signed = MessageDigest.getInstance("SHA-256").digest(request) + iv + ciphertext
        val verifier = Signature.getInstance("SHA256withECDSA")
        verifier.initVerify(p256PublicKey(invitation.devicePublicKey))
        verifier.update(signed)
        require(verifier.verify(signature)) { "Accessory pairing signature verification failed" }

        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(
            Cipher.DECRYPT_MODE,
            SecretKeySpec(hkdf(invitation.secret, clientNonce, RESPONSE_KEY_INFO), "AES"),
            GCMParameterSpec(128, iv),
        )
        cipher.updateAAD(invitation.deviceId.encodeToByteArray() + controllerId)
        val cleartext = cipher.doFinal(ciphertext)
        require(cleartext.size <= MAXIMUM_PROFILE_BYTES) { "Accessory profile exceeds its bound" }
        val profile = JSONObject(cleartext.decodeToString())
        val expectedKeys = setOf(
            "v",
            "device_id",
            "name",
            "host",
            "control_port",
            "media_port",
            "control_certificate_sha256",
            "media_certificate_sha256",
            "controller_token",
            "softap_ssid",
            "softap_password",
        )
        require(profile.keys().asSequence().toSet() == expectedKeys) { "Accessory profile fields are invalid" }
        require(
            profile.getInt("v") == AccessoryProtocol.VERSION &&
                profile.getString("device_id") == invitation.deviceId,
        ) { "Accessory profile identity changed during pairing" }
        return AccessoryProfile(
            deviceId = invitation.deviceId,
            name = profile.getString("name"),
            host = profile.getString("host"),
            controlPort = profile.getInt("control_port"),
            mediaPort = profile.getInt("media_port"),
            controlCertificateSha256 = profile.getString("control_certificate_sha256").unbase64Url(),
            mediaCertificateSha256 = profile.getString("media_certificate_sha256").unbase64Url(),
            controllerToken = profile.getString("controller_token").unbase64Url(),
            softApSsid = profile.getString("softap_ssid"),
            softApPassword = profile.getString("softap_password"),
        )
    }

    private fun requestTranscript(): ByteArray =
        REQUEST_CONTEXT + invitation.deviceId.encodeToByteArray() + controllerId + clientNonce +
            controllerName.toByteArray(StandardCharsets.UTF_8)

    private fun p256PublicKey(encoded: ByteArray): java.security.PublicKey {
        require(encoded.size == 65 && encoded[0] == 4.toByte()) { "Invalid accessory P-256 public key" }
        val parameters = AlgorithmParameters.getInstance("EC").apply {
            init(ECGenParameterSpec("secp256r1"))
        }.getParameterSpec(ECParameterSpec::class.java)
        val point = ECPoint(
            BigInteger(1, encoded.copyOfRange(1, 33)),
            BigInteger(1, encoded.copyOfRange(33, 65)),
        )
        return KeyFactory.getInstance("EC").generatePublic(ECPublicKeySpec(point, parameters))
    }

    private fun hkdf(input: ByteArray, salt: ByteArray, info: ByteArray): ByteArray {
        val pseudoRandomKey = hmac(salt, input)
        return hmac(pseudoRandomKey, info + byteArrayOf(1))
    }

    private fun hmac(key: ByteArray, value: ByteArray): ByteArray =
        Mac.getInstance("HmacSHA256").run {
            init(SecretKeySpec(key, "HmacSHA256"))
            doFinal(value)
        }

    private fun ByteArray.base64Url(): String = Base64.getUrlEncoder().withoutPadding().encodeToString(this)
    private fun String.unbase64Url(): ByteArray {
        require(isNotEmpty() && none(Char::isWhitespace) && all { it.isLetterOrDigit() || it == '-' || it == '_' }) {
            "Invalid Base64URL value"
        }
        return Base64.getUrlDecoder().decode(this + "=".repeat((4 - length % 4) % 4))
    }

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }

    private companion object {
        val REQUEST_CONTEXT = "kidi-pair-request-v1\u0000".encodeToByteArray()
        val RESPONSE_KEY_INFO = "kidi-pair-response-v1".encodeToByteArray()
        const val MAXIMUM_PROFILE_BYTES = 4096
        const val MAXIMUM_ENCRYPTED_PROFILE_BYTES = MAXIMUM_PROFILE_BYTES + 16
    }
}
