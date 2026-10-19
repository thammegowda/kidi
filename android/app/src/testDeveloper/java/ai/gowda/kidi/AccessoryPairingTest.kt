package ai.gowda.kidi

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.MessageDigest
import java.security.Signature
import java.security.interfaces.ECPublicKey
import java.security.spec.ECGenParameterSpec
import java.util.Base64
import javax.crypto.Cipher
import javax.crypto.Mac
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

class AccessoryPairingTest {
    @Test
    fun matchesSharedInteroperabilityVector() {
        val vector = JSONObject(
            requireNotNull(javaClass.classLoader?.getResourceAsStream("accessory-v1-test-vectors.json"))
                .bufferedReader()
                .use { it.readText() },
        )
        val controllerId = vector.getString("controller_id_hex").hex()
        val session = AccessoryPairingSession(
            PairingInvitation(
                deviceId = vector.getString("device_id"),
                deviceName = "Kidi Lab",
                devicePublicKey = vector.getString("device_public_key_base64url").decode(),
                secret = vector.getString("invitation_secret_base64url").decode(),
                expiresAtSeconds = 2_000_000_000L,
            ),
            vector.getString("controller_name"),
            controllerId,
            vector.getString("client_nonce_base64url").decode(),
        )
        assertEquals(vector.getString("request_utf8"), session.request.decodeToString())
        val response = JSONObject()
            .put("v", 1)
            .put("iv", vector.getString("response_iv_base64url"))
            .put("ciphertext", vector.getString("response_ciphertext_base64url"))
            .put("signature", vector.getString("response_signature_der_base64url"))
            .toString()
            .encodeToByteArray()

        val profile = session.acceptResponse(response)

        assertEquals("Kidi Lab", profile.name)
        assertEquals("192.168.4.1", profile.host)
        assertEquals("Kidi-AABBCC", profile.softApSsid)
        assertEquals("AbCdEfGh23456789", profile.softApPassword)
        assertArrayEquals(ByteArray(32) { (it + 160).toByte() }, profile.controllerToken)
    }

    @Test
    fun verifiesSignedEncryptedProfile() {
        val fixture = Fixture()
        val result = fixture.session.acceptResponse(fixture.response())

        assertEquals(fixture.invitation.deviceId, result.deviceId)
        assertEquals("192.168.4.1", result.host)
        assertArrayEquals(ByteArray(32) { 9 }, result.controllerToken)
        assertEquals("kidi-prototype", result.softApSsid)
    }

    @Test
    fun rejectsCiphertextAndSignatureTampering() {
        for (field in listOf("ciphertext", "signature")) {
            val fixture = Fixture()
            val response = JSONObject(fixture.response().decodeToString())
            val bytes = response.getString(field).decode().also { it[it.lastIndex] = (it.last() xor 1) }
            response.put(field, bytes.encode())
            assertThrows(Exception::class.java) {
                fixture.session.acceptResponse(response.toString().encodeToByteArray())
            }
        }
    }

    @Test
    fun requestProofBindsControllerAndInvitation() {
        val fixture = Fixture()
        val request = JSONObject(fixture.session.request.decodeToString())
        assertEquals(fixture.invitation.deviceId, request.getString("device_id"))
        assertEquals(fixture.controllerId.toHex(), request.getString("controller_id"))
        assertEquals("Pixel prototype", request.getString("name"))
        assertEquals(32, request.getString("proof").decode().size)
    }

    private class Fixture {
        private val keyPair: KeyPair = KeyPairGenerator.getInstance("EC").apply {
            initialize(ECGenParameterSpec("secp256r1"))
        }.generateKeyPair()
        private val secret = ByteArray(32) { (it + 2).toByte() }
        val controllerId = ByteArray(16) { (it + 5).toByte() }
        private val nonce = ByteArray(32) { (it + 7).toByte() }
        val invitation = PairingInvitation(
            deviceId = "00112233445566778899aabbccddeeff",
            deviceName = "Kidi accessory",
            devicePublicKey = (keyPair.public as ECPublicKey).uncompressed(),
            secret = secret,
            expiresAtSeconds = 2_000_000_300L,
        )
        val session = AccessoryPairingSession(invitation, "Pixel prototype", controllerId, nonce)

        fun response(): ByteArray {
            val profile = JSONObject()
                .put("v", 1)
                .put("device_id", invitation.deviceId)
                .put("name", invitation.deviceName)
                .put("host", "192.168.4.1")
                .put("control_port", AccessoryProtocol.CONTROL_PORT)
                .put("media_port", AccessoryProtocol.MEDIA_PORT)
                .put("control_certificate_sha256", ByteArray(32) { 1 }.encode())
                .put("media_certificate_sha256", ByteArray(32) { 2 }.encode())
                .put("controller_token", ByteArray(32) { 9 }.encode())
                .put("softap_ssid", "kidi-prototype")
                .put("softap_password", "prototype-password")
                .toString()
                .encodeToByteArray()
            val iv = ByteArray(12) { (it + 11).toByte() }
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(
                Cipher.ENCRYPT_MODE,
                SecretKeySpec(hkdf(secret, nonce, "kidi-pair-response-v1".encodeToByteArray()), "AES"),
                GCMParameterSpec(128, iv),
            )
            cipher.updateAAD(invitation.deviceId.encodeToByteArray() + controllerId)
            val ciphertext = cipher.doFinal(profile)
            val signed = MessageDigest.getInstance("SHA-256").digest(session.request) + iv + ciphertext
            val signature = Signature.getInstance("SHA256withECDSA").run {
                initSign(keyPair.private)
                update(signed)
                sign()
            }
            return JSONObject()
                .put("v", 1)
                .put("iv", iv.encode())
                .put("ciphertext", ciphertext.encode())
                .put("signature", signature.encode())
                .toString()
                .encodeToByteArray()
        }

        private fun ECPublicKey.uncompressed(): ByteArray =
            byteArrayOf(4) + w.affineX.fixed(32) + w.affineY.fixed(32)

        private fun java.math.BigInteger.fixed(size: Int): ByteArray {
            val value = toByteArray().let { if (it.size > size && it[0] == 0.toByte()) it.copyOfRange(1, it.size) else it }
            require(value.size <= size)
            return ByteArray(size - value.size) + value
        }
    }

    private companion object {
        fun hkdf(input: ByteArray, salt: ByteArray, info: ByteArray): ByteArray {
            val extracted = hmac(salt, input)
            return hmac(extracted, info + byteArrayOf(1))
        }

        fun hmac(key: ByteArray, value: ByteArray): ByteArray =
            Mac.getInstance("HmacSHA256").run {
                init(SecretKeySpec(key, "HmacSHA256"))
                doFinal(value)
            }

        fun ByteArray.encode(): String = Base64.getUrlEncoder().withoutPadding().encodeToString(this)
        fun String.decode(): ByteArray = Base64.getUrlDecoder().decode(this + "=".repeat((4 - length % 4) % 4))
        fun String.hex(): ByteArray {
            require(length % 2 == 0)
            return ByteArray(length / 2) { index ->
                substring(index * 2, index * 2 + 2).toInt(16).toByte()
            }
        }
        fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }
        infix fun Byte.xor(value: Int): Byte = (toInt() xor value).toByte()
    }
}
