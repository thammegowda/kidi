package ai.gowda.kidi

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Base64
import java.util.UUID

class AccessoryProtocolTest {
    @Test
    fun parsesStrictPairingInvitation() {
        val invitation = AccessoryProtocol.parsePairingUri(uri(), NOW)

        assertEquals("00112233445566778899aabbccddeeff", invitation.deviceId)
        assertEquals("Kidi prototype", invitation.deviceName)
        assertEquals(65, invitation.devicePublicKey.size)
        assertArrayEquals(ByteArray(32) { it.toByte() }, invitation.secret)
        assertEquals(2_000_000_300L, invitation.expiresAtSeconds)
    }

    @Test
    fun rejectsExpiredOrLongLivedInvitation() {
        for (expiry in listOf(NOW - 31, NOW + 901)) {
            val error = assertThrows(IllegalArgumentException::class.java) {
                AccessoryProtocol.parsePairingUri(uri(expiresAt = expiry), NOW)
            }
            assertTrue(error.message!!.contains(if (expiry < NOW) "expired" else "too long"))
        }
    }

    @Test
    fun rejectsWrongServiceUnknownFieldsAndMalformedKeys() {
        val wrongService = payload().put("ble_service", "00000000-0000-0000-0000-000000000000")
        val unknown = payload().put("unexpected", true)
        val shortKey = payload().put("device_public_key", encode(ByteArray(64)))
        for (objectValue in listOf(wrongService, unknown, shortKey)) {
            assertThrows(IllegalArgumentException::class.java) {
                AccessoryProtocol.parsePairingUri(uri(objectValue), NOW)
            }
        }
    }

    @Test
    fun usesHostedNetworkSplitPortRanges() {
        assertTrue(AccessoryProtocol.CONTROL_PORT in 61440..65535)
        assertTrue(AccessoryProtocol.MEDIA_PORT in 49152..61439)
        assertEquals(4, AccessoryProtocol.MAXIMUM_CONTROLLERS)
        assertEquals(1, AccessoryProtocol.MAXIMUM_MEDIA_SESSIONS)
    }

    @Test
    fun bleUuidsAreDerivedFromDocumentedTags() {
        val dnsNamespace = UUID.fromString(AccessoryProtocol.BLE_UUID_NAMESPACE_UUID)
        assertEquals("DNS", AccessoryProtocol.BLE_UUID_NAMESPACE)
        assertEquals(UUID.fromString("6ba7b810-9dad-11d1-80b4-00c04fd430c8"), dnsNamespace)
        for ((tag, expected) in listOf(
            AccessoryProtocol.BLE_SERVICE_TAG to AccessoryProtocol.BLE_SERVICE_UUID,
            AccessoryProtocol.BLE_PAIR_REQUEST_TAG to AccessoryProtocol.BLE_PAIR_REQUEST_UUID,
            AccessoryProtocol.BLE_PAIR_RESPONSE_TAG to AccessoryProtocol.BLE_PAIR_RESPONSE_UUID,
            AccessoryProtocol.BLE_STATUS_TAG to AccessoryProtocol.BLE_STATUS_UUID,
        )) {
            assertTrue(tag.startsWith("ai.gowda.kidi.accessory.v1."))
            assertEquals(expected, uuidV5(dnsNamespace, tag))
        }
    }

    private fun uri(expiresAt: Long) = uri(payload().put("expires_at", expiresAt))

    private fun uri(objectValue: JSONObject = payload()): String =
        "kidi://pair/v1#${encode(objectValue.toString().encodeToByteArray())}"

    private fun payload() = JSONObject()
        .put("ble_service", AccessoryProtocol.BLE_SERVICE_UUID.toString())
        .put("device_id", "00112233445566778899aabbccddeeff")
        .put("device_public_key", encode(byteArrayOf(4) + ByteArray(64) { (it + 1).toByte() }))
        .put("expires_at", 2_000_000_300L)
        .put("invitation", encode(ByteArray(32) { it.toByte() }))
        .put("name", "Kidi prototype")
        .put("v", 1)

    private fun encode(value: ByteArray): String = Base64.getUrlEncoder().withoutPadding().encodeToString(value)

    private fun uuidV5(namespace: UUID, tag: String): UUID {
        val namespaceBytes = ByteBuffer.allocate(16)
            .putLong(namespace.mostSignificantBits)
            .putLong(namespace.leastSignificantBits)
            .array()
        val digest = MessageDigest.getInstance("SHA-1").apply {
            update(namespaceBytes)
            update(tag.toByteArray(StandardCharsets.UTF_8))
        }.digest()
        digest[6] = ((digest[6].toInt() and 0x0f) or 0x50).toByte()
        digest[8] = ((digest[8].toInt() and 0x3f) or 0x80).toByte()
        return ByteBuffer.wrap(digest).let { UUID(it.long, it.long) }
    }

    private companion object {
        const val NOW = 2_000_000_000L
    }
}
