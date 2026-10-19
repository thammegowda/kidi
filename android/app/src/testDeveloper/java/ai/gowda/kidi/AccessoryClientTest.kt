package ai.gowda.kidi

import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import okhttp3.tls.HandshakeCertificates
import okhttp3.tls.HeldCertificate
import okio.Buffer
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import javax.net.SocketFactory
import javax.net.ssl.SSLHandshakeException

class AccessoryClientTest {
    private lateinit var controlServer: MockWebServer
    private lateinit var mediaServer: MockWebServer
    private lateinit var certificate: HeldCertificate
    private lateinit var profile: AccessoryProfile
    private val token = ByteArray(32) { (it + 1).toByte() }
    private val ticket = "test-ticket-abcdefghijklmnopqrstuvwxyz"
    private val jpeg = byteArrayOf(0xff.toByte(), 0xd8.toByte(), 1, 2, 3, 0xff.toByte(), 0xd9.toByte())
    private val pcm = byteArrayOf(1, 0, 0xfe.toByte(), 0xff.toByte(), 0xff.toByte(), 0x7f, 0, 0x80.toByte())

    @Before
    fun setUp() {
        certificate = HeldCertificate.Builder()
            .commonName("localhost")
            .addSubjectAlternativeName("localhost")
            .build()
        val handshake = HandshakeCertificates.Builder().heldCertificate(certificate).build()
        controlServer = MockWebServer().apply {
            useHttps(handshake.sslSocketFactory(), false)
            dispatcher = ControlDispatcher()
            start(AccessoryProtocol.CONTROL_PORT)
        }
        mediaServer = MockWebServer().apply {
            useHttps(handshake.sslSocketFactory(), false)
            dispatcher = MediaDispatcher()
            start(AccessoryProtocol.MEDIA_PORT)
        }
        val pin = MessageDigest.getInstance("SHA-256").digest(certificate.certificate.encoded)
        profile = AccessoryProfile(
            deviceId = "00112233445566778899aabbccddeeff",
            name = "Test accessory",
            host = "127.0.0.1",
            controlPort = controlServer.port,
            mediaPort = mediaServer.port,
            controlCertificateSha256 = pin,
            mediaCertificateSha256 = pin,
            controllerToken = token,
            softApSsid = "kidi-test",
            softApPassword = "not-a-real-secret",
        )
    }

    @After
    fun tearDown() {
        controlServer.shutdown()
        mediaServer.shutdown()
    }

    @Test
    fun obtainsTicketAndValidatesPhotoBytesAndDigest() = runBlocking {
        val result = client().capturePhoto("chat")

        assertArrayEquals(jpeg, result)
        assertEquals(AccessoryProtocol.MEDIA_TICKET_PATH, controlServer.takeRequest().requestUrl!!.encodedPath)
        assertEquals(AccessoryProtocol.PHOTO_PATH, mediaServer.takeRequest().requestUrl!!.encodedPath)
    }

    @Test
    fun streamsLittleEndianPcmChunks() = runBlocking {
        val samples = mutableListOf<Short>()
        val format = client().streamAudio(maximumSeconds = 1) { samples += it.toList() }

        assertEquals(AccessoryAudioFormat(16000, 1, 16), format)
        assertEquals(listOf(1, -2, 32767, -32768).map(Int::toShort), samples)
    }

    @Test
    fun refusesWrongCertificatePinBeforeMediaDisclosure() {
        val wrong = profile.copy(controlCertificateSha256 = ByteArray(32) { 7 })

        assertThrows(SSLHandshakeException::class.java) {
            runBlocking { AccessoryClient(wrong, SocketFactory.getDefault()).capturePhoto() }
        }
    }

    @Test
    fun refusesPhotoDigestMismatch() {
        mediaServer.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse =
                MockResponse()
                    .setHeader("Content-Type", "image/jpeg")
                    .setHeader("X-Kidi-Width", "640")
                    .setHeader("X-Kidi-Height", "480")
                    .setHeader("X-Kidi-SHA256", "0".repeat(64))
                    .setBody(Buffer().write(jpeg))
        }

        val error = assertThrows(IllegalArgumentException::class.java) {
            runBlocking { client().capturePhoto() }
        }
        assertTrue(error.message!!.contains("digest mismatch"))
    }

    @Test
    fun unpairsThroughAuthenticatedControlEndpoint() = runBlocking {
        client().unpair()

        assertEquals(
            AccessoryProtocol.CONTROLLERS_SELF_PATH,
            controlServer.takeRequest().requestUrl!!.encodedPath,
        )
    }

    private fun client() = AccessoryClient(profile, SocketFactory.getDefault())

    private inner class ControlDispatcher : Dispatcher() {
        override fun dispatch(request: RecordedRequest): MockResponse {
            if (request.path == AccessoryProtocol.CONTROLLERS_SELF_PATH) {
                assertControllerAuthorization(request)
                require(request.method == "DELETE")
                return MockResponse().setResponseCode(204)
            }
            require(request.path!!.startsWith(AccessoryProtocol.MEDIA_TICKET_PATH))
            val expected = java.util.Base64.getUrlEncoder().withoutPadding().encodeToString(token)
            require(request.getHeader("Authorization") == "Bearer $expected")
            val body = JSONObject(request.body.readUtf8())
            require(body.getString("kind") in setOf("photo", "audio"))
            return ticket()
        }
    }

    private fun assertControllerAuthorization(request: RecordedRequest) {
        val expected = java.util.Base64.getUrlEncoder().withoutPadding().encodeToString(token)
        require(request.getHeader("Authorization") == "Bearer $expected")
    }

    private inner class MediaDispatcher : Dispatcher() {
        override fun dispatch(request: RecordedRequest): MockResponse =
            when (request.path) {
                AccessoryProtocol.PHOTO_PATH -> {
                    require(request.getHeader("Authorization") == "Ticket $ticket")
                    MockResponse()
                        .setHeader("Content-Type", "image/jpeg")
                        .setHeader("X-Kidi-Width", "640")
                        .setHeader("X-Kidi-Height", "480")
                        .setHeader("X-Kidi-SHA256", MessageDigest.getInstance("SHA-256").digest(jpeg).toHex())
                        .setBody(Buffer().write(jpeg))
                }
                AccessoryProtocol.AUDIO_PATH -> {
                    require(request.getHeader("Authorization") == "Ticket $ticket")
                    MockResponse()
                        .setHeader("Content-Type", "application/vnd.kidi.pcm16")
                        .setHeader("X-Kidi-Sample-Rate", "16000")
                        .setHeader("X-Kidi-Channels", "1")
                        .setHeader("X-Kidi-Bits-Per-Sample", "16")
                        .setBody(Buffer().write(pcm))
                        .throttleBody(3, 1, TimeUnit.MILLISECONDS)
                }
                else -> MockResponse().setResponseCode(404)
            }
    }

    private fun ticket() = MockResponse()
        .setHeader("Content-Type", "application/json")
        .setBody(JSONObject().put("protocol", 1).put("ticket", ticket).toString())

    private fun ByteArray.toHex() = joinToString("") { "%02x".format(it) }
}
