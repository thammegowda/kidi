package ai.gowda.kidi

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okio.Buffer
import org.json.JSONObject
import java.security.MessageDigest
import java.security.SecureRandom
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import java.util.Base64
import java.util.concurrent.TimeUnit
import javax.net.SocketFactory
import javax.net.ssl.SSLContext
import javax.net.ssl.X509TrustManager
import kotlin.coroutines.coroutineContext

internal data class AccessoryProfile(
    val deviceId: String,
    val name: String,
    val host: String,
    val controlPort: Int = AccessoryProtocol.CONTROL_PORT,
    val mediaPort: Int = AccessoryProtocol.MEDIA_PORT,
    val controlCertificateSha256: ByteArray,
    val mediaCertificateSha256: ByteArray,
    val controllerToken: ByteArray,
    val softApSsid: String,
    val softApPassword: String,
) {
    init {
        require(deviceId.matches(Regex("[0-9a-f]{32}"))) { "Invalid accessory ID" }
        require(name.isNotBlank() && name.length <= 64 && name.none(Char::isISOControl)) {
            "Invalid accessory name"
        }
        require(host.split('.').let { parts ->
            parts.size == 4 && parts.all { part ->
                part.isNotEmpty() && part.all(Char::isDigit) &&
                    part.toIntOrNull()?.let { it in 0..255 } == true
            }
        }) { "Accessory host must be an IPv4 literal" }
        require(controlPort in 61440..65535 && mediaPort in 49152..61439) {
            "Accessory ports do not match the Network Split ranges"
        }
        require(controlCertificateSha256.size == 32 && mediaCertificateSha256.size == 32) {
            "Accessory certificate pins must be SHA-256"
        }
        require(controllerToken.size == 32) { "Accessory controller token must be 256 bits" }
        require(softApSsid.toByteArray().size in 1..32 && '\u0000' !in softApSsid) {
            "Accessory SoftAP SSID is invalid"
        }
        require(softApPassword.toByteArray().size in 8..63 && '\u0000' !in softApPassword) {
            "Accessory SoftAP password is invalid"
        }
    }
}

internal data class AccessoryAudioFormat(
    val sampleRate: Int,
    val channels: Int,
    val bitsPerSample: Int,
)

private data class MediaTicket(val value: String)

internal class AccessoryClient(
    private val profile: AccessoryProfile,
    private val networkSocketFactory: SocketFactory,
) {
    suspend fun unpair() = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url(
                "https://${profile.host}:${profile.controlPort}" +
                    AccessoryProtocol.CONTROLLERS_SELF_PATH,
            )
            .header(
                "Authorization",
                "Bearer ${
                    Base64.getUrlEncoder().withoutPadding()
                        .encodeToString(profile.controllerToken)
                }",
            )
            .delete()
            .build()
        controlClient().newCall(request).execute().use { response ->
            require(response.code == 204) {
                "Accessory unpair failed with HTTP ${response.code}"
            }
        }
    }

    suspend fun capturePhoto(profileName: String = "default"): ByteArray = withContext(Dispatchers.IO) {
        val ticket = requestTicket("photo", JSONObject().put("profile", profileName))
        val request = Request.Builder()
            .url("https://${profile.host}:${profile.mediaPort}${AccessoryProtocol.PHOTO_PATH}")
            .header("Authorization", "Ticket ${ticket.value}")
            .post(
                JSONObject()
                    .put("profile", profileName)
                    .put("maximum_bytes", MAXIMUM_PHOTO_BYTES)
                    .toString()
                    .toRequestBody(JSON),
            )
            .build()
        mediaClient().newCall(request).execute().use { response ->
            require(response.isSuccessful) { "Accessory photo failed with HTTP ${response.code}" }
            require(response.header("Content-Type")?.substringBefore(';') == "image/jpeg") {
                "Accessory returned an unsupported photo format"
            }
            val declared = response.body?.contentLength() ?: -1
            require(declared in 1..MAXIMUM_PHOTO_BYTES.toLong()) { "Accessory photo size is invalid" }
            val jpeg = requireNotNull(response.body).bytes()
            require(jpeg.size.toLong() == declared && jpeg.size <= MAXIMUM_PHOTO_BYTES) {
                "Accessory photo was incomplete or exceeded its bound"
            }
            require(
                jpeg.size >= 4 &&
                    jpeg[0] == 0xff.toByte() &&
                    jpeg[1] == 0xd8.toByte() &&
                    jpeg[jpeg.lastIndex - 1] == 0xff.toByte() &&
                    jpeg[jpeg.lastIndex] == 0xd9.toByte(),
            ) { "Accessory photo is not a complete JPEG" }
            require(response.requiredIntHeader(AccessoryProtocol.PHOTO_WIDTH_HEADER) in 1..MAXIMUM_PHOTO_DIMENSION) {
                "Accessory photo width is invalid"
            }
            require(response.requiredIntHeader(AccessoryProtocol.PHOTO_HEIGHT_HEADER) in 1..MAXIMUM_PHOTO_DIMENSION) {
                "Accessory photo height is invalid"
            }
            val expectedDigest = response.header(AccessoryProtocol.PHOTO_DIGEST_HEADER)
                ?.takeIf { it.matches(Regex("[0-9a-f]{64}")) }
                ?: error("Accessory photo is missing its SHA-256 digest")
            require(MessageDigest.getInstance("SHA-256").digest(jpeg).toHex() == expectedDigest) {
                "Accessory photo digest mismatch"
            }
            jpeg
        }
    }

    suspend fun streamAudio(
        maximumSeconds: Int = AccessoryProtocol.MAXIMUM_AUDIO_SECONDS,
        shouldContinue: () -> Boolean = { true },
        onSamples: suspend (ShortArray) -> Unit,
    ): AccessoryAudioFormat = withContext(Dispatchers.IO) {
        require(maximumSeconds in 1..AccessoryProtocol.MAXIMUM_AUDIO_SECONDS) {
            "Accessory audio duration is out of range"
        }
        val ticket = requestTicket("audio", JSONObject().put("maximum_seconds", maximumSeconds))
        val request = Request.Builder()
            .url("https://${profile.host}:${profile.mediaPort}${AccessoryProtocol.AUDIO_PATH}")
            .header("Authorization", "Ticket ${ticket.value}")
            .post(
                JSONObject()
                    .put("sample_rate", AccessoryProtocol.AUDIO_SAMPLE_RATE)
                    .put("channels", AccessoryProtocol.AUDIO_CHANNELS)
                    .put("bits_per_sample", AccessoryProtocol.AUDIO_BITS_PER_SAMPLE)
                    .put("maximum_seconds", maximumSeconds)
                    .toString()
                    .toRequestBody(JSON),
            )
            .build()
        mediaClient().newCall(request).execute().use { response ->
            require(response.isSuccessful) { "Accessory audio failed with HTTP ${response.code}" }
            require(response.header("Content-Type")?.substringBefore(';') == "application/vnd.kidi.pcm16") {
                "Accessory returned an unsupported audio format"
            }
            val format = AccessoryAudioFormat(
                sampleRate = response.requiredIntHeader(AccessoryProtocol.AUDIO_SAMPLE_RATE_HEADER),
                channels = response.requiredIntHeader(AccessoryProtocol.AUDIO_CHANNELS_HEADER),
                bitsPerSample = response.requiredIntHeader(AccessoryProtocol.AUDIO_BITS_PER_SAMPLE_HEADER),
            )
            require(
                format == AccessoryAudioFormat(
                    AccessoryProtocol.AUDIO_SAMPLE_RATE,
                    AccessoryProtocol.AUDIO_CHANNELS,
                    AccessoryProtocol.AUDIO_BITS_PER_SAMPLE,
                ),
            ) { "Accessory audio format does not match the requested speech profile" }
            val source = requireNotNull(response.body).source()
            val buffer = Buffer()
            var received = 0L
            var pendingLowByte: Int? = null
            val maximumBytes = maximumSeconds.toLong() * format.sampleRate * Short.SIZE_BYTES
            while (true) {
                coroutineContext.ensureActive()
                if (!shouldContinue()) break
                if (received == maximumBytes) {
                    require(source.exhausted()) { "Accessory audio exceeded its duration bound" }
                    break
                }
                val requested = minOf(AUDIO_CHUNK_BYTES.toLong(), maximumBytes - received)
                val count = source.read(buffer, requested)
                if (count == -1L) break
                val data = buffer.readByteArray()
                received += data.size
                val samples = ShortArray((data.size + if (pendingLowByte == null) 0 else 1) / 2)
                var sourceOffset = 0
                var sampleOffset = 0
                pendingLowByte?.let { low ->
                    if (data.isNotEmpty()) {
                        samples[sampleOffset++] = ((data[sourceOffset++].toInt() shl 8) or low).toShort()
                        pendingLowByte = null
                    }
                }
                while (sourceOffset + 1 < data.size) {
                    val low = data[sourceOffset++].toInt() and 0xff
                    val high = data[sourceOffset++].toInt()
                    samples[sampleOffset++] = ((high shl 8) or low).toShort()
                }
                if (sourceOffset < data.size) pendingLowByte = data[sourceOffset].toInt() and 0xff
                if (sampleOffset > 0) onSamples(samples.copyOf(sampleOffset))
            }
            require(pendingLowByte == null) { "Accessory audio ended within a PCM sample" }
            require(received > 0) { "Accessory returned no audio samples" }
            format
        }
    }

    private fun requestTicket(kind: String, parameters: JSONObject): MediaTicket {
        val request = Request.Builder()
            .url("https://${profile.host}:${profile.controlPort}${AccessoryProtocol.MEDIA_TICKET_PATH}")
            .header("Authorization", "Bearer ${Base64.getUrlEncoder().withoutPadding().encodeToString(profile.controllerToken)}")
            .post(
                JSONObject()
                    .put("kind", kind)
                    .put("parameters", parameters)
                    .toString()
                    .toRequestBody(JSON),
            )
            .build()
        controlClient().newCall(request).execute().use { response ->
            require(response.isSuccessful) { "Accessory media ticket failed with HTTP ${response.code}" }
            val body = response.body?.string() ?: error("Accessory ticket response is empty")
            require(body.length <= MAXIMUM_TICKET_RESPONSE_BYTES) { "Accessory ticket response exceeds its bound" }
            val objectValue = JSONObject(body)
            require(objectValue.getInt("protocol") == AccessoryProtocol.VERSION) {
                "Accessory ticket protocol is unsupported"
            }
            val ticket = objectValue.getString("ticket")
            require(ticket.length in 32..512 && ticket.none(Char::isWhitespace)) {
                "Accessory returned an invalid media ticket"
            }
            return MediaTicket(ticket)
        }
    }

    private fun controlClient() = pinnedClient(profile.controlCertificateSha256)
    private fun mediaClient() = pinnedClient(profile.mediaCertificateSha256)

    private fun pinnedClient(expectedSha256: ByteArray): OkHttpClient {
        require(expectedSha256.size == 32) { "Accessory certificate pin must be SHA-256" }
        val trustManager = CertificatePinTrustManager(expectedSha256)
        val context = SSLContext.getInstance("TLS")
        context.init(null, arrayOf(trustManager), SecureRandom())
        return OkHttpClient.Builder()
            .socketFactory(networkSocketFactory)
            .sslSocketFactory(context.socketFactory, trustManager)
            .hostnameVerifier { host, session ->
                host == profile.host && runCatching {
                    val certificate = session.peerCertificates.first() as X509Certificate
                    MessageDigest.getInstance("SHA-256").digest(certificate.encoded)
                        .contentEquals(expectedSha256)
                }.getOrDefault(false)
            }
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(35, TimeUnit.SECONDS)
            .writeTimeout(35, TimeUnit.SECONDS)
            .build()
    }

    private class CertificatePinTrustManager(private val expectedSha256: ByteArray) : X509TrustManager {
        override fun checkClientTrusted(chain: Array<X509Certificate>?, authType: String?) =
            throw CertificateException("Accessory client certificates are not accepted")

        override fun checkServerTrusted(chain: Array<X509Certificate>?, authType: String?) {
            val certificate = chain?.firstOrNull() ?: throw CertificateException("Accessory certificate chain is invalid")
            val digest = MessageDigest.getInstance("SHA-256").digest(certificate.encoded)
            if (!MessageDigest.isEqual(digest, expectedSha256)) {
                throw CertificateException("Accessory certificate pin mismatch")
            }
        }

        override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
    }

    private fun okhttp3.Response.requiredIntHeader(name: String): Int =
        header(name)?.toIntOrNull() ?: error("Accessory response is missing $name")

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }

    private companion object {
        val JSON = "application/json; charset=utf-8".toMediaType()
        const val MAXIMUM_PHOTO_BYTES = 4 * 1024 * 1024
        const val MAXIMUM_PHOTO_DIMENSION = 8192
        const val MAXIMUM_TICKET_RESPONSE_BYTES = 4096
        const val AUDIO_CHUNK_BYTES = 4096
    }
}
