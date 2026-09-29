package ai.gowda.kidi

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.net.URL
import java.security.MessageDigest
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.SocketPolicy
import okio.Buffer
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelDownloaderTest {
    @get:Rule val temporary = TemporaryFolder()

    @Test
    fun interruptedTransferResumesAndVerifiesBeforePublishing() = runBlocking {
        val payload = "verified model bytes".toByteArray()
        val phases = mutableListOf<ModelDownloadPhase>()
        val server = MockWebServer()
        val offset = payload.size / 2
        server.enqueue(MockResponse().setBody(Buffer().write(payload))
            .setSocketPolicy(SocketPolicy.DISCONNECT_DURING_RESPONSE_BODY))
        server.enqueue(MockResponse().setResponseCode(206)
            .setHeader("Content-Range", "bytes $offset-${payload.lastIndex}/${payload.size}")
            .setBody(Buffer().write(payload, offset, payload.size - offset)))
        server.start()
        try {
            val target = File(temporary.root, "model.safetensors")
            ModelDownloader(readTimeoutMs = 1000, retryDelayMs = 0).download(
                server.url("/model").toUrl(), target, payload.size.toLong(), sha256(payload),
            ) { update -> phases.add(update.phase); assertFalse(target.exists()) }
            assertArrayEquals(payload, target.readBytes())
            assertEquals(2, server.requestCount)
            assertNull(server.takeRequest().getHeader("Range"))
            assertEquals("bytes=$offset-", server.takeRequest().getHeader("Range"))
            assertTrue(ModelDownloadPhase.RETRYING in phases)
            assertTrue(ModelDownloadPhase.VERIFYING in phases)
            assertFalse(File(temporary.root, "model.safetensors.part").exists())
        } finally {
            server.shutdown()
        }
    }

    @Test
    fun resumeOffsetsRemainLongBeyondTwoGigabytes() = runBlocking {
        val offset = Int.MAX_VALUE.toLong() + 64
        val tail = byteArrayOf(1, 2, 3)
        val target = File(temporary.root, "large.safetensors")
        RandomAccessFile(File(temporary.root, "large.safetensors.part"), "rw").use { it.setLength(offset) }
        val server = MockWebServer()
        server.enqueue(MockResponse().setResponseCode(206)
            .setHeader("Content-Range", "bytes $offset-${offset + 2}/${offset + 3}")
            .setBody(Buffer().write(tail)))
        server.start()
        try {
            ModelDownloader(retryDelayMs = 0).download(server.url("/model").toUrl(),
                target, offset + 3, null) {}
            assertEquals("bytes=$offset-", server.takeRequest().getHeader("Range"))
            assertEquals(offset + 3, target.length())
            RandomAccessFile(target, "r").use { file ->
                file.seek(offset)
                val actual = ByteArray(3)
                file.readFully(actual)
                assertArrayEquals(tail, actual)
            }
        } finally {
            server.shutdown()
        }
    }

    @Test
    fun exhaustedTimeoutRetriesReportFailureAndKeepPartial() = runBlocking {
        val target = File(temporary.root, "model.safetensors")
        val partial = File(temporary.root, "model.safetensors.part")
        partial.writeBytes(byteArrayOf(1, 2))
        MockWebServer().use { server ->
            repeat(2) { server.enqueue(MockResponse().setSocketPolicy(SocketPolicy.NO_RESPONSE)) }
            server.start()
            try {
                ModelDownloader(readTimeoutMs = 100, retryDelayMs = 0, maximumAttempts = 2)
                    .download(server.url("/model").toUrl(), target, 4, null) {}
                fail("Timed-out download must fail after bounded attempts")
            } catch (error: IOException) {
                assertTrue(error.message.orEmpty().contains("resume saved progress"))
                assertEquals(2, server.requestCount)
                assertFalse(target.exists())
                assertArrayEquals(byteArrayOf(1, 2), partial.readBytes())
            }
        }
    }

    @Test
    fun permanentHttpErrorIsReportedWithoutRetrying() = runBlocking {
        val target = File(temporary.root, "model.safetensors")
        MockWebServer().use { server ->
            server.enqueue(MockResponse().setResponseCode(404))
            server.start()
            try {
                ModelDownloader(retryDelayMs = 0).download(server.url("/model").toUrl(), target, 4, null) {}
                fail("HTTP failure must be reported")
            } catch (error: IOException) {
                assertTrue(error.message.orEmpty().contains("HTTP 404"))
                assertEquals(1, server.requestCount)
                assertFalse(target.exists())
            }
        }
    }

    @Test
    fun ignoredRangeRestartsInsteadOfAppendingDuplicateBytes() = runBlocking {
        val payload = byteArrayOf(1, 2, 3, 4)
        val target = File(temporary.root, "model.safetensors")
        File(temporary.root, "model.safetensors.part").writeBytes(payload.copyOf(2))
        MockWebServer().use { server ->
            server.enqueue(MockResponse().setBody(Buffer().write(payload)))
            server.start()
            ModelDownloader(retryDelayMs = 0).download(server.url("/model").toUrl(), target, 4, sha256(payload)) {}
            assertEquals("bytes=2-", server.takeRequest().getHeader("Range"))
            assertArrayEquals(payload, target.readBytes())
        }
    }

    @Test
    fun invalidResumeResponseCannotModifyPartial() = runBlocking {
        val target = File(temporary.root, "model.safetensors")
        val partial = File(temporary.root, "model.safetensors.part")
        partial.writeBytes(byteArrayOf(1, 2))
        MockWebServer().use { server ->
            server.enqueue(MockResponse().setResponseCode(206).setHeader("Content-Range", "bytes 1-3/4")
                .setBody(Buffer().write(byteArrayOf(2, 3, 4))))
            server.start()
            try {
                ModelDownloader(retryDelayMs = 0).download(server.url("/model").toUrl(), target, 4, null) {}
                fail("Invalid ranges must not be accepted")
            } catch (error: IllegalStateException) {
                assertTrue(error.message.orEmpty().contains("Invalid resume"))
                assertArrayEquals(byteArrayOf(1, 2), partial.readBytes())
                assertFalse(target.exists())
            }
        }
    }

    @Test
    fun checksumFailureCannotPublishOrRetainCorruptPartial() = runBlocking {
        val target = File(temporary.root, "model.safetensors")
        MockWebServer().use { server ->
            server.enqueue(MockResponse().setBody(Buffer().write(byteArrayOf(1, 2, 3))))
            server.start()
            try {
                ModelDownloader().download(server.url("/model").toUrl(), target, 3, sha256(byteArrayOf(4, 5, 6))) {}
                fail("Invalid checksums must not be published")
            } catch (error: IllegalStateException) {
                assertTrue(error.message.orEmpty().contains("Checksum mismatch"))
                assertFalse(target.exists())
                assertFalse(File(temporary.root, "model.safetensors.part").exists())
            }
        }
    }

    @Test
    fun completePartialIsVerifiedWithoutAnotherNetworkRequest() = runBlocking {
        val payload = byteArrayOf(4, 5, 6)
        val target = File(temporary.root, "model.safetensors")
        File(temporary.root, "model.safetensors.part").writeBytes(payload)
        ModelDownloader().download(URL("http://127.0.0.1:1/unreachable"), target,
            payload.size.toLong(), sha256(payload)) {}
        assertArrayEquals(payload, target.readBytes())
    }

    @Test
    fun cancellationDuringVerificationRetainsPartial() = runBlocking {
        val payload = byteArrayOf(4, 5, 6)
        val target = File(temporary.root, "model.safetensors")
        val partial = File(temporary.root, "model.safetensors.part")
        partial.writeBytes(payload)
        try {
            ModelDownloader().download(URL("http://127.0.0.1:1/unreachable"), target,
                payload.size.toLong(), sha256(payload)) { throw CancellationException("cancel test") }
            fail("Cancellation must propagate")
        } catch (_: CancellationException) {
            assertFalse(target.exists())
            assertArrayEquals(payload, partial.readBytes())
        }
    }

    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it) }
}