package ai.gowda.kidi

import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import java.io.EOFException
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

internal enum class ModelDownloadPhase(val label: String) {
    CONNECTING("Connecting"), DOWNLOADING("Downloading"), VERIFYING("Verifying"), RETRYING("Retrying"),
}

internal data class FileDownloadProgress(
    val phase: ModelDownloadPhase,
    val bytes: Long,
    val attempt: Int,
)

internal class ModelDownloader(
    private val readTimeoutMs: Int = 30_000,
    private val retryDelayMs: Long = 1_000,
    private val maximumAttempts: Int = 3,
) {
    private class HttpFailure(val code: Int) : IOException("HTTP $code") {
        val retryable: Boolean get() = code == 403 || code == 408 || code == 429 || code in 500..599
    }

    suspend fun download(
        source: URL,
        target: File,
        size: Long,
        sha256: String?,
        progress: (FileDownloadProgress) -> Unit,
    ) {
        require(size > 0 && maximumAttempts > 0 && readTimeoutMs > 0 && retryDelayMs >= 0)
        val partial = File(target.parentFile, "${target.name}.part")
        var lastPhase: ModelDownloadPhase? = null
        var lastUpdate = 0L
        var attempt = 1
        fun publish(phase: ModelDownloadPhase, bytes: Long) {
            val now = System.nanoTime()
            if (phase != lastPhase || bytes == size || now - lastUpdate >= 250_000_000) {
                progress(FileDownloadProgress(phase, bytes, attempt))
                lastPhase = phase
                lastUpdate = now
            }
        }
        suspend fun verify(file: File): Boolean {
            coroutineContext.ensureActive()
            if (sha256 == null) return true
            publish(ModelDownloadPhase.VERIFYING, 0)
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { input ->
                val buffer = ByteArray(BUFFER_BYTES)
                var checked = 0L
                while (true) {
                    coroutineContext.ensureActive()
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                    checked += count
                    publish(ModelDownloadPhase.VERIFYING, checked)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) } == sha256
        }
        if (target.isFile && target.length() == size && verify(target)) return
        if (target.exists()) check(target.delete()) { "Unable to replace ${target.name}" }
        if (partial.length() > size) check(partial.delete()) { "Unable to reset ${target.name}" }
        if (partial.isFile && partial.length() == size) {
            if (verify(partial)) {
                publish(partial, target)
                return
            }
            check(partial.delete()) { "Unable to reset ${target.name}" }
        }
        while (true) {
            coroutineContext.ensureActive()
            var offset = partial.length()
            publish(ModelDownloadPhase.CONNECTING, offset)
            try {
                val connection = connect(source, offset)
                try {
                    val code = connection.responseCode
                    if (code != HttpURLConnection.HTTP_OK && code != HttpURLConnection.HTTP_PARTIAL)
                        throw HttpFailure(code)
                    if (code == HttpURLConnection.HTTP_OK) {
                        offset = 0
                    } else {
                        val range = CONTENT_RANGE.matchEntire(connection.getHeaderField("Content-Range").orEmpty())
                        val start = range?.groupValues?.get(1)?.toLongOrNull()
                        val end = range?.groupValues?.get(2)?.toLongOrNull()
                        val total = range?.groupValues?.get(3)?.toLongOrNull()
                        check(start == offset && end != null && end >= offset && end < size && total == size) {
                            "Invalid resume response for ${target.name}. Retry the download."
                        }
                    }
                    publish(ModelDownloadPhase.DOWNLOADING, offset)
                    connection.inputStream.use { input ->
                        FileOutputStream(partial, offset > 0).use { output ->
                            val buffer = ByteArray(BUFFER_BYTES)
                            var downloaded = offset
                            while (true) {
                                coroutineContext.ensureActive()
                                val count = input.read(buffer)
                                if (count < 0) break
                                check(count.toLong() <= size - downloaded) { "Download exceeded expected size for ${target.name}" }
                                output.write(buffer, 0, count)
                                downloaded += count
                                publish(ModelDownloadPhase.DOWNLOADING, downloaded)
                            }
                            output.fd.sync()
                        }
                    }
                } finally {
                    connection.disconnect()
                }
                if (partial.length() != size) throw EOFException("Connection ended before the file was complete")
                break
            } catch (error: IOException) {
                coroutineContext.ensureActive()
                if (attempt >= maximumAttempts || error is HttpFailure && !error.retryable) {
                    val detail = if (error is HttpFailure) "HTTP ${error.code}" else error.message ?: error.javaClass.simpleName
                    throw IOException("Download ${target.name} stopped: $detail. Download again to resume saved progress.", error)
                }
                attempt++
                publish(ModelDownloadPhase.RETRYING, partial.length())
                delay(retryDelayMs * (attempt - 1))
            }
        }
        if (!verify(partial)) {
            check(partial.delete()) { "Unable to remove damaged ${target.name}" }
            error("Checksum mismatch for ${target.name}. Download again to replace the damaged file.")
        }
        coroutineContext.ensureActive()
        publish(partial, target)
    }

    private suspend fun connect(source: URL, offset: Long): HttpURLConnection {
        var url = source
        repeat(8) {
            coroutineContext.ensureActive()
            val connection = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 15_000
                readTimeout = readTimeoutMs
                instanceFollowRedirects = false
                useCaches = false
                setRequestProperty("Accept-Encoding", "identity")
                setRequestProperty("Cache-Control", "no-cache")
                if (offset > 0) setRequestProperty("Range", "bytes=$offset-")
            }
            try {
                if (connection.responseCode !in setOf(301, 302, 303, 307, 308)) return connection
                val location = connection.getHeaderField("Location")
                    ?: error("Model download redirect has no location")
                val redirected = URL(url, location)
                check(redirected.protocol in setOf("http", "https") &&
                    (url.protocol != "https" || redirected.protocol == "https")) { "Unsafe model download redirect" }
                url = redirected
            } catch (error: Throwable) {
                connection.disconnect()
                throw error
            }
            connection.disconnect()
        }
        error("Too many model download redirects")
    }

    private fun publish(source: File, target: File) {
        try {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private companion object {
        const val BUFFER_BYTES = 1024 * 1024
        val CONTENT_RANGE = Regex("bytes (\\d+)-(\\d+)/(\\d+)")
    }
}