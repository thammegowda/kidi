package ai.gowda.kidi

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

internal class AndroidAudioPlayer {
    private val stopRequested = AtomicBoolean()

    @Volatile
    private var active: AudioTrack? = null

    suspend fun play(samples: ShortArray, sampleRate: Int, onStarted: () -> Unit = {}) =
        withContext(Dispatchers.IO) {
            require(samples.isNotEmpty() && sampleRate > 0) { "Native speech audio is empty or invalid" }
            stopRequested.set(false)
            val minimumBuffer = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            require(minimumBuffer > 0) { "This device cannot play native speech audio" }
            val player = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build(),
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .build(),
                )
                .setBufferSizeInBytes(maxOf(minimumBuffer, sampleRate * Short.SIZE_BYTES / 2))
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
            require(player.state == AudioTrack.STATE_INITIALIZED) { "Unable to initialize native speech playback" }
            active = player
            try {
                player.play()
                val playbackStartedAt = SystemClock.elapsedRealtime()
                onStarted()
                var offset = 0
                while (offset < samples.size && !stopRequested.get()) {
                    val written = player.write(
                        samples,
                        offset,
                        minOf(4096, samples.size - offset),
                        AudioTrack.WRITE_BLOCKING,
                    )
                    require(written > 0) { "Native speech playback failed: $written" }
                    offset += written
                }
                val expectedPlaybackMs = samples.size * 1000L / sampleRate
                val deadline = playbackStartedAt + expectedPlaybackMs + PLAYBACK_DRAIN_GRACE_MS
                while (!stopRequested.get() && SystemClock.elapsedRealtime() < deadline) {
                    Thread.sleep(10)
                }
            } finally {
                runCatching { player.stop() }
                player.release()
                active = null
            }
        }

    fun stop() {
        stopRequested.set(true)
        active?.let { player ->
            runCatching { player.pause() }
            runCatching { player.flush() }
            runCatching { player.stop() }
        }
    }

    private companion object {
        const val PLAYBACK_DRAIN_GRACE_MS = 100L
    }
}
