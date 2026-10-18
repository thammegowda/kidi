package ai.gowda.kidi

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.util.concurrent.Executors

internal class TtsPlayer(context: Context) : AutoCloseable {
    private val audio = context.getSystemService(AudioManager::class.java)
    private val attributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_MEDIA)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()
    private val focus = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
        .setAudioAttributes(attributes)
        .setOnAudioFocusChangeListener { change ->
            if (change == AudioManager.AUDIOFOCUS_LOSS || change == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT) {
                track?.let(::complete)
            }
        }
        .build()
    private var track: AudioTrack? = null
    private var completion: (() -> Unit)? = null
    private var focusHeld = false
    private val main = Handler(Looper.getMainLooper())
    private val writer = Executors.newSingleThreadExecutor { task -> Thread(task, "kidi-tts-playback") }
    @Volatile private var generation = 0L

    fun play(
        samples: ShortArray,
        sampleRate: Int,
        onError: (Throwable) -> Unit = { error -> Log.e("KidiTts", "playback_failed", error) },
        onComplete: () -> Unit,
    ) {
        require(samples.isNotEmpty() && sampleRate > 0) { "Synthesized audio is empty" }
        stop()
        val minimumBytes = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        require(minimumBytes > 0) { "This device cannot play synthesized PCM" }
        val next = AudioTrack.Builder()
            .setAudioAttributes(attributes)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(maxOf(sampleRate / 4 * Short.SIZE_BYTES, minimumBytes))
            .build()
        try {
            require(next.state == AudioTrack.STATE_INITIALIZED) { "Unable to initialize audio playback" }
            next.setPlaybackPositionUpdateListener(
                object : AudioTrack.OnPlaybackPositionUpdateListener {
                    override fun onMarkerReached(audioTrack: AudioTrack) {
                        Log.i("KidiTts", "playback_complete frames=${samples.size}")
                        complete(audioTrack)
                    }

                    override fun onPeriodicNotification(audioTrack: AudioTrack) = Unit
                },
                Handler(Looper.getMainLooper()),
            )
            next.notificationMarkerPosition = samples.size
            next.setVolume(AudioTrack.getMaxVolume())
            focusHeld = audio.requestAudioFocus(focus) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
            if (!focusHeld) {
                Log.w("KidiTts", "audio_focus_denied; continuing on the media mixer")
            }
            track = next
            completion = onComplete
            val request = generation
            next.play()
            Log.i("KidiTts", "playback_started frames=${samples.size} sample_rate=$sampleRate")
            writer.execute {
                try {
                    var offset = 0
                    while (offset < samples.size && generation == request) {
                        val count =
                            next.write(samples, offset, minOf(4096, samples.size - offset), AudioTrack.WRITE_BLOCKING)
                        if (generation != request) return@execute
                        check(count > 0) { "Unable to stream synthesized audio: AudioTrack error $count" }
                        offset += count
                    }
                } catch (error: Exception) {
                    main.post {
                        if (generation == request && track === next) {
                            Log.e("KidiTts", "playback_failed", error)
                            completion = null
                            release(next)
                            onError(error)
                        }
                    }
                }
            }
        } catch (error: Exception) {
            release(next)
            throw error
        }
    }

    fun stop() {
        ++generation
        completion = null
        val current = track
        if (current != null) release(current) else abandonFocus()
    }

    private fun complete(value: AudioTrack) {
        if (track !== value) return
        val callback = completion
        completion = null
        release(value)
        callback?.invoke()
    }

    private fun release(value: AudioTrack) {
        if (track === value) {
            ++generation
            track = null
        }
        try {
            value.stop()
            value.flush()
        } catch (error: IllegalStateException) {
            Log.w("KidiTts", "playback_release_state_changed", error)
        }
        value.release()
        abandonFocus()
    }

    private fun abandonFocus() {
        if (focusHeld) audio.abandonAudioFocusRequest(focus)
        focusHeld = false
    }

    override fun close() {
        stop()
        writer.shutdownNow()
    }
}
