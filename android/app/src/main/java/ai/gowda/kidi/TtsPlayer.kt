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

internal class TtsPlayer(context: Context) {
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

    fun play(samples: ShortArray, sampleRate: Int, onComplete: () -> Unit) {
        require(samples.isNotEmpty() && sampleRate > 0) { "Synthesized audio is empty" }
        stop()
        val audioBytes = Math.multiplyExact(samples.size, Short.SIZE_BYTES)
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
            .setBufferSizeInBytes(maxOf(audioBytes, minimumBytes))
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
            require(next.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING) == samples.size) {
                "Unable to buffer synthesized audio"
            }
            next.notificationMarkerPosition = samples.size
            next.setVolume(AudioTrack.getMaxVolume())
            focusHeld = audio.requestAudioFocus(focus) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
            if (!focusHeld) {
                Log.w("KidiTts", "audio_focus_denied; continuing on the media mixer")
            }
            track = next
            completion = onComplete
            next.play()
            Log.i("KidiTts", "playback_started frames=${samples.size} sample_rate=$sampleRate")
        } catch (error: Exception) {
            release(next)
            throw error
        }
    }

    fun stop() {
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
        if (track === value) track = null
        runCatching { value.stop() }
        value.flush()
        value.release()
        abandonFocus()
    }

    private fun abandonFocus() {
        if (focusHeld) audio.abandonAudioFocusRequest(focus)
        focusHeld = false
    }
}
