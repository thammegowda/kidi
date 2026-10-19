package ai.gowda.kidi

import android.Manifest
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

internal class AndroidAudioRecorder {
    data class Capture(val samples: FloatArray, val heardSpeech: Boolean)

    private val stopRequested = AtomicBoolean()

    @Volatile
    private var active: AudioRecord? = null

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    suspend fun capture(
        voiceMode: Boolean,
        threshold: Float,
        minimumSpeechMs: Int,
        silenceMs: Int,
        onSnapshot: (FloatArray) -> Unit,
    ): Capture = withContext(Dispatchers.IO) {
        require(threshold.isFinite() && threshold > 0f && threshold <= 1f)
        require(minimumSpeechMs > 0 && silenceMs > 0)
        stopRequested.set(false)
        val minimumBuffer = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        require(minimumBuffer > 0) { "This device cannot capture 16 kHz mono audio" }
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minimumBuffer * 2, SAMPLE_RATE),
        )
        require(recorder.state == AudioRecord.STATE_INITIALIZED) { "Unable to initialize the microphone" }
        active = recorder
        val captured = ShortArray(MAXIMUM_SAMPLES)
        val buffer = ShortArray(maxOf(minimumBuffer / Short.SIZE_BYTES, FRAME_SAMPLES))
        var size = 0
        var heardSpeech = false
        var speechSamples = 0
        var silentSamples = 0
        var nextSnapshot = SAMPLE_RATE
        try {
            recorder.startRecording()
            require(recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                "Unable to start the microphone"
            }
            while (!stopRequested.get() && size < captured.size) {
                val requested = minOf(buffer.size, captured.size - size)
                val count = recorder.read(buffer, 0, requested, AudioRecord.READ_BLOCKING)
                if (count < 0) {
                    if (stopRequested.get()) break
                    error("Microphone read failed: $count")
                }
                if (count == 0) continue
                buffer.copyInto(captured, destinationOffset = size, endIndex = count)
                size += count
                var energy = 0.0
                for (index in 0 until count) {
                    val value = buffer[index] / 32768.0
                    energy += value * value
                }
                val activeSpeech = sqrt(energy / count) >= threshold
                if (!heardSpeech) {
                    speechSamples = if (activeSpeech) speechSamples + count else 0
                    heardSpeech = speechSamples * 1000 >= minimumSpeechMs * SAMPLE_RATE
                } else if (activeSpeech) {
                    silentSamples = 0
                } else {
                    silentSamples += count
                }
                if (size >= nextSnapshot) {
                    onSnapshot(captured.toFloatSamples(size))
                    nextSnapshot += SAMPLE_RATE
                }
                if (voiceMode && heardSpeech && silentSamples * 1000 >= silenceMs * SAMPLE_RATE) break
            }
        } finally {
            runCatching { recorder.stop() }
            recorder.release()
            active = null
        }
        require(size > 0) { "No microphone audio was captured" }
        Capture(captured.toFloatSamples(size), heardSpeech)
    }

    fun stop() {
        stopRequested.set(true)
        runCatching { active?.stop() }
    }

    private fun ShortArray.toFloatSamples(size: Int) = FloatArray(size) { this[it] / 32768f }

    companion object {
        const val SAMPLE_RATE = 16_000
        const val MAXIMUM_SECONDS = 30
        private const val MAXIMUM_SAMPLES = SAMPLE_RATE * MAXIMUM_SECONDS
        private const val FRAME_SAMPLES = 320
    }
}
