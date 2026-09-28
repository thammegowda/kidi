package ai.gowda.kidi

import android.Manifest
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

internal class SpeechRecorder {
    private val stopRequested = AtomicBoolean()

    @Volatile
    private var active: AudioRecord? = null

    fun prepare() {
        stopRequested.set(false)
    }

    fun stop() {
        stopRequested.set(true)
        runCatching { active?.stop() }
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    suspend fun capture(onSamples: (Int) -> Boolean, onSnapshot: (FloatArray) -> Unit): FloatArray =
        withContext(Dispatchers.IO) {
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
            val buffer = ShortArray(maxOf(minimumBuffer / Short.SIZE_BYTES, 1024))
            var size = 0
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
                        throw IllegalStateException("Microphone read failed: $count")
                    }
                    if (count == 0) continue
                    buffer.copyInto(captured, destinationOffset = size, endIndex = count)
                    size += count
                    if (onSamples(size)) onSnapshot(captured.toWaveform(size))
                }
            } finally {
                runCatching { recorder.stop() }
                recorder.release()
                active = null
            }
            require(size > 0) { "No speech was recorded" }
            captured.toWaveform(size)
        }

    private fun ShortArray.toWaveform(size: Int) = FloatArray(size) { this[it] / 32768f }

    private companion object {
        const val SAMPLE_RATE = 16000
        const val MAXIMUM_SAMPLES = SAMPLE_RATE * 30
    }
}