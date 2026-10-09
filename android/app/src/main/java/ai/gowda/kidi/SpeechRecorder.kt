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
    /** Recorded samples and where speech last ended, so a draft that already covers the speech can be reused. */
    class Recording(val samples: FloatArray, val speechEnd: Int)

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

    /**
     * Captures 16 kHz mono speech. [onSamples] receives the sample count and the end of the last voiced frame, and
     * returns true when a snapshot should be passed to [onSnapshot].
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    suspend fun capture(onSamples: (Int, Int) -> Boolean, onSnapshot: (FloatArray) -> Unit): Recording =
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
            val collector = SampleCollector(onSamples, onSnapshot)
            val buffer = ShortArray(maxOf(minimumBuffer / Short.SIZE_BYTES, 1024))
            try {
                recorder.startRecording()
                require(recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    "Unable to start the microphone"
                }
                while (!stopRequested.get() && !collector.full) {
                    val requested = minOf(buffer.size, collector.remaining)
                    val count = recorder.read(buffer, 0, requested, AudioRecord.READ_BLOCKING)
                    if (count < 0) {
                        if (stopRequested.get()) break
                        throw IllegalStateException("Microphone read failed: $count")
                    }
                    if (count == 0) continue
                    collector.append(buffer, count)
                }
            } finally {
                runCatching { recorder.stop() }
                recorder.release()
                active = null
            }
            collector.recording()
        }

    suspend fun captureAccessory(
        client: AccessoryClient,
        onSamples: (Int, Int) -> Boolean,
        onSnapshot: (FloatArray) -> Unit,
    ): Recording = withContext(Dispatchers.IO) {
        val collector = SampleCollector(onSamples, onSnapshot)
        client.streamAudio(
            maximumSeconds = MAXIMUM_SECONDS,
            shouldContinue = { !stopRequested.get() && !collector.full },
        ) { collector.append(it, it.size) }
        collector.recording()
    }

    private class SampleCollector(
        private val onSamples: (Int, Int) -> Boolean,
        private val onSnapshot: (FloatArray) -> Unit,
    ) {
        private val captured = ShortArray(MAXIMUM_SAMPLES)
        private val activity = VoiceActivity()
        private var size = 0

        val full: Boolean get() = size == captured.size
        val remaining: Int get() = captured.size - size

        fun append(source: ShortArray, count: Int) {
            require(count in 1..source.size && count <= remaining) { "Invalid speech sample block" }
            source.copyInto(captured, destinationOffset = size, endIndex = count)
            size += count
            activity.update(captured, size)
            if (onSamples(size, activity.speechEnd)) onSnapshot(captured.toWaveform(size))
        }

        fun recording(): Recording {
            require(size > 0) { "No speech was recorded" }
            return Recording(captured.toWaveform(size), activity.speechEnd)
        }

        private fun ShortArray.toWaveform(size: Int) = FloatArray(size) { this[it] / 32768f }
    }

    /**
     * Energy-based speech tracking over 20 ms frames. A frame is speech when it clearly exceeds both a slowly rising
     * noise-floor estimate and a fraction of the loudest speech so far; thresholds err toward counting speech, which
     * only costs a final transcription pass.
     */
    private class VoiceActivity {
        var speechEnd = 0
            private set
        private var analyzed = 0
        private var floor = Float.MAX_VALUE
        private var peak = 0f

        fun update(samples: ShortArray, size: Int) {
            while (analyzed + FRAME_SAMPLES <= size) {
                var energy = 0.0
                for (index in analyzed until analyzed + FRAME_SAMPLES) {
                    val value = samples[index] / 32768.0
                    energy += value * value
                }
                val level = kotlin.math.sqrt(energy / FRAME_SAMPLES).toFloat()
                analyzed += FRAME_SAMPLES
                floor = minOf(level, if (floor == Float.MAX_VALUE) level else floor * FLOOR_RISE)
                if (level >= maxOf(MINIMUM_SPEECH_LEVEL, floor * FLOOR_RATIO, peak * PEAK_RATIO)) {
                    peak = maxOf(peak, level)
                    speechEnd = analyzed
                }
            }
        }
    }

    private companion object {
        const val SAMPLE_RATE = 16000
        const val MAXIMUM_SECONDS = 30
        const val MAXIMUM_SAMPLES = SAMPLE_RATE * MAXIMUM_SECONDS
        const val FRAME_SAMPLES = SAMPLE_RATE / 50
        const val MINIMUM_SPEECH_LEVEL = 0.002f
        const val FLOOR_RATIO = 2.5f
        const val PEAK_RATIO = 0.05f
        const val FLOOR_RISE = 1.002f
    }
}