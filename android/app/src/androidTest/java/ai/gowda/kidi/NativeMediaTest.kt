package ai.gowda.kidi

import android.Manifest
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeMediaTest {
    @Test
    fun capturesAndPlaysAudioWithoutWebViewMediaApis() = runBlocking {
        withTimeout(20_000) {
            val instrumentation = InstrumentationRegistry.getInstrumentation()
            instrumentation.uiAutomation.grantRuntimePermission(
                instrumentation.targetContext.packageName,
                Manifest.permission.RECORD_AUDIO,
            )
            val recorder = AndroidAudioRecorder()
            val recording = async {
                recorder.capture(
                    voiceMode = false,
                    threshold = 0.012f,
                    minimumSpeechMs = 250,
                    silenceMs = 800,
                    onSnapshot = {},
                )
            }
            delay(1_000)
            recorder.stop()
            val captured = recording.await()
            assertTrue(captured.samples.size >= AndroidAudioRecorder.SAMPLE_RATE / 2)

            val player = AndroidAudioPlayer()
            player.play(ShortArray(AndroidAudioRecorder.SAMPLE_RATE / 10), AndroidAudioRecorder.SAMPLE_RATE)
        }
    }
}
