package ai.gowda.kidi

import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

class NativeRuntimeTest {
    @Test
    fun configuresAndroidCpuRuntime() {
        val result = JSONObject(NativeRuntime.configure(1))

        assertTrue(result.getBoolean("configured"))
        assertEquals(1, result.getInt("threads"))
        assertEquals("android-cpu", result.getString("backend"))

        val missingModel = "/does-not-exist/kidi-\uD83C\uDF99-model"
        val load = JSONObject(NativeRuntime.load(missingModel))
        assertTrue(load.getString("error").contains("config does not exist"))
        assertTrue(load.getString("error").contains(missingModel))

        val loadAsr = JSONObject(NativeRuntime.loadAsr("/does-not-exist/kidi-speech-model"))
        assertTrue(loadAsr.getString("error").contains("not a model directory"))

        val transcribe = JSONObject(NativeRuntime.transcribe(floatArrayOf(0f), "auto", 1))
        assertTrue(transcribe.getString("error").contains("Load a speech model first"))
    }

    @Test
    fun transcribesCachedSmallInt8() {
        val arguments = InstrumentationRegistry.getArguments()
        val directory = arguments.getString("whisperSmallDirectory")
        val audio = arguments.getString("whisperSmallAudio")
        assumeTrue("Optional staged Whisper Small fixture", directory != null && audio != null)
        val samples = readAudio(requireNotNull(audio))
        val modelDirectory = requireNotNull(directory)
        if (arguments.getString("selectWhisperSmall") == "true") {
            val repository = ModelRepository(InstrumentationRegistry.getInstrumentation().targetContext)
            val installed = runBlocking { repository.prepareSpeech("openai/whisper-small") {} }
            assertEquals(File(modelDirectory).canonicalPath, installed.directory.canonicalPath)
        }
        NativeRuntime.unloadAsr()
        try {
            val configured = JSONObject(NativeRuntime.configure(4))
            assertTrue(configured.getBoolean("configured"))
            val loaded = JSONObject(NativeRuntime.loadAsr(modelDirectory, true))
            assertTrue(loaded.toString(), loaded.optBoolean("ready"))
            assertEquals("int8", loaded.getString("precision"))
            assertEquals("android-cpu", loaded.getString("backend"))
            assertTrue(loaded.getLong("model_bytes") in 240_000_000L..260_000_000L)
            val cache = File(modelDirectory, "kidi-int8-v2/model.safetensors")
            val timestamp = cache.lastModified()
            val draft = JSONObject(NativeRuntime.transcribe(samples.copyOfRange(0, 32000), "auto", 128))
            assertEquals("the capital of France's Paris.", draft.getString("text").trim())
            val final = JSONObject(NativeRuntime.transcribe(samples, "auto", 128))
            assertEquals("en", final.getString("language"))
            assertEquals("the capital of France's Paris. Binary search repeatedly divides a sorted list in half. This is a speech recognition test on a mobile phone.",
                final.getString("text").trim())
            NativeRuntime.unloadAsr()
            val reloaded = JSONObject(NativeRuntime.loadAsr(modelDirectory, true))
            assertTrue(reloaded.toString(), reloaded.optBoolean("ready"))
            assertEquals(timestamp, cache.lastModified())
            val partials = mutableListOf<String>()
            var completed = false
            val repeated = JSONObject(NativeRuntime.transcribe(samples, "auto", 128) { payload ->
                check(!completed) { "Partial transcript arrived after completion" }
                val partial = JSONObject(payload)
                check(partial.getString("language") == "en")
                partials += partial.getString("text")
            })
            completed = true
            assertTrue(repeated.toString(), !repeated.has("error"))
            assertTrue("Expected multiple intermediate texts", partials.distinct().size > 1)
            assertTrue(partials.any { it.length < repeated.getString("text").length })
            assertEquals(repeated.getString("text"), partials.last())
            assertEquals(final.getJSONArray("token_ids").toString(), repeated.getJSONArray("token_ids").toString())
            val failed = JSONObject(NativeRuntime.transcribe(samples, "auto", 128) {
                throw IllegalStateException("test callback failure")
            })
            assertEquals("Partial transcript listener failed", failed.getString("error"))
            val recovered = JSONObject(NativeRuntime.transcribe(samples, "auto", 128))
            assertEquals(final.getJSONArray("token_ids").toString(), recovered.getJSONArray("token_ids").toString())
        } finally {
            NativeRuntime.unloadAsr()
        }
    }

    private fun readAudio(file: String): FloatArray {
        val bytes = File(file).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals("RIFF", String(bytes, 0, 4, Charsets.US_ASCII))
        assertEquals("WAVE", String(bytes, 8, 4, Charsets.US_ASCII))
        var offset = 12
        var samples = floatArrayOf()
        var pcm = false
        while (offset + 8 <= bytes.size) {
            val size = buffer.getInt(offset + 4)
            require(size >= 0 && size <= bytes.size - offset - 8)
            when (String(bytes, offset, 4, Charsets.US_ASCII)) {
                "fmt " -> {
                    require(size >= 16)
                    assertEquals(1, buffer.getShort(offset + 8).toInt())
                    assertEquals(1, buffer.getShort(offset + 10).toInt())
                    assertEquals(16000, buffer.getInt(offset + 12))
                    assertEquals(16, buffer.getShort(offset + 22).toInt())
                    pcm = true
                }
                "data" -> samples = FloatArray(size / 2) { index -> buffer.getShort(offset + 8 + index * 2) / 32768f }
            }
            offset += 8 + size + (size and 1)
        }
        assertTrue(pcm && samples.isNotEmpty())
        return samples
    }
}