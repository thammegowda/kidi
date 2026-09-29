package ai.gowda.kidi

import android.content.ContextWrapper
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assume.assumeTrue
import org.junit.Test

class NativeRuntimeTest {
    @Test
    fun restoresLegacyAndQuantizedSpeechLayouts() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.cacheDir, "speech-layout-${UUID.randomUUID()}").apply { mkdirs() }
        val scopedContext = object : ContextWrapper(context) {
            override fun getExternalFilesDir(type: String?): File = directory
        }
        val repository = ModelRepository(scopedContext)
        val root = File(directory, "models").apply { mkdirs() }
        val revision = "a".repeat(40)
        val weightsRevision = "b".repeat(40)
        fun fixture(name: String, weights: String): File = File(root, "openai-whisper-small/$name").apply {
            mkdirs()
            listOf(weights, "config.json", "tokenizer.json", "preprocessor_config.json", "generation_config.json")
                .forEach { File(this, it).writeText("fixture") }
            File(this, "ready.json").writeText(JSONObject().put("revision", revision).toString())
        }
        try {
            val legacy = fixture(revision, "model.safetensors")
            val current = JSONObject().put("model_id", "openai/whisper-small").put("revision", revision)
            File(root, "speech-current.json").writeText(current.toString())
            assertEquals(legacy, repository.installedSpeech()?.directory)
            val quantized = fixture("$revision-ggml-$weightsRevision", "ggml-model.bin")
            current.put("weights_revision", weightsRevision)
            File(root, "speech-current.json").writeText(current.toString())
            assertEquals(quantized, repository.installedSpeech()?.directory)
            File(quantized, "ggml-model.bin").delete()
            assertNull(repository.installedSpeech())
            File(quantized, "ggml-model.bin").writeText("fixture")
            repository.removeInstalledSpeech()
            assertFalse(quantized.exists())
            assertTrue(File(legacy, "model.safetensors").isFile)
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun downloadsAndLoadsQuantizedSmall() = runBlocking {
        assumeTrue("Optional real network download", InstrumentationRegistry.getArguments()
            .getString("downloadWhisperSmall") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.cacheDir, "speech-download-${UUID.randomUUID()}").apply { mkdirs() }
        val scopedContext = object : ContextWrapper(context) {
            override fun getExternalFilesDir(type: String?): File = directory
        }
        try {
            val repository = ModelRepository(scopedContext)
            var downloadedBytes = 0L
            val installed = repository.prepareSpeech("openai/whisper-small") { update ->
                downloadedBytes = update.totalBytes
                assertFalse("FP32 checkpoint must not be downloaded", update.file.contains("model.safetensors"))
            }
            assertTrue("Expected compact Small Q8 download, got $downloadedBytes", downloadedBytes in 265_000_000L..275_000_000L)
            assertFalse(File(installed.directory, "model.safetensors").exists())
            assertTrue(File(installed.directory, "ggml-model.bin").length() in 260_000_000L..270_000_000L)
            val restored = ModelRepository(scopedContext).installedSpeech()
            assertEquals(installed.directory, restored?.directory)
            assertTrue(JSONObject(NativeRuntime.configure(4)).getBoolean("configured"))
            val loaded = JSONObject(NativeRuntime.loadAsr(installed.directory.absolutePath, true))
            assertTrue(loaded.toString(), loaded.optBoolean("ready"))
            assertEquals("int8", loaded.getString("precision"))
            assertTrue(loaded.getLong("model_bytes") in 240_000_000L..260_000_000L)
            val cache = File(installed.directory, "ggml-model.bin.kidi-int8-v1/model.safetensors")
            assertTrue(cache.isFile)
            val timestamp = cache.lastModified()
            NativeRuntime.unloadAsr()
            assertTrue(JSONObject(NativeRuntime.loadAsr(installed.directory.absolutePath, true)).getBoolean("ready"))
            assertEquals(timestamp, cache.lastModified())
        } finally {
            NativeRuntime.unloadAsr()
            directory.deleteRecursively()
        }
    }

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
            val cacheName = if (File(modelDirectory, "ggml-model.bin").isFile)
                "ggml-model.bin.kidi-int8-v1" else "kidi-int8-v2"
            val cache = File(modelDirectory, "$cacheName/model.safetensors")
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