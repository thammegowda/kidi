package ai.gowda.kidi

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.coroutineContext

internal class LiteAppServiceHost(
    context: Context,
    private val sendText: (String) -> Unit,
    private val requestMicrophonePermission: suspend () -> Boolean,
    private val fatal: (String) -> Unit,
) : Closeable {
    private class ServiceFailure(val code: String, message: String) : Exception(message)

    private data class ImageUpload(
        val size: Int,
        val contentType: String,
        var sequence: Int = 0,
        val bytes: ByteArrayOutputStream = ByteArrayOutputStream(),
    )

    private data class AsrInput(
        val requestId: String,
        val streamId: Int,
        val language: String,
        val voiceMode: Boolean,
        var partialRunning: Boolean = false,
    )

    private val appContext = context.applicationContext
    private val preferences = appContext.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
    private val repository = ModelRepository(appContext, "liteapp-models")
    private val rootJob = SupervisorJob()
    private val scope = CoroutineScope(rootJob + Dispatchers.Default)
    private val runtimeDispatcher = singleThread("kidi-liteapp-runtime")
    private val speechDispatcher = singleThread("kidi-liteapp-speech")
    private val ttsDispatcher = singleThread("kidi-liteapp-tts")
    private val jobs = ConcurrentHashMap<String, Job>()
    private val methods = ConcurrentHashMap<String, String>()
    private val nativeRequests = ConcurrentHashMap<String, Long>()
    private val sequences = ConcurrentHashMap<String, AtomicInteger>()
    private val imageLock = Any()
    private val imageUploads = mutableMapOf<Int, ImageUpload>()
    private val images = mutableMapOf<Int, File>()
    private val asrLock = Any()
    private var asrInput: AsrInput? = null
    private val audioRecorder = AndroidAudioRecorder()
    private val audioPlayer = AndroidAudioPlayer()
    private val imageDirectory = File(appContext.cacheDir, "liteapp-images")
    private val inferenceMutex = Mutex()

    @Volatile private var chatLoaded = false
    @Volatile private var asrLoaded = false
    @Volatile private var ttsLoaded = false
    @Volatile private var chatResident = false
    @Volatile private var asrResident = false
    @Volatile private var ttsResident = false
    @Volatile private var chatDirectory: String? = null
    @Volatile private var asrDirectory: String? = null
    @Volatile private var ttsDirectory: String? = null
    @Volatile private var chatBackend = ""
    @Volatile private var asrBackend = ""
    @Volatile private var ttsBackend = ""
    @Volatile private var chatAccelerator = DEFAULT_CHAT_ACCELERATOR
    @Volatile private var threads = preferences.getInt(THREADS_KEY, defaultThreads()).coerceIn(1, 8)
    @Volatile private var maximumTokens = preferences.getInt(TOKENS_KEY, 512).coerceIn(1, 8192)
    @Volatile private var modelId = DEFAULT_CHAT_MODEL
    @Volatile private var systemPrompt = DEFAULT_SYSTEM_PROMPT

    init {
        val savedAccelerator = preferences.getString(CHAT_ACCELERATOR_KEY, null)
            ?.takeIf(ACCELERATORS::contains)
        val migrateAuto = preferences.getInt(ACCELERATOR_POLICY_VERSION_KEY, 0) < ACCELERATOR_POLICY_VERSION &&
            savedAccelerator == "auto"
        chatAccelerator = if (migrateAuto) DEFAULT_CHAT_ACCELERATOR else savedAccelerator ?: DEFAULT_CHAT_ACCELERATOR
        preferences.edit {
            putInt(ACCELERATOR_POLICY_VERSION_KEY, ACCELERATOR_POLICY_VERSION)
            if (migrateAuto) putString(CHAT_ACCELERATOR_KEY, chatAccelerator)
        }
        repository.installed()?.let { installed ->
            chatLoaded = true
            chatDirectory = installed.directory.absolutePath
            modelId = installed.modelId
        }
        repository.installedSpeech()?.let { installed ->
            asrLoaded = true
            asrDirectory = installed.directory.absolutePath
        }
        repository.installedTts()?.let { directory ->
            ttsLoaded = true
            ttsDirectory = directory.absolutePath
        }
        scope.launch(runtimeDispatcher) {
            checked(NativeRuntime.setDataDirectory(
                appContext.filesDir.absolutePath,
                appContext.applicationInfo.nativeLibraryDir,
            ))
        }
    }

    fun receiveText(source: String) {
        val message = try {
            JSONObject(source)
        } catch (_: Exception) {
            fatal("Invalid LiteApp service envelope")
            return
        }
        if (message.optInt("version") != 1 || message.optString("id").isBlank()) {
            fatal("Invalid LiteApp service envelope")
            return
        }
        val id = message.getString("id")
        when (message.optString("kind")) {
            "cancel" -> cancel(id)
            "request" -> request(id, message.optString("method"), message.optJSONObject("payload") ?: JSONObject())
            else -> fatal("Invalid LiteApp service request")
        }
    }

    fun receiveBinary(data: ByteArray) {
        if (data.size < HEADER_BYTES) {
            fatal("Binary service frame is truncated")
            return
        }
        val frame = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
        if (frame.get(0) != 'K'.code.toByte() || frame.get(1) != 'D'.code.toByte() ||
            frame.get(2) != 'A'.code.toByte() || frame.get(3) != '1'.code.toByte() ||
            frame.get(4).toInt() != 1 || frame.getInt(20) != data.size - HEADER_BYTES) {
            fatal("Binary service frame is invalid")
            return
        }
        val kind = frame.get(5).toInt()
        val streamId = frame.getInt(8)
        val sequence = frame.getInt(12)
        frame.position(HEADER_BYTES)
        try {
            when (kind) {
                IMAGE_INPUT -> appendImage(streamId, sequence, data, HEADER_BYTES)
                else -> fatal("Unknown Android binary service frame")
            }
        } catch (error: Exception) {
            fatal(error.message ?: "Unable to process Android binary service frame")
        }
    }

    private fun request(id: String, method: String, payload: JSONObject) {
        if (method.isBlank()) {
            error(id, "invalid_request", "Service method is required")
            return
        }
        val job = scope.launch(start = CoroutineStart.LAZY) {
            try {
                val result = dispatch(method, id, payload)
                terminal(id, "completed", result)
            } catch (cancelled: CancellationException) {
                terminal(id, "cancelled", JSONObject().put("message", "Request cancelled"))
            } catch (failure: ServiceFailure) {
                terminal(id, "error", JSONObject().put("code", failure.code).put("message", failure.message))
                Log.e(TAG, "request_failed id=$id method=$method code=${failure.code}", failure)
            } catch (failure: Exception) {
                terminal(id, "error", JSONObject()
                    .put("code", "service_error")
                    .put("message", failure.message ?: "Android service request failed"))
                Log.e(TAG, "request_failed id=$id method=$method code=service_error", failure)
            } finally {
                jobs.remove(id)
                methods.remove(id)
                sequences.remove(id)
            }
        }
        if (jobs.putIfAbsent(id, job) != null) {
            error(id, "duplicate_request", "Request ID is already active")
            return
        }
        methods[id] = method
        job.start()
    }

    private suspend fun dispatch(method: String, id: String, payload: JSONObject): JSONObject = when (method) {
        "session.status" -> snapshot()
        "permissions.microphone.request" -> requestMicrophone()
        "runtime.restore" -> restoreRuntime(id)
        "runtime.load" -> loadChat(id, payload)
        "asr.preload" -> loadAsr(id)
        "tts.preload" -> loadTts(id)
        "settings.system_prompt.set" -> setSystemPrompt(payload)
        "image.start" -> startImage(payload)
        "image.finish" -> finishImage(payload)
        "chat.start" -> runChat(id, payload)
        "reader.translate" -> runTranslation(id, payload)
        "asr.start" -> runAsr(id, payload)
        "asr.stop" -> stopAsr(payload)
        "tts.start" -> runTts(id, payload)
        else -> throw ServiceFailure("unknown_method", "Unknown service method: $method")
    }

    private suspend fun requestMicrophone(): JSONObject {
        if (!requestMicrophonePermission())
            throw ServiceFailure("microphone_permission", "Microphone permission is required for dictation")
        return JSONObject().put("granted", true)
    }

    private suspend fun restoreRuntime(id: String): JSONObject {
        if (chatLoaded && !chatResident) {
            event(id, "runtime.loading", JSONObject().put("runtime", runtime().put("loading", true)))
            inferenceMutex.withLock { ensureChatResident() }
        }
        return JSONObject()
            .put("ready", chatResident)
            .put("backend", chatBackend)
            .put("runtime", runtime())
            .put("components", components())
    }

    private suspend fun loadChat(id: String, payload: JSONObject): JSONObject {
        val requested = payload.optString("model", DEFAULT_CHAT_MODEL)
        if (!MODEL_ID.matches(requested))
            throw ServiceFailure("invalid_model", "Chat model must be a Hugging Face OWNER/REPO ID")
        val requestedThreads = payload.optInt("threads", threads)
        if (requestedThreads !in 1..8) throw ServiceFailure("invalid_threads", "CPU threads must be between 1 and 8")
        val requestedTokens = payload.optInt("maximum_tokens", maximumTokens)
        if (requestedTokens !in 1..8192)
            throw ServiceFailure("invalid_tokens", "Output tokens must be between 1 and 8192")
        val requestedAccelerator = payload.optString("backend", chatAccelerator)
        if (requestedAccelerator !in ACCELERATORS)
            throw ServiceFailure("invalid_accelerator", "Accelerator must be auto, npu, gpu, or cpu")
        threads = requestedThreads
        maximumTokens = requestedTokens
        chatAccelerator = requestedAccelerator
        preferences.edit {
            putInt(THREADS_KEY, threads)
            putInt(TOKENS_KEY, maximumTokens)
            putString(CHAT_ACCELERATOR_KEY, chatAccelerator)
        }
        event(id, "runtime.loading", JSONObject().put("runtime", runtime().put("loading", true)))
        val installed = repository.prepare(requested) { progress(id, "runtime.progress", it) }
        event(id, "runtime.progress", JSONObject().put("type", "initializing").put("file", "Native runtime"))
        val loaded = inferenceMutex.withLock {
            withContext(runtimeDispatcher) {
                checked(NativeRuntime.configure(threads))
                checked(NativeRuntime.load(installed.directory.absolutePath, chatAccelerator))
            }
        }
        modelId = requested
        chatLoaded = true
        chatResident = true
        ttsResident = false
        chatDirectory = installed.directory.absolutePath
        chatBackend = loaded.optString("backend", "android-cpu")
        return JSONObject()
            .put("ready", true)
            .put("backend", chatBackend)
            .put("runtime", runtime())
            .put("components", components())
    }

    private suspend fun loadAsr(id: String): JSONObject {
        event(id, "asr.progress", JSONObject()
            .put("phase", "loading").put("stage", "Preparing speech recognition model"))
        val installed = repository.prepareSpeech(DEFAULT_ASR_MODEL) { progress(id, "asr.progress", it) }
        val loaded = inferenceMutex.withLock {
            withContext(speechDispatcher) {
                checked(NativeRuntime.configure(threads))
                checked(NativeRuntime.loadAsr(installed.directory.absolutePath, true, ASR_ACCELERATOR))
            }
        }
        asrLoaded = true
        asrResident = true
        ttsResident = false
        asrDirectory = installed.directory.absolutePath
        asrBackend = loaded.optString("backend", "android-cpu")
        return JSONObject().put("component", asrComponent()).put("runtime", runtime())
    }

    private suspend fun loadTts(id: String): JSONObject {
        event(id, "tts.progress", JSONObject()
            .put("phase", "loading").put("stage", "Preparing TTS (text-to-speech) model"))
        val installed = repository.prepareTts(DEFAULT_TTS_MODEL) { progress(id, "tts.progress", it) }
        val loaded = inferenceMutex.withLock {
            withContext(ttsDispatcher) {
                checked(NativeRuntime.configure(threads))
                checked(NativeRuntime.loadTts(installed.directory.absolutePath, TTS_ACCELERATOR))
            }
        }
        ttsLoaded = true
        ttsResident = true
        chatResident = false
        asrResident = false
        ttsDirectory = installed.directory.absolutePath
        ttsBackend = loaded.optString("backend", "android-cpu")
        return JSONObject().put("component", ttsComponent()).put("runtime", runtime())
    }

    private fun setSystemPrompt(payload: JSONObject): JSONObject {
        val prompt = payload.optString("prompt")
        if (prompt.length > 32768) throw ServiceFailure("invalid_system_prompt", "System prompt is too long")
        systemPrompt = prompt
        return JSONObject().put("prompt", prompt)
    }

    private fun startImage(payload: JSONObject): JSONObject {
        val streamId = unsignedInt(payload, "stream_id")
        val size = payload.optLong("size", -1).toInt()
        val contentType = payload.optString("content_type")
        if (size !in 1..MAX_IMAGE_BYTES || contentType !in setOf("image/jpeg", "image/png"))
            throw ServiceFailure("invalid_image", "Image upload must be JPEG or PNG and no larger than 32 MiB")
        synchronized(imageLock) {
            if (imageUploads.containsKey(streamId)) throw ServiceFailure("duplicate_stream", "Image stream already exists")
            imageUploads[streamId] = ImageUpload(size, contentType)
        }
        return JSONObject().put("stream_id", streamId)
    }

    private fun appendImage(streamId: Int, sequence: Int, data: ByteArray, offset: Int) {
        synchronized(imageLock) {
            val upload = imageUploads[streamId] ?: throw ServiceFailure("unknown_stream", "Image stream is unavailable")
            require(upload.sequence == sequence) { "Image chunks are out of order" }
            require(upload.bytes.size() + data.size - offset <= upload.size) { "Image stream exceeds its announced size" }
            upload.bytes.write(data, offset, data.size - offset)
            upload.sequence++
        }
    }

    private fun finishImage(payload: JSONObject): JSONObject {
        val streamId = unsignedInt(payload, "stream_id")
        val upload = synchronized(imageLock) {
            imageUploads.remove(streamId) ?: throw ServiceFailure("unknown_stream", "Image stream is unavailable")
        }
        if (upload.bytes.size() != upload.size)
            throw ServiceFailure("invalid_image", "Image upload is incomplete")
        check(imageDirectory.isDirectory || imageDirectory.mkdirs()) { "Unable to create image cache" }
        val extension = if (upload.contentType == "image/png") ".png" else ".jpg"
        val target = File(imageDirectory, "$streamId$extension")
        target.writeBytes(upload.bytes.toByteArray())
        synchronized(imageLock) { images.put(streamId, target)?.delete() }
        return JSONObject().put("stream_id", streamId)
    }

    private suspend fun runChat(id: String, payload: JSONObject): JSONObject {
        if (!chatLoaded) throw ServiceFailure("not_ready", "Load the Android chat model first")
        val messages = nativeMessages(payload.optJSONArray("messages")
            ?: throw ServiceFailure("invalid_messages", "messages is required"))
        val languages = payload.optString("languages")
        val languageNames = payload.optString("language_names", languages)
        if (languages.isNotBlank() && !LANGUAGES.matches(languages))
            throw ServiceFailure("invalid_languages", "Languages must be comma-separated two- or three-letter codes")
        if (languages.isNotBlank()) {
            return generate(id, prependSystem(messages, "$systemPrompt\nRespond entirely in $languageNames."), "chat")
        } else if (systemPrompt.isNotBlank()) {
            return generate(id, prependSystem(messages, systemPrompt), "chat")
        }
        return generate(id, messages, "chat")
    }

    private suspend fun runTranslation(id: String, payload: JSONObject): JSONObject {
        if (!chatLoaded) throw ServiceFailure("not_ready", "Load the Android chat model first")
        val text = payload.optString("text")
        val previous = payload.optString("previous")
        val next = payload.optString("next")
        val target = payload.optString("target_language_name")
        if (text.isBlank() || target.isBlank() || text.length > 16384 ||
            previous.length > 16384 || next.length > 16384)
            throw ServiceFailure("invalid_translation", "Reader translation payload is invalid or too large")
        val prompt = "You are a literary translator. Translate <text_to_translate> into $target. " +
            "Use surrounding context only to resolve meaning and return only the translated text."
        val request = "<preceding_context>\n$previous\n</preceding_context>\n" +
            "<text_to_translate>\n$text\n</text_to_translate>\n" +
            "<following_context>\n$next\n</following_context>"
        val messages = JSONArray()
            .put(JSONObject().put("role", "system").put("content", prompt))
            .put(JSONObject().put("role", "user").put("content", request))
        return generate(id, messages, "reader.translation")
    }

    private suspend fun generate(id: String, messages: JSONArray, eventPrefix: String): JSONObject =
        inferenceMutex.withLock {
            if (!chatResident)
                event(id, "runtime.loading", JSONObject().put("runtime", runtime().put("loading", true)))
            ensureChatResident()
            withContext(runtimeDispatcher) {
                val started = System.nanoTime()
                event(id, "$eventPrefix.started", JSONObject()
                    .put("backend", chatBackend)
                    .put("runtime", runtime())
                    .put("components", components()))
                val requestId = checked(NativeRuntime.enqueue(messages.toString(), maximumTokens)).getLong("request_id")
                nativeRequests[id] = requestId
                try {
                    var completed: JSONObject? = null
                    while (completed == null) {
                        coroutineContext.ensureActive()
                        val step = checked(NativeRuntime.step())
                        val events = step.getJSONArray("events")
                        for (index in 0 until events.length()) {
                            val item = events.getJSONObject(index)
                            item.optString("text").takeIf(String::isNotEmpty)?.let {
                                event(id, "$eventPrefix.delta", JSONObject().put("text", it))
                            }
                            item.optJSONObject("completed")?.let { completed = it }
                        }
                        if (step.optInt("pending") == 0 && completed == null)
                            throw ServiceFailure("inference_failed", "Native generation ended without a result")
                    }
                    val result = requireNotNull(completed)
                    JSONObject()
                        .put("text", result.optString("text"))
                        .put("stats", JSONObject()
                            .put("elapsed_ms", (System.nanoTime() - started) / 1e6)
                            .put("prompt_tokens", result.optInt("prompt_tokens"))
                            .put("decode_tokens", result.optInt("decode_tokens"))
                            .put("prefill_ms", result.optDouble("prefill_ms"))
                            .put("decode_ms", result.optDouble("decode_ms"))
                            .put("preparation_ms", 0)
                            .put("first_token_ms", result.optDouble("first_token_ms")))
                } finally {
                    nativeRequests.remove(id)
                }
            }
        }

    private suspend fun runAsr(id: String, payload: JSONObject): JSONObject {
        if (!asrLoaded) loadAsr(id)
        val streamId = unsignedInt(payload, "stream_id")
        val language = payload.optString("language", "auto")
        val voiceMode = payload.optBoolean("voice_mode")
        if (!LANGUAGES.matches(language))
            throw ServiceFailure("invalid_asr", "ASR language must be auto or comma-separated language codes")
        val input = AsrInput(id, streamId, language, voiceMode)
        synchronized(asrLock) {
            if (asrInput != null) throw ServiceFailure("busy", "Speech recognition is already active")
            asrInput = input
        }
        try {
            inferenceMutex.withLock { ensureAsrResident() }
            event(id, "asr.started", JSONObject()
                .put("stream_id", streamId)
                .put("sample_rate", AndroidAudioRecorder.SAMPLE_RATE)
                .put("native_capture", true))
            if (ContextCompat.checkSelfPermission(appContext, Manifest.permission.RECORD_AUDIO) !=
                PackageManager.PERMISSION_GRANTED)
                throw ServiceFailure("microphone_permission", "Microphone permission was revoked")
            val vad = asrComponent().getJSONObject("settings").getJSONObject("vad")
            val capture = audioRecorder.capture(
                voiceMode = voiceMode,
                threshold = vad.getDouble("threshold").toFloat(),
                minimumSpeechMs = vad.getInt("minimum_speech_ms"),
                silenceMs = vad.getInt("silence_ms"),
            ) { samples ->
                publishAsrSnapshot(input, samples)
            }
            val result = inferenceMutex.withLock {
                ensureAsrResident()
                withContext(speechDispatcher) {
                    checked(NativeRuntime.transcribe(capture.samples, language, 128))
                }
            }
            return JSONObject()
                .put("text", result.optString("text"))
                .put("language", result.optString("language", "auto"))
                .put("stats", JSONObject()
                    .put("feature_ms", result.optDouble("feature_ms"))
                    .put("encode_ms", result.optDouble("encode_ms"))
                    .put("decode_ms", result.optDouble("decode_ms")))
        } finally {
            synchronized(asrLock) { if (asrInput === input) asrInput = null }
        }
    }

    private fun publishAsrSnapshot(input: AsrInput, samples: FloatArray) {
        synchronized(asrLock) {
            if (asrInput !== input || input.partialRunning) return
            input.partialRunning = true
        }
        scope.launch(speechDispatcher) {
            try {
                val result = inferenceMutex.withLock {
                    ensureAsrResident()
                    checked(NativeRuntime.transcribe(samples, input.language, 128))
                }
                event(input.requestId, "asr.partial", JSONObject()
                    .put("stable", "")
                    .put("provisional", result.optString("text"))
                    .put("language", result.optString("language", "auto")))
            } catch (_: Exception) {
            } finally {
                synchronized(asrLock) {
                    if (asrInput === input) input.partialRunning = false
                }
            }
        }
    }

    private fun stopAsr(payload: JSONObject): JSONObject {
        val requestId = payload.optString("request_id")
        val stopped = synchronized(asrLock) {
            asrInput?.takeIf { it.requestId == requestId } != null
        }
        if (stopped) audioRecorder.stop()
        return JSONObject().put("stopped", stopped)
    }

    private suspend fun runTts(id: String, payload: JSONObject): JSONObject {
        if (!ttsLoaded) loadTts(id)
        val streamId = unsignedInt(payload, "stream_id")
        val text = payload.optString("text")
        if (text.isBlank() || text.length > 65536)
            throw ServiceFailure("invalid_tts", "TTS text must contain 1-65536 characters")
        val options = payload.optJSONObject("options") ?: JSONObject()
        val language = options.optString("language", "English")
        val steps = options.optInt("steps", 10)
        if (steps !in 1..128) throw ServiceFailure("invalid_tts", "TTS steps must be between 1 and 128")
        val voice = options.optJSONObject("voice") ?: JSONObject()
        event(id, "tts.started", JSONObject().put("stream_id", streamId).put("format", "native"))
        val audio = inferenceMutex.withLock {
            ensureTtsResident()
            withContext(ttsDispatcher) {
                NativeRuntime.synthesize(text, language, steps, voice.toString()) { completed, total ->
                    event(id, "tts.progress", JSONObject()
                        .put("phase", "synthesizing").put("step", completed).put("steps", total))
                }
            }
        }
        val sampleRate = NativeRuntime.ttsSampleRate()
        event(id, "tts.progress", JSONObject().put("phase", "playing"))
        audioPlayer.play(audio, sampleRate)
        return JSONObject()
            .put("stream_id", streamId)
            .put("sample_rate", sampleRate)
            .put("samples", audio.size)
            .put("chunks", 1)
            .put("native_playback", true)
            .put("stats", JSONObject()
                .put("generation_ms", 0)
                .put("decode_ms", 0)
                .put("preparation_ms", 0)
                .put("audio_tokens", 0))
    }

    private fun nativeMessages(source: JSONArray): JSONArray {
        if (source.length() !in 1..128) throw ServiceFailure("invalid_messages", "messages must not be empty")
        val result = JSONArray()
        for (index in 0 until source.length()) {
            val input = source.getJSONObject(index)
            val output = JSONObject()
                .put("role", input.getString("role"))
                .put("content", input.optString("content"))
            input.optJSONArray("images")?.let { sourceImages ->
                val paths = JSONArray()
                synchronized(imageLock) {
                    for (image in 0 until sourceImages.length()) {
                        val path = images[sourceImages.getInt(image)]
                            ?: throw ServiceFailure("invalid_messages", "Message image is unavailable")
                        paths.put(path.absolutePath)
                    }
                }
                output.put("images", paths)
            }
            result.put(output)
        }
        return result
    }

    private fun prependSystem(messages: JSONArray, prompt: String): JSONArray {
        val result = JSONArray().put(JSONObject().put("role", "system").put("content", prompt))
        for (index in 0 until messages.length()) result.put(messages.get(index))
        return result
    }

    private suspend fun ensureChatResident() {
        if (chatResident) return
        val directory = chatDirectory ?: throw ServiceFailure("not_ready", "Load the Android chat model first")
        val loaded = withContext(runtimeDispatcher) {
            checked(NativeRuntime.configure(threads))
            checked(NativeRuntime.load(directory, chatAccelerator))
        }
        chatBackend = loaded.optString("backend", "android-cpu")
        chatResident = true
        ttsResident = false
    }

    private suspend fun ensureAsrResident() {
        if (asrResident) return
        val directory = asrDirectory ?: throw ServiceFailure("asr_not_ready", "Load the Android ASR model first")
        val loaded = withContext(speechDispatcher) {
            checked(NativeRuntime.configure(threads))
            checked(NativeRuntime.loadAsr(directory, true, ASR_ACCELERATOR))
        }
        asrBackend = loaded.optString("backend", "android-cpu")
        asrResident = true
        ttsResident = false
    }

    private suspend fun ensureTtsResident() {
        if (ttsResident) return
        val directory = ttsDirectory ?: throw ServiceFailure("tts_not_ready", "Load the Android TTS model first")
        val loaded = withContext(ttsDispatcher) {
            checked(NativeRuntime.configure(threads))
            checked(NativeRuntime.loadTts(directory, TTS_ACCELERATOR))
        }
        ttsBackend = loaded.optString("backend", "android-cpu")
        ttsResident = true
        chatResident = false
        asrResident = false
    }

    private fun snapshot() = JSONObject()
        .put("ready", chatResident)
        .put("config", "Android app-private settings")
        .put("runtime", runtime())
        .put("components", components())
        .put("model", modelId)
        .put("backend", chatBackend)
        .put("vision", true)

    private fun runtime() = JSONObject()
        .put("host", "android")
        .put("threads", threads)
        .put("hardware_concurrency", Runtime.getRuntime().availableProcessors())
        .put("concurrency", "native")
        .put("model_hub_remote", "https://huggingface.co")
        .put("model_hub_local", "Android app-private storage")
        .put("speech_backend", ttsBackend.ifBlank { asrBackend })
        .put("preload_speech", false)
        .put("idle_model_timeout_seconds", 300)
        .put("model_id", modelId)
        .put("requested_backend", chatAccelerator)
        .put("resident", chatResident)
        .put("maximum_tokens", maximumTokens)
        .put("loading", chatLoaded && !chatResident)

    private fun components() = JSONObject()
        .put("chat", JSONObject()
            .put("enabled", true)
            .put("loaded", chatLoaded)
            .put("model_id", modelId)
            .put("directory", modelId)
            .put("accelerator", chatBackend.ifBlank { "android-cpu" })
            .put("precision", "checkpoint")
            .put("kv_cache_precision", "auto")
            .put("system_prompt", systemPrompt)
            .put("configured_system_prompt", DEFAULT_SYSTEM_PROMPT)
            .put("translation_prompt", "Translate the selected reader text into {target_language}.")
            .put("translation", JSONObject()
                .put("target_chars", 800).put("max_chars", 1200)
                .put("max_sentence_chars", 600).put("max_input_tokens", 1536))
            .put("maximum_tokens", maximumTokens)
            .put("context_tokens", 9216)
            .put("prefix_cache_mb", 512))
        .put("asr", asrComponent())
        .put("tts", ttsComponent())

    private fun asrComponent() = JSONObject()
        .put("enabled", true)
        .put("loaded", asrLoaded)
        .put("model_id", DEFAULT_ASR_MODEL)
        .put("directory", DEFAULT_ASR_MODEL)
        .put("accelerator", asrBackend.ifBlank { "android-cpu" })
        .put("settings", JSONObject()
            .put("language", "auto").put("max_tokens", 128)
            .put("vad", JSONObject()
                .put("threshold", 0.012).put("minimum_speech_ms", 250).put("silence_ms", 800)))

    private fun ttsComponent() = JSONObject()
        .put("enabled", true)
        .put("loaded", ttsLoaded)
        .put("model_id", DEFAULT_TTS_MODEL)
        .put("directory", DEFAULT_TTS_MODEL)
        .put("accelerator", ttsBackend.ifBlank { "android-auto" })
        .put("settings", JSONObject()
            .put("language", "English").put("steps", 10).put("realtime_steps", 10)
            .put("pace", 0.74).put("reference_language", "English")
            .put("reference_text", "Nice to meet you.")
            .put("voice", JSONObject()
                .put("gender", "female").put("age", "young adult")
                .put("pitch", "moderate").put("style", "").put("accent", "")))

    private fun progress(id: String, type: String, progress: ModelDownloadProgress) {
        event(id, type, JSONObject()
            .put("phase", progress.phase.name.lowercase())
            .put("stage", progress.phase.label)
            .put("file", progress.file)
            .put("loadedBytes", progress.completedBytes)
            .put("totalBytes", progress.totalBytes)
            .put("downloadedBytes", progress.completedBytes))
    }

    private fun event(id: String, type: String, payload: JSONObject = JSONObject()) {
        val sequence = sequences.computeIfAbsent(id) { AtomicInteger() }.getAndIncrement()
        sendText(JSONObject()
            .put("version", 1)
            .put("kind", "event")
            .put("id", id)
            .put("sequence", sequence)
            .put("type", type)
            .put("payload", payload)
            .toString())
    }

    private fun terminal(id: String, type: String, payload: JSONObject) = event(id, type, payload)

    private fun error(id: String, code: String, message: String) =
        event(id, "error", JSONObject().put("code", code).put("message", message))

    private fun cancel(id: String) {
        when (methods[id]) {
            "chat.start", "reader.translate" -> nativeRequests[id]?.let { requestId ->
                scope.launch(runtimeDispatcher) { runCatching { NativeRuntime.cancel(requestId) } }
            }
            "asr.start" -> {
                NativeRuntime.cancelTranscription()
                audioRecorder.stop()
            }
            "tts.start" -> {
                NativeRuntime.cancelTts()
                audioPlayer.stop()
            }
        }
        jobs[id]?.cancel()
    }

    private fun checked(source: String): JSONObject {
        val result = JSONObject(source)
        result.optString("error").takeIf(String::isNotBlank)?.let {
            throw ServiceFailure("native_error", it)
        }
        return result
    }

    private fun unsignedInt(payload: JSONObject, name: String): Int {
        val value = payload.optLong(name, -1)
        if (value !in 0..0xffffffffL) throw ServiceFailure("invalid_stream", "$name must be unsigned")
        return value.toInt()
    }

    override fun close() {
        scope.cancel()
        NativeRuntime.cancelTranscription()
        NativeRuntime.cancelTts()
        audioRecorder.stop()
        audioPlayer.stop()
        NativeRuntime.unload()
        NativeRuntime.unloadAsr()
        NativeRuntime.unloadTts()
        synchronized(imageLock) {
            images.values.forEach(File::delete)
            images.clear()
            imageUploads.clear()
        }
        listOf(runtimeDispatcher, speechDispatcher, ttsDispatcher)
            .filterIsInstance<Closeable>()
            .forEach(Closeable::close)
    }

    private companion object {
        const val HEADER_BYTES = 24
        const val IMAGE_INPUT = 3
        const val MAX_IMAGE_BYTES = 32 * 1024 * 1024
        const val DEFAULT_CHAT_MODEL = "google/gemma-4-E2B-it-qat-mobile-transformers"
        const val DEFAULT_ASR_MODEL = "openai/whisper-small"
        const val DEFAULT_TTS_MODEL = "thammegowda/OmniVoice-q8"
        const val DEFAULT_CHAT_ACCELERATOR = "cpu"
        const val ASR_ACCELERATOR = "cpu"
        const val TTS_ACCELERATOR = "auto"
        const val PREFERENCES = "kidi-liteapp-runtime-v1"
        const val CHAT_ACCELERATOR_KEY = "chat-accelerator"
        const val ACCELERATOR_POLICY_VERSION_KEY = "accelerator-policy-version"
        const val ACCELERATOR_POLICY_VERSION = 1
        const val THREADS_KEY = "threads"
        const val TOKENS_KEY = "tokens"
        const val DEFAULT_SYSTEM_PROMPT =
            "You are a helpful local AI assistant. Respond directly, briefly, respectfully, and safely."
        val MODEL_ID = Regex("[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
        val LANGUAGES = Regex("(?:auto|[a-z]{2,3}(?:,[a-z]{2,3})*)")
        val ACCELERATORS = setOf("auto", "npu", "gpu", "cpu")
        const val TAG = "KidiLiteApp"

        fun defaultThreads() = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)

        fun singleThread(name: String): CoroutineDispatcher =
            Executors.newSingleThreadExecutor { runnable -> Thread(runnable, name) }.asCoroutineDispatcher()
    }
}
