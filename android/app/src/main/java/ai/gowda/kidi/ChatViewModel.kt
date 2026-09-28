package ai.gowda.kidi

import android.Manifest
import android.app.Application
import androidx.annotation.RequiresPermission
import androidx.core.content.edit
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

private const val DEFAULT_MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
private const val DEFAULT_SPEECH_MODEL_ID = "openai/whisper-small"

internal enum class MessageRole { USER, ASSISTANT }

internal data class MessageStats(
    val tokens: Int,
    val elapsedMs: Double,
    val decodeMs: Double,
    val decodeTokens: Int,
)

internal data class ChatMessage(
    val role: MessageRole,
    val content: String,
    val stats: MessageStats? = null,
)

internal data class KidiUiState(
    val modelId: String = DEFAULT_MODEL_ID,
    val modelRevision: String? = null,
    val messages: List<ChatMessage> = emptyList(),
    val draft: String = "",
    val composerText: String = "",
    val modelReady: Boolean = false,
    val loadingModel: Boolean = false,
    val generating: Boolean = false,
    val stopping: Boolean = false,
    val progressFile: String = "",
    val progress: Float = 0f,
    val status: String = "Model offline",
    val error: String? = null,
    val threadCount: Int = defaultThreadCount(),
    val maximumTokens: Int = 1024,
    val speechModelId: String = DEFAULT_SPEECH_MODEL_ID,
    val speechModelRevision: String? = null,
    val speechReady: Boolean = false,
    val loadingSpeech: Boolean = false,
    val recording: Boolean = false,
    val transcribing: Boolean = false,
    val recordingSeconds: Float = 0f,
    val speechProgressFile: String = "",
    val speechProgress: Float = 0f,
)

internal class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private val preferences = application.getSharedPreferences(PREFERENCES, 0)
    private val repository = ModelRepository(application)
    private val speechRecorder = SpeechRecorder()
    private val runtimeExecutor = Executors.newSingleThreadExecutor { runnable -> Thread(runnable, "kidi-runtime") }
    private val runtimeDispatcher: CoroutineDispatcher = runtimeExecutor.asCoroutineDispatcher()
    private val _state = MutableStateFlow(
        KidiUiState(
            modelId = preferences.getString(MODEL_ID_KEY, DEFAULT_MODEL_ID) ?: DEFAULT_MODEL_ID,
            speechModelId = preferences.getString(SPEECH_MODEL_ID_KEY, DEFAULT_SPEECH_MODEL_ID)
                ?: DEFAULT_SPEECH_MODEL_ID,
            messages = readMessages(),
            threadCount = preferences.getInt(THREADS_KEY, defaultThreadCount()).coerceIn(1, 8),
            maximumTokens = preferences.getInt(TOKENS_KEY, 1024).coerceIn(1, 8192),
        ),
    )
    val state: StateFlow<KidiUiState> = _state.asStateFlow()

    @Volatile
    private var requestId: Long? = null
    private val stopRequested = AtomicBoolean()
    private var modelJob: Job? = null
    private var speechJob: Job? = null
    private var recordingJob: Job? = null

    init {
        val installed = repository.installed()
        if (installed != null) load(installed)
        val installedSpeech = repository.installedSpeech()
        if (installedSpeech != null) loadSpeech(installedSpeech)
    }

    fun setModelId(value: String) {
        _state.update { it.copy(modelId = value, error = null) }
    }

    fun setSpeechModelId(value: String) {
        _state.update { it.copy(speechModelId = value, error = null) }
    }

    fun setComposerText(value: String) {
        _state.update { it.copy(composerText = value) }
    }

    fun setThreadCount(value: Int) {
        if (value !in 1..8 || _state.value.loadingModel || _state.value.loadingSpeech || _state.value.generating ||
            _state.value.recording || _state.value.transcribing)
            return
        val previous = _state.value.threadCount
        preferences.edit { putInt(THREADS_KEY, value) }
        _state.update { it.copy(threadCount = value) }
        if (_state.value.modelReady || _state.value.speechReady) {
            viewModelScope.launch(runtimeDispatcher) {
                runCatching { checked(NativeRuntime.configure(value)) }.onFailure { error ->
                    preferences.edit { putInt(THREADS_KEY, previous) }
                    _state.update { it.copy(threadCount = previous, error = error.userMessage()) }
                }
            }
        }
    }

    fun setMaximumTokens(value: Int) {
        if (value !in 1..8192) return
        preferences.edit { putInt(TOKENS_KEY, value) }
        _state.update { it.copy(maximumTokens = value) }
    }

    fun installModel() {
        if (_state.value.loadingModel || _state.value.loadingSpeech || _state.value.generating ||
            _state.value.recording || _state.value.transcribing)
            return
        val modelId = _state.value.modelId.trim()
        preferences.edit { putString(MODEL_ID_KEY, modelId) }
        modelJob = viewModelScope.launch {
            _state.update {
                it.copy(loadingModel = true, modelReady = false, progress = 0f, status = "Resolving model", error = null)
            }
            try {
                val model = repository.prepare(modelId) { update ->
                    val ratio = if (update.totalBytes > 0) update.completedBytes.toFloat() / update.totalBytes else 0f
                    _state.update {
                        it.copy(
                            progressFile = update.file,
                            progress = ratio.coerceIn(0f, 1f),
                            status = "Downloading ${(ratio * 100).toInt()}%",
                        )
                    }
                }
                load(model)
            } catch (error: CancellationException) {
                _state.update { it.copy(loadingModel = false, status = "Model offline", progressFile = "") }
                throw error
            } catch (error: Throwable) {
                _state.update {
                    it.copy(loadingModel = false, status = "Model offline", error = error.userMessage())
                }
            } finally {
                modelJob = null
            }
        }
    }

    fun cancelModelInstall() {
        modelJob?.cancel()
    }

    fun installSpeechModel() {
        if (_state.value.loadingModel || _state.value.loadingSpeech || _state.value.generating ||
            _state.value.recording || _state.value.transcribing)
            return
        val modelId = _state.value.speechModelId.trim()
        preferences.edit { putString(SPEECH_MODEL_ID_KEY, modelId) }
        speechJob = viewModelScope.launch {
            _state.update {
                it.copy(
                    loadingSpeech = true,
                    speechReady = false,
                    speechProgress = 0f,
                    status = "Resolving speech model",
                    error = null,
                )
            }
            try {
                val model = repository.prepareSpeech(modelId) { update ->
                    val ratio = if (update.totalBytes > 0) update.completedBytes.toFloat() / update.totalBytes else 0f
                    _state.update {
                        it.copy(
                            speechProgressFile = update.file,
                            speechProgress = ratio.coerceIn(0f, 1f),
                            status = "Downloading speech ${(ratio * 100).toInt()}%",
                        )
                    }
                }
                loadSpeech(model)
            } catch (error: CancellationException) {
                _state.update { it.copy(loadingSpeech = false, status = readyStatus(it), speechProgressFile = "") }
                throw error
            } catch (error: Throwable) {
                _state.update {
                    it.copy(loadingSpeech = false, status = readyStatus(it), error = error.userMessage())
                }
            } finally {
                speechJob = null
            }
        }
    }

    fun cancelSpeechInstall() {
        speechJob?.cancel()
    }

    fun send(content: String) {
        val prompt = content.trim()
        val current = _state.value
        if (prompt.isEmpty() || !current.modelReady || current.generating || current.loadingModel) return
        val messages = current.messages + ChatMessage(MessageRole.USER, prompt)
        _state.update {
            it.copy(
                messages = messages,
                draft = "",
                composerText = "",
                generating = true,
                stopping = false,
                status = "Preparing prompt",
                error = null,
            )
        }
        persistMessages(messages)
        stopRequested.set(false)
        viewModelScope.launch(runtimeDispatcher) {
            var completed = false
            try {
                val enqueue = checked(NativeRuntime.enqueue(messages.toNativeJson().toString(), _state.value.maximumTokens))
                requestId = enqueue.getLong("request_id")
                while (true) {
                    val step = checked(NativeRuntime.step())
                    val events = step.getJSONArray("events")
                    for (index in 0 until events.length()) {
                        val event = events.getJSONObject(index)
                        val text = event.optString("text")
                        if (text.isNotEmpty()) {
                            _state.update { it.copy(draft = it.draft + text, status = "Generating") }
                        }
                        if (event.has("completed")) {
                            finish(event.getJSONObject("completed"))
                            completed = true
                        }
                    }
                    if (stopRequested.getAndSet(false) && step.getInt("pending") > 0) {
                        checked(NativeRuntime.cancel(requireNotNull(requestId)))
                        finishPartial()
                        completed = true
                        break
                    }
                    if (step.getInt("pending") == 0) break
                    yield()
                }
                if (!completed) finishPartial()
            } catch (error: Throwable) {
                failGeneration(error)
            } finally {
                requestId = null
                stopRequested.set(false)
            }
        }
    }

    fun stop() {
        if (!_state.value.generating || _state.value.stopping) return
        stopRequested.set(true)
        _state.update { it.copy(stopping = true, status = "Stopping after current step") }
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun startRecording() {
        val current = _state.value
        if (!current.speechReady) {
            _state.update { it.copy(error = "Load a Whisper speech model in settings first") }
            return
        }
        if (current.generating || current.loadingModel || current.loadingSpeech || current.recording ||
            current.transcribing)
            return
        speechRecorder.prepare()
        _state.update {
            it.copy(recording = true, recordingSeconds = 0f, status = "Listening", error = null)
        }
        recordingJob = viewModelScope.launch {
            val composerPrefix = current.composerText.trim()
            val draftBusy = AtomicBoolean()
            val draftEnabled = AtomicBoolean(true)
            var lastDraftSamples = 0
            val mergeTranscript = { transcript: String ->
                listOf(composerPrefix, transcript.trim()).filter(String::isNotEmpty).joinToString(" ")
            }
            fun publishPartial(payload: String, draft: Boolean) {
                val partial = checked(payload)
                _state.update { state ->
                    if ((draft && state.recording) || (!draft && state.transcribing)) {
                        state.copy(
                            composerText = mergeTranscript(partial.optString("text")),
                            status = if (draft) "Draft transcript · ${partial.optString("language")}" else "Refining transcript",
                        )
                    } else state
                }
            }
            try {
                var reportedTenths = -1
                val samples = speechRecorder.capture(
                    onSamples = { count ->
                        val tenths = count / 1600
                        if (tenths != reportedTenths) {
                            reportedTenths = tenths
                            _state.update { it.copy(recordingSeconds = count / 16000f) }
                        }
                        val due = count >= MINIMUM_DRAFT_SAMPLES &&
                            (lastDraftSamples == 0 || count - lastDraftSamples >= DRAFT_INTERVAL_SAMPLES)
                        val requested = due && draftEnabled.get() && draftBusy.compareAndSet(false, true)
                        if (requested) lastDraftSamples = count
                        requested
                    },
                    onSnapshot = { snapshot ->
                        viewModelScope.launch(runtimeDispatcher) {
                            try {
                                val draft = checked(NativeRuntime.transcribe(snapshot, "auto", 128) { payload ->
                                    publishPartial(payload, true)
                                })
                                if (_state.value.recording) {
                                    _state.update {
                                        it.copy(
                                            composerText = mergeTranscript(draft.optString("text")),
                                            status = "Draft transcript · ${draft.optString("language")}",
                                        )
                                    }
                                }
                            } catch (error: CancellationException) {
                                throw error
                            } catch (error: Throwable) {
                                draftEnabled.set(false)
                                _state.update {
                                    it.copy(error = "Live transcript failed: ${error.userMessage()}")
                                }
                            } finally {
                                draftBusy.set(false)
                            }
                        }
                    },
                )
                _state.update { it.copy(recording = false, transcribing = true, status = "Refining transcript") }
                val result = withContext(runtimeDispatcher) {
                    checked(NativeRuntime.transcribe(samples, "auto", 128) { payload ->
                        publishPartial(payload, false)
                    })
                }
                val transcript = result.optString("text").trim()
                _state.update {
                    it.copy(
                        composerText = mergeTranscript(transcript),
                        recording = false,
                        transcribing = false,
                        recordingSeconds = 0f,
                        status = readyStatus(it),
                    )
                }
            } catch (error: CancellationException) {
                _state.update {
                    it.copy(recording = false, transcribing = false, recordingSeconds = 0f, status = readyStatus(it))
                }
                throw error
            } catch (error: Throwable) {
                _state.update {
                    it.copy(
                        recording = false,
                        transcribing = false,
                        recordingSeconds = 0f,
                        status = readyStatus(it),
                        error = error.userMessage(),
                    )
                }
            } finally {
                recordingJob = null
            }
        }
    }

    fun stopRecording() {
        if (!_state.value.recording) return
        _state.update { it.copy(status = "Finishing recording") }
        speechRecorder.stop()
    }

    fun newChat() {
        if (_state.value.generating || _state.value.loadingModel ||
            _state.value.recording || _state.value.transcribing)
            return
        persistMessages(emptyList())
        _state.update { it.copy(messages = emptyList(), draft = "", status = readyStatus(it), error = null) }
    }

    fun removeModel() {
        if (_state.value.generating || _state.value.loadingModel || _state.value.loadingSpeech ||
            _state.value.recording || _state.value.transcribing)
            return
        viewModelScope.launch {
            withContext(runtimeDispatcher) { NativeRuntime.unload() }
            repository.removeInstalled()
            _state.update {
                it.copy(modelReady = false, modelRevision = null, status = "Model offline", progress = 0f, error = null)
            }
        }
    }

    fun removeSpeechModel() {
        if (_state.value.generating || _state.value.loadingModel || _state.value.loadingSpeech ||
            _state.value.recording || _state.value.transcribing)
            return
        viewModelScope.launch {
            withContext(runtimeDispatcher) { NativeRuntime.unloadAsr() }
            repository.removeInstalledSpeech()
            _state.update {
                it.copy(
                    speechReady = false,
                    speechModelRevision = null,
                    speechProgress = 0f,
                    status = readyStatus(it.copy(speechReady = false)),
                    error = null,
                )
            }
        }
    }

    fun clearError() {
        _state.update { it.copy(error = null) }
    }

    fun reportError(message: String) {
        _state.update { it.copy(error = message) }
    }

    private fun load(model: InstalledModel) {
        viewModelScope.launch(runtimeDispatcher) {
            _state.update { it.copy(loadingModel = true, status = "Loading model", error = null) }
            runCatching {
                checked(NativeRuntime.configure(_state.value.threadCount))
                checked(NativeRuntime.load(model.directory.absolutePath))
            }.onSuccess {
                _state.update {
                    it.copy(
                        modelId = model.modelId,
                        modelRevision = model.revision,
                        modelReady = true,
                        loadingModel = false,
                        progress = 1f,
                        status = "Ready on device",
                    )
                }
            }.onFailure { error ->
                _state.update {
                    it.copy(modelReady = false, loadingModel = false, status = "Model offline", error = error.userMessage())
                }
            }
        }
    }

    private fun loadSpeech(model: InstalledModel) {
        viewModelScope.launch(runtimeDispatcher) {
            val int8 = model.modelId == DEFAULT_SPEECH_MODEL_ID
            _state.update { it.copy(loadingSpeech = true, status = if (int8) "Preparing Whisper Small INT8" else "Loading speech model", error = null) }
            runCatching {
                checked(NativeRuntime.configure(_state.value.threadCount))
                checked(NativeRuntime.loadAsr(model.directory.absolutePath, int8))
            }.onSuccess {
                _state.update {
                    it.copy(
                        speechModelId = model.modelId,
                        speechModelRevision = model.revision,
                        speechReady = true,
                        loadingSpeech = false,
                        speechProgress = 1f,
                        status = if (it.generating) it.status else readyStatus(it.copy(speechReady = true)),
                    )
                }
            }.onFailure { error ->
                _state.update {
                    it.copy(
                        speechReady = false,
                        loadingSpeech = false,
                        status = if (it.generating) it.status else readyStatus(it),
                        error = error.userMessage(),
                    )
                }
            }
        }
    }

    private fun finish(result: JSONObject) {
        val stats = MessageStats(
            tokens = result.optInt("output_tokens"),
            elapsedMs = result.optDouble("generation_ms"),
            decodeMs = result.optDouble("decode_ms"),
            decodeTokens = result.optInt("decode_tokens"),
        )
        val finalText = result.optString("text").ifBlank { _state.value.draft }
        val messages = _state.value.messages + ChatMessage(MessageRole.ASSISTANT, finalText, stats)
        persistMessages(messages)
        _state.update {
            it.copy(messages = messages, draft = "", generating = false, stopping = false, status = readyStatus(it))
        }
    }

    private fun finishPartial() {
        val current = _state.value
        val messages = if (current.draft.isBlank()) {
            current.messages.dropLastWhile { it.role == MessageRole.USER }.takeIf { it.size < current.messages.size }
                ?: current.messages
        } else {
            current.messages + ChatMessage(MessageRole.ASSISTANT, current.draft)
        }
        persistMessages(messages)
        _state.update {
            it.copy(messages = messages, draft = "", generating = false, stopping = false, status = readyStatus(it))
        }
    }

    private fun failGeneration(error: Throwable) {
        val current = _state.value
        val messages = if (current.draft.isBlank()) current.messages.dropLast(1) else {
            current.messages + ChatMessage(MessageRole.ASSISTANT, current.draft)
        }
        persistMessages(messages)
        _state.update {
            it.copy(
                messages = messages,
                draft = "",
                generating = false,
                stopping = false,
                status = readyStatus(it),
                error = error.userMessage(),
            )
        }
    }

    private fun readMessages(): List<ChatMessage> = runCatching {
        val source = JSONArray(preferences.getString(MESSAGES_KEY, "[]"))
        buildList {
            for (index in 0 until source.length()) {
                val item = source.getJSONObject(index)
                val role = MessageRole.valueOf(item.getString("role"))
                add(ChatMessage(role, item.getString("content")))
            }
        }.takeLast(MAXIMUM_SAVED_MESSAGES)
    }.getOrDefault(emptyList())

    private fun persistMessages(messages: List<ChatMessage>) {
        val value = JSONArray()
        messages.takeLast(MAXIMUM_SAVED_MESSAGES).forEach { message ->
            value.put(JSONObject().put("role", message.role.name).put("content", message.content))
        }
        preferences.edit { putString(MESSAGES_KEY, value.toString()) }
    }

    private fun List<ChatMessage>.toNativeJson() = JSONArray().also { result ->
        forEach { message ->
            result.put(
                JSONObject()
                    .put("role", if (message.role == MessageRole.USER) "user" else "assistant")
                    .put("content", message.content),
            )
        }
    }

    private fun checked(source: String): JSONObject {
        val result = JSONObject(source)
        if (result.has("error")) throw IllegalStateException(result.getString("error"))
        return result
    }

    private fun readyStatus(state: KidiUiState) = when {
        state.modelReady -> "Ready on device"
        state.speechReady -> "Speech ready"
        else -> "Model offline"
    }

    override fun onCleared() {
        stopRequested.set(true)
        speechRecorder.stop()
        recordingJob?.cancel()
        val id = requestId
        runtimeExecutor.execute {
            if (id != null) runCatching { NativeRuntime.cancel(id) }
            NativeRuntime.unloadAsr()
            NativeRuntime.unload()
        }
        runtimeExecutor.shutdown()
        super.onCleared()
    }

    private fun Throwable.userMessage() = message?.substringAfterLast("Exception: ")?.ifBlank { null }
        ?: "Unexpected application error"

    private companion object {
        const val PREFERENCES = "kidi-android-v1"
        const val MODEL_ID_KEY = "model-id"
        const val SPEECH_MODEL_ID_KEY = "speech-model-id"
        const val THREADS_KEY = "threads"
        const val TOKENS_KEY = "tokens"
        const val MESSAGES_KEY = "messages"
        const val MAXIMUM_SAVED_MESSAGES = 100
        const val MINIMUM_DRAFT_SAMPLES = 12800
        const val DRAFT_INTERVAL_SAMPLES = 19200
    }
}

private fun defaultThreadCount() = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)