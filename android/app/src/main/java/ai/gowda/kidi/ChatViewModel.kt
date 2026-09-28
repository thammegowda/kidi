package ai.gowda.kidi

import android.Manifest
import android.app.Application
import android.os.SystemClock
import android.util.Log
import androidx.annotation.RequiresPermission
import androidx.core.content.edit
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
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

internal data class KidiUiState(
    val modelId: String = DEFAULT_MODEL_ID,
    val modelRevision: String? = null,
    val messages: List<ChatMessage> = emptyList(),
    val activeChatId: String? = null,
    val loadingChat: Boolean = false,
    val hasOlderMessages: Boolean = false,
    val loadingOlderMessages: Boolean = false,
    val history: List<ChatSummary> = emptyList(),
    val historyQuery: String = "",
    val historyLoading: Boolean = false,
    val historyHasMore: Boolean = false,
    val historyError: String? = null,
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
    private val chats = ChatRepository(application)
    private val speechRecorder = SpeechRecorder()
    private val runtimeExecutor = Executors.newSingleThreadExecutor { runnable -> Thread(runnable, "kidi-runtime") }
    private val runtimeDispatcher: CoroutineDispatcher = runtimeExecutor.asCoroutineDispatcher()
    private val _state = MutableStateFlow(
        KidiUiState(
            modelId = preferences.getString(MODEL_ID_KEY, DEFAULT_MODEL_ID) ?: DEFAULT_MODEL_ID,
            speechModelId = preferences.getString(SPEECH_MODEL_ID_KEY, DEFAULT_SPEECH_MODEL_ID)
                ?: DEFAULT_SPEECH_MODEL_ID,
            loadingChat = true,
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
    private var historyJob: Job? = null
    private var responseAgent = ChatParticipant.LEGACY_AGENT

    init {
        viewModelScope.launch {
            val started = SystemClock.elapsedRealtime()
            try {
                val restored = withContext(Dispatchers.IO) {
                    chats.restore(preferences.getString(MESSAGES_KEY, null))
                }
                _state.update { it.copy(activeChatId = restored?.id, messages = restored?.page?.messages.orEmpty(),
                    hasOlderMessages = restored?.page?.hasMore ?: false, loadingChat = false) }
                Log.i("KidiStartup", "history_ready ms=${SystemClock.elapsedRealtime() - started}")
                preferences.edit { remove(MESSAGES_KEY) }
                refreshHistory()
            } catch (error: Exception) {
                _state.update { it.copy(loadingChat = false, error = "Unable to restore chats: ${error.userMessage()}") }
            }
        }
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
        if (prompt.isEmpty() || !current.modelReady || current.generating || current.loadingModel || current.loadingChat ||
            current.recording || current.transcribing) return
        val pending = ChatMessage(MessageRole.USER, prompt)
        val messages = current.messages + pending
        responseAgent = ChatParticipant.agent(current.modelId)
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
        stopRequested.set(false)
        viewModelScope.launch(runtimeDispatcher) {
            var completed = false
            try {
                val (threadId, saved) = withContext(Dispatchers.IO) {
                    val id = current.activeChatId ?: chats.createThread(listOf(ChatParticipant.USER, responseAgent))
                    chats.addParticipant(id, responseAgent)
                    val message = chats.append(id, pending)
                    chats.select(id)
                    id to message
                }
                _state.update { it.copy(activeChatId = threadId, messages = it.messages.map { message ->
                    if (message.id == saved.id) saved else message
                }) }
                refreshHistory()
                val enqueue = checked(NativeRuntime.enqueue(messages.textGenerationContext()
                    .toNativeJson().toString(), _state.value.maximumTokens))
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
        if (_state.value.generating || _state.value.loadingChat ||
            _state.value.recording || _state.value.transcribing)
            return
        _state.update { it.copy(loadingChat = true) }
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { chats.select(null) }
                _state.update { it.copy(activeChatId = null, messages = emptyList(), draft = "", composerText = "",
                    hasOlderMessages = false, loadingOlderMessages = false, loadingChat = false,
                    status = readyStatus(it), error = null) }
            } catch (error: Exception) {
                _state.update { it.copy(loadingChat = false, error = error.userMessage()) }
            }
        }
    }

    fun openChat(id: String) {
        val current = _state.value
        if (current.generating || current.recording || current.transcribing || current.loadingChat || id == current.activeChatId) return
        _state.update { it.copy(loadingChat = true) }
        viewModelScope.launch {
            try {
                val saved = withContext(Dispatchers.IO) {
                    requireNotNull(chats.load(id)) { "Chat no longer exists" }.also { chats.select(id) }
                }
                _state.update { it.copy(activeChatId = id, messages = saved.page.messages, hasOlderMessages = saved.page.hasMore,
                    draft = "", composerText = "", loadingChat = false, loadingOlderMessages = false, error = null) }
            } catch (error: Exception) {
                _state.update { it.copy(loadingChat = false, error = error.userMessage()) }
            }
        }
    }

    fun loadOlderMessages() {
        val current = _state.value
        val id = current.activeChatId ?: return
        if (!current.hasOlderMessages || current.loadingOlderMessages || current.loadingChat || current.generating) return
        val before = current.messages.firstOrNull()?.sequence ?: return
        _state.update { it.copy(loadingOlderMessages = true) }
        viewModelScope.launch {
            try {
                val page = withContext(Dispatchers.IO) { chats.messages(id, before) }
                _state.update {
                    if (it.activeChatId == id) it.copy(messages = (page.messages + it.messages).distinctBy { message -> message.id },
                        hasOlderMessages = page.hasMore, loadingOlderMessages = false) else it
                }
            } catch (error: Exception) {
                _state.update { if (it.activeChatId == id) it.copy(loadingOlderMessages = false, error = error.userMessage()) else it }
            }
        }
    }

    fun searchHistory(query: String) {
        _state.update { it.copy(historyQuery = query) }
        requestHistory(false)
    }

    fun refreshHistory() {
        viewModelScope.launch { requestHistory(false) }
    }

    fun loadMoreHistory() = requestHistory(true)

    private fun requestHistory(more: Boolean) {
        val current = _state.value
        if (more && (current.historyLoading || !current.historyHasMore)) return
        historyJob?.cancel()
        val query = current.historyQuery
        val after = if (more) current.history.lastOrNull() else null
        _state.update { it.copy(historyLoading = true, historyError = null,
            history = if (more) it.history else emptyList(), historyHasMore = if (more) it.historyHasMore else false) }
        historyJob = viewModelScope.launch {
            try {
                if (!more && query.isNotBlank()) delay(200)
                val page = withContext(Dispatchers.IO) { chats.recent(query, after) }
                _state.update { it.copy(history = if (more) it.history + page.chats else page.chats,
                    historyHasMore = page.hasMore, historyLoading = false) }
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                _state.update { it.copy(historyLoading = false, historyError = error.userMessage()) }
            }
        }
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
        val queued = SystemClock.elapsedRealtime()
        viewModelScope.launch(runtimeDispatcher) {
            val started = SystemClock.elapsedRealtime()
            Log.i("KidiStartup", "gemma_start queue_ms=${started - queued}")
            _state.update { it.copy(loadingModel = true, status = "Loading model", error = null) }
            runCatching {
                checked(NativeRuntime.configure(_state.value.threadCount))
                val configured = SystemClock.elapsedRealtime()
                checked(NativeRuntime.load(model.directory.absolutePath)).also {
                    Log.i("KidiStartup", "gemma_ready configure_ms=${configured - started} native_ms=${it.optDouble("load_ms")} stages=${it.optJSONObject("stages_ms")}")
                }
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
                Log.e("KidiStartup", "gemma_failed elapsed_ms=${SystemClock.elapsedRealtime() - started}")
                _state.update {
                    it.copy(modelReady = false, loadingModel = false, status = "Model offline", error = error.userMessage())
                }
            }
        }
    }

    private fun loadSpeech(model: InstalledModel) {
        val queued = SystemClock.elapsedRealtime()
        viewModelScope.launch(runtimeDispatcher) {
            val started = SystemClock.elapsedRealtime()
            Log.i("KidiStartup", "whisper_start queue_ms=${started - queued}")
            val int8 = model.modelId == DEFAULT_SPEECH_MODEL_ID
            _state.update { it.copy(loadingSpeech = true, status = if (int8) "Preparing Whisper Small INT8" else "Loading speech model", error = null) }
            runCatching {
                checked(NativeRuntime.configure(_state.value.threadCount))
                val configured = SystemClock.elapsedRealtime()
                checked(NativeRuntime.loadAsr(model.directory.absolutePath, int8)).also {
                    Log.i("KidiStartup", "whisper_ready configure_ms=${configured - started} native_ms=${it.optDouble("load_ms")} stages=${it.optJSONObject("stages_ms")}")
                }
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
                Log.e("KidiStartup", "whisper_failed elapsed_ms=${SystemClock.elapsedRealtime() - started}")
            }
        }
    }

    private suspend fun finish(result: JSONObject) {
        val stats = MessageStats(
            tokens = result.optInt("output_tokens"),
            elapsedMs = result.optDouble("generation_ms"),
            decodeMs = result.optDouble("decode_ms"),
            decodeTokens = result.optInt("decode_tokens"),
        )
        val finalText = result.optString("text").ifBlank { _state.value.draft }
        val message = saveReply(ChatMessage(MessageRole.ASSISTANT, finalText, stats, sender = responseAgent))
        val messages = _state.value.messages + message
        _state.update {
            it.copy(messages = messages, draft = "", generating = false, stopping = false, status = readyStatus(it))
        }
    }

    private suspend fun finishPartial() {
        val current = _state.value
        val message = saveReply(ChatMessage(MessageRole.ASSISTANT, current.draft, sender = responseAgent, status = MessageStatus.STOPPED))
        val messages = current.messages + message
        _state.update {
            it.copy(messages = messages, draft = "", generating = false, stopping = false, status = readyStatus(it))
        }
    }

    private suspend fun failGeneration(error: Throwable) {
        val current = _state.value
        val failed = ChatMessage(MessageRole.ASSISTANT, current.draft, sender = responseAgent, status = MessageStatus.FAILED)
        val message = runCatching { saveReply(failed) }.getOrDefault(failed)
        val messages = current.messages + message
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

    private suspend fun saveReply(message: ChatMessage): ChatMessage {
        val id = requireNotNull(_state.value.activeChatId)
        val saved = withContext(Dispatchers.IO) { chats.append(id, message) }
        refreshHistory()
        return saved
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
            chats.close()
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
        const val MINIMUM_DRAFT_SAMPLES = 12800
        const val DRAFT_INTERVAL_SAMPLES = 19200
    }
}

private fun defaultThreadCount() = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)