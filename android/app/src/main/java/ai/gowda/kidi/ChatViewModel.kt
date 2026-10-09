package ai.gowda.kidi

import android.Manifest
import android.app.Application
import android.content.SharedPreferences
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.RequiresPermission
import androidx.core.content.edit
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
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
import java.util.concurrent.atomic.AtomicReference

/** A completed draft transcription of the first [samples] recorded samples. */
private class SpeechDraft(val samples: Int, val result: JSONObject)

/** A draft transcription of the first [samples] samples that may still be running; null when cancelled or failed. */
private class PendingDraft(val samples: Int, val job: Deferred<JSONObject?>)

private const val DEFAULT_MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
private const val DEFAULT_SPEECH_MODEL_ID = "openai/whisper-small"

internal data class HardwareDiagnostic(
    val name: String,
    val backend: String,
    val recognized: Boolean,
    val available: Boolean,
    val detail: String,
)

private data class HardwareDiagnostics(
    val cpu: HardwareDiagnostic,
    val gpu: HardwareDiagnostic,
    val npu: HardwareDiagnostic,
)

private fun JSONObject.hardwareDiagnostic(key: String): HardwareDiagnostic {
    val device = getJSONObject(key)
    val detail = device.optString("detail").ifBlank { device.optString("reason") }
    return HardwareDiagnostic(
        name = device.optString("name", "Unknown"),
        backend = device.optString("backend"),
        recognized = device.optBoolean("recognized"),
        available = device.optBoolean("available"),
        detail = detail,
    )
}

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
    val provisionalTextStart: Int? = null,
    val pendingImages: List<MessageAttachment> = emptyList(),
    val importingImage: Boolean = false,
    val photoSource: CaptureSource = CaptureSource.PHONE,
    val speechSource: CaptureSource = CaptureSource.PHONE,
    val pairedAccessoryName: String? = null,
    val pairingInvitation: PairingInvitation? = null,
    val pairing: Boolean = false,
    val accessoryBusy: Boolean = false,
    val visionReady: Boolean = false,
    val modelReady: Boolean = false,
    val loadingModel: Boolean = false,
    val generating: Boolean = false,
    val generationStartedAtMs: Long = 0,
    val generationTokens: Int = 0,
    val generationDecodeTokens: Int = 0,
    val generationDecodeMs: Double = 0.0,
    val stopping: Boolean = false,
    val progressFile: String = "",
    val progress: Float = 0f,
    val progressPhase: ModelDownloadPhase = ModelDownloadPhase.CONNECTING,
    val modelError: String? = null,
    val status: String = "Model offline",
    val error: String? = null,
    val threadCount: Int = defaultThreadCount(),
    val chatAccelerator: String = DEFAULT_ACCELERATOR,
    val speechAccelerator: String = DEFAULT_ACCELERATOR,
    val chatExecution: String = "",
    val speechExecution: String = "",
    val cpuDiagnostic: HardwareDiagnostic? = null,
    val gpuDiagnostic: HardwareDiagnostic? = null,
    val npuDiagnostic: HardwareDiagnostic? = null,
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
    val speechProgressPhase: ModelDownloadPhase = ModelDownloadPhase.CONNECTING,
    val speechModelError: String? = null,
)

internal fun KidiUiState.withProvisionalTranscript(text: String, start: Int, complete: Boolean = false): KidiUiState {
    if (text == composerText || (!complete && text.length < composerText.length)) return this
    return copy(composerText = text, provisionalTextStart = start.takeIf { it < text.length })
}

internal class ReplyUpdateBuffer(private val clockMs: () -> Long, private val publish: (String) -> Unit) {
    private val content = StringBuilder()
    private var lastPublishedAt = clockMs()
    private var pending = false

    fun append(text: String) {
        content.append(text)
        pending = true
    }

    fun flush(force: Boolean = false) {
        if (!pending) return
        val now = clockMs()
        if (!force && now - lastPublishedAt < 100L) return
        publish(content.toString())
        pending = false
        lastPublishedAt = now
    }
}

internal class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private val preferences = application.getSharedPreferences(PREFERENCES, 0)
    private val repository = ModelRepository(application)
    private val chats = ChatRepository(application)
    private val speechRecorder = SpeechRecorder()
    private val imageStore = ImageStore(application)
    private val accessoryProfiles = AccessoryProfileStore(application)
    private val accessoryNetwork = AccessoryNetwork(application)
    private val accessoryPairer = AccessoryBlePairer(application, accessoryProfiles)
    private val runtimeExecutor = Executors.newSingleThreadExecutor { runnable -> Thread(runnable, "kidi-runtime") }
    private val runtimeDispatcher: CoroutineDispatcher = runtimeExecutor.asCoroutineDispatcher()
    // Speech has its own thread, like the web app's speech worker, so dictation never queues behind chat work.
    private val speechExecutor = Executors.newSingleThreadExecutor { runnable -> Thread(runnable, "kidi-speech") }
    private val speechDispatcher: CoroutineDispatcher = speechExecutor.asCoroutineDispatcher()
    @Volatile
    private var stopRequestedAtMs = 0L
    private val _state = MutableStateFlow(
        KidiUiState(
            modelId = preferences.getString(MODEL_ID_KEY, DEFAULT_MODEL_ID) ?: DEFAULT_MODEL_ID,
            speechModelId = preferences.getString(SPEECH_MODEL_ID_KEY, DEFAULT_SPEECH_MODEL_ID)
                ?: DEFAULT_SPEECH_MODEL_ID,
            loadingChat = true,
            threadCount = preferences.getInt(THREADS_KEY, defaultThreadCount()).coerceIn(1, 8),
            maximumTokens = preferences.getInt(TOKENS_KEY, 1024).coerceIn(1, 8192),
            chatAccelerator = preferences.getString(CHAT_ACCELERATOR_KEY, null)
                ?.takeIf { it in ACCELERATOR_LABELS } ?: DEFAULT_ACCELERATOR,
            speechAccelerator = preferences.getString(SPEECH_ACCELERATOR_KEY, null)
                ?.takeIf { it in ACCELERATOR_LABELS } ?: DEFAULT_ACCELERATOR,
            photoSource = preferences.captureSource(PHOTO_SOURCE_KEY),
            speechSource = preferences.captureSource(SPEECH_SOURCE_KEY),
            pairedAccessoryName = runCatching { accessoryProfiles.profiles().firstOrNull()?.name }.getOrNull(),
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
        // Runs first on the single runtime thread, so NPU libraries are in place before any model loads.
        viewModelScope.launch(runtimeDispatcher) {
            runCatching {
                checked(NativeRuntime.setDataDirectory(
                    application.filesDir.absolutePath, application.applicationInfo.nativeLibraryDir))
                val devices = checked(NativeRuntime.deviceInfo())
                Log.i("KidiDiagnostics", devices.toString())
                HardwareDiagnostics(
                    cpu = devices.hardwareDiagnostic("cpu"),
                    gpu = devices.hardwareDiagnostic("gpu"),
                    npu = devices.hardwareDiagnostic("npu"),
                )
            }.onSuccess { devices ->
                _state.update {
                    it.copy(
                        cpuDiagnostic = devices.cpu,
                        gpuDiagnostic = devices.gpu,
                        npuDiagnostic = devices.npu,
                    )
                }
            }.onFailure { error -> Log.e("KidiStartup", "native_storage_failed ${error.userMessage()}") }
        }
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
        _state.update { it.copy(modelId = value, error = null, modelError = null) }
    }

    fun setSpeechModelId(value: String) {
        _state.update { it.copy(speechModelId = value, error = null, speechModelError = null) }
    }

    fun setComposerText(value: String) {
        _state.update { it.copy(composerText = value, provisionalTextStart = null) }
    }

    fun captureImageUri(): Uri = imageStore.captureUri()

    fun finishCapture(uri: Uri, captured: Boolean) {
        if (captured) attachImage(uri) else imageStore.removeCapture(uri)
    }

    fun setPhotoSource(source: CaptureSource) {
        if (source == CaptureSource.ACCESSORY && _state.value.pairedAccessoryName == null) {
            _state.update { it.copy(error = "Pair an accessory before selecting its camera") }
            return
        }
        preferences.edit { putString(PHOTO_SOURCE_KEY, source.name) }
        _state.update { it.copy(photoSource = source) }
    }

    fun setSpeechSource(source: CaptureSource) {
        if (source == CaptureSource.ACCESSORY && _state.value.pairedAccessoryName == null) {
            _state.update { it.copy(error = "Pair an accessory before selecting its microphone") }
            return
        }
        preferences.edit { putString(SPEECH_SOURCE_KEY, source.name) }
        _state.update { it.copy(speechSource = source) }
    }

    fun acceptPairingUri(value: String) {
        if (_state.value.pairing || _state.value.accessoryBusy) return
        runCatching {
            AccessoryProtocol.parsePairingUri(value, System.currentTimeMillis() / 1000)
        }.onSuccess { invitation ->
            _state.update { it.copy(pairingInvitation = invitation, error = null) }
        }.onFailure { error ->
            _state.update { it.copy(error = "Invalid accessory invitation: ${error.userMessage()}") }
        }
    }

    fun cancelPairing() {
        if (!_state.value.pairing) _state.update { it.copy(pairingInvitation = null) }
    }

    @RequiresPermission(allOf = [Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT])
    fun pairPendingAccessory(controllerName: String) {
        val invitation = _state.value.pairingInvitation ?: return
        if (_state.value.pairing || _state.value.chatBusyState) return
        _state.update { it.copy(pairing = true, status = "Pairing ${invitation.deviceName}", error = null) }
        viewModelScope.launch {
            try {
                val profile = accessoryPairer.pair(invitation, controllerName)
                _state.update {
                    it.copy(
                        pairing = false,
                        pairingInvitation = null,
                        pairedAccessoryName = profile.name,
                        status = readyStatus(it),
                    )
                }
            } catch (error: Exception) {
                _state.update {
                    it.copy(
                        pairing = false,
                        status = readyStatus(it),
                        error = "Accessory pairing failed: ${error.userMessage()}",
                    )
                }
            }
        }
    }

    @RequiresPermission(allOf = [Manifest.permission.CHANGE_WIFI_STATE, Manifest.permission.ACCESS_WIFI_STATE])
    fun unpairAccessory() {
        val current = _state.value
        if (current.chatBusyState || current.accessoryBusy || current.pairing) return
        val profile = runCatching { accessoryProfiles.profiles().firstOrNull() }.getOrNull()
        if (profile == null) {
            _state.update { it.copy(error = "No accessory is paired") }
            return
        }
        _state.update { it.copy(accessoryBusy = true, status = "Unpairing ${profile.name}", error = null) }
        viewModelScope.launch {
            try {
                accessoryNetwork.connect(profile).use { lease ->
                    AccessoryClient(profile, lease.network.socketFactory).unpair()
                }
                accessoryProfiles.remove(profile.deviceId)
                preferences.edit {
                    putString(PHOTO_SOURCE_KEY, CaptureSource.PHONE.name)
                    putString(SPEECH_SOURCE_KEY, CaptureSource.PHONE.name)
                }
                _state.update {
                    it.copy(
                        pairedAccessoryName = null,
                        photoSource = CaptureSource.PHONE,
                        speechSource = CaptureSource.PHONE,
                        accessoryBusy = false,
                        status = readyStatus(it),
                    )
                }
            } catch (error: Exception) {
                _state.update {
                    it.copy(
                        accessoryBusy = false,
                        status = readyStatus(it),
                        error = "Accessory unpair failed: ${error.userMessage()}",
                    )
                }
            }
        }
    }

    @RequiresPermission(allOf = [Manifest.permission.CHANGE_WIFI_STATE, Manifest.permission.ACCESS_WIFI_STATE])
    fun captureAccessoryPhoto() {
        val current = _state.value
        if (!current.visionReady) {
            _state.update { it.copy(error = "Load a vision model before attaching a photo") }
            return
        }
        if (current.chatBusyState || current.accessoryBusy) return
        val profile = runCatching { accessoryProfiles.profiles().firstOrNull() }.getOrNull()
        if (profile == null) {
            _state.update { it.copy(error = "Pair an accessory before requesting a photo") }
            return
        }
        _state.update { it.copy(importingImage = true, accessoryBusy = true, status = "Waking accessory camera") }
        viewModelScope.launch {
            try {
                val image = accessoryNetwork.connect(profile).use { lease ->
                    val jpeg = AccessoryClient(profile, lease.network.socketFactory).capturePhoto("chat")
                    withContext(Dispatchers.IO) { imageStore.importJpeg(jpeg, "${profile.name} photo") }
                }
                withContext(Dispatchers.IO) { current.pendingImages.forEach(imageStore::removeDraft) }
                _state.update {
                    it.copy(
                        pendingImages = listOf(image),
                        importingImage = false,
                        accessoryBusy = false,
                        status = readyStatus(it),
                    )
                }
            } catch (error: Exception) {
                _state.update {
                    it.copy(
                        importingImage = false,
                        accessoryBusy = false,
                        status = readyStatus(it),
                        error = "Accessory photo failed: ${error.userMessage()}",
                    )
                }
            }
        }
    }

    fun attachImage(uri: Uri) {
        val current = _state.value
        if (current.generating || current.recording || current.transcribing || current.importingImage) return
        _state.update { it.copy(importingImage = true, error = null) }
        viewModelScope.launch {
            try {
                val image = withContext(Dispatchers.IO) { imageStore.import(uri) }
                withContext(Dispatchers.IO) { current.pendingImages.forEach(imageStore::removeDraft) }
                _state.update { it.copy(pendingImages = listOf(image), importingImage = false) }
            } catch (error: Exception) {
                _state.update { it.copy(importingImage = false, error = "Unable to import photo: ${error.userMessage()}") }
            } finally {
                imageStore.removeCapture(uri)
            }
        }
    }

    fun removePendingImage() {
        if (_state.value.importingImage || _state.value.generating) return
        val pending = _state.value.pendingImages
        _state.update { it.copy(pendingImages = emptyList()) }
        viewModelScope.launch(Dispatchers.IO) { pending.forEach(imageStore::removeDraft) }
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

    private fun busy() = _state.value.let {
        it.loadingModel || it.loadingSpeech || it.generating || it.recording || it.transcribing ||
            it.pairing || it.accessoryBusy
    }

    fun setChatAccelerator(value: String) {
        if (value !in ACCELERATOR_LABELS || value == _state.value.chatAccelerator || busy()) return
        preferences.edit { putString(CHAT_ACCELERATOR_KEY, value) }
        _state.update { it.copy(chatAccelerator = value) }
        repository.installed()?.let(::load)
    }

    fun setSpeechAccelerator(value: String) {
        if (value !in ACCELERATOR_LABELS || value == _state.value.speechAccelerator || busy()) return
        preferences.edit { putString(SPEECH_ACCELERATOR_KEY, value) }
        _state.update { it.copy(speechAccelerator = value) }
        repository.installedSpeech()?.let(::loadSpeech)
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
                it.copy(loadingModel = true, modelReady = false, progress = 0f, progressFile = "",
                    status = "Resolving model", error = null, modelError = null)
            }
            try {
                val model = repository.prepare(modelId) { update ->
                    val ratio = if (update.totalBytes > 0) update.completedBytes.toFloat() / update.totalBytes else 0f
                    _state.update {
                        it.copy(
                            progressFile = update.file,
                            progress = ratio.coerceIn(0f, 1f),
                            progressPhase = update.phase,
                            status = "${update.phase.label} chat ${(ratio * 100).toInt()}%",
                        )
                    }
                }
                load(model)
            } catch (error: CancellationException) {
                _state.update { it.copy(loadingModel = false, status = "Model offline", progressFile = "") }
                throw error
            } catch (error: Throwable) {
                val message = "Chat download failed: ${error.userMessage()}"
                Log.e("KidiDownload", message)
                _state.update {
                    it.copy(loadingModel = false, status = "Model offline", error = message, modelError = message)
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
                    speechProgressFile = "",
                    speechModelError = null,
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
                            speechProgressPhase = update.phase,
                            status = "${update.phase.label} speech ${(ratio * 100).toInt()}%",
                        )
                    }
                }
                loadSpeech(model)
            } catch (error: CancellationException) {
                _state.update { it.copy(loadingSpeech = false, status = readyStatus(it), speechProgressFile = "") }
                throw error
            } catch (error: Throwable) {
                val message = "Speech download failed: ${error.userMessage()}"
                Log.e("KidiDownload", message)
                _state.update {
                    it.copy(loadingSpeech = false, status = readyStatus(it), error = message, speechModelError = message)
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
        val current = _state.value
        val prompt = content.trim().ifEmpty { if (current.pendingImages.isNotEmpty()) "What is in this image?" else "" }
        if (prompt.isEmpty() || !current.modelReady || current.generating || current.loadingModel || current.loadingChat ||
            current.recording || current.transcribing || current.importingImage) return
        if (current.pendingImages.isNotEmpty() && !current.visionReady) {
            reportError("The loaded model does not support images")
            return
        }
        val pending = ChatMessage(MessageRole.USER, prompt, attachments = current.pendingImages)
        val messages = current.messages + pending
        responseAgent = ChatParticipant.agent(current.modelId)
        _state.update {
            it.copy(
                messages = messages,
                draft = "",
                composerText = "",
                provisionalTextStart = null,
                pendingImages = emptyList(),
                generating = true,
                generationStartedAtMs = SystemClock.elapsedRealtime(),
                generationTokens = 0,
                generationDecodeTokens = 0,
                generationDecodeMs = 0.0,
                stopping = false,
                status = if (messages.any { it.attachments.isNotEmpty() }) "Analyzing image" else "Preparing prompt",
                error = null,
            )
        }
        stopRequested.set(false)
        viewModelScope.launch(runtimeDispatcher) {
            var completed = false
            var generatedTokens = 0
            var decodeTokens = 0
            var decodeMs = 0.0
            val reply = ReplyUpdateBuffer(SystemClock::elapsedRealtime) { text ->
                _state.update { it.copy(draft = text, generationTokens = generatedTokens,
                    generationDecodeTokens = decodeTokens, generationDecodeMs = decodeMs,
                    status = if (it.stopping) it.status else "Responding") }
            }
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
                _state.update { it.copy(status = if (it.stopping) it.status else "Preparing response") }
                while (true) {
                    val step = checked(NativeRuntime.step())
                    val stepDecodeMs = step.optDouble("decode_ms", 0.0)
                    decodeMs += stepDecodeMs
                    val events = step.getJSONArray("events")
                    for (index in 0 until events.length()) {
                        val event = events.getJSONObject(index)
                        val text = event.optString("text")
                        if (event.has("token")) {
                            generatedTokens++
                            if (stepDecodeMs > 0) decodeTokens++
                        }
                        if (text.isNotEmpty() || event.has("token")) {
                            reply.append(text)
                            reply.flush()
                        }
                        if (event.has("completed")) {
                            reply.flush(force = true)
                            finish(event.getJSONObject("completed"))
                            completed = true
                        }
                    }
                    if (stopRequested.getAndSet(false) && step.getInt("pending") > 0) {
                        checked(NativeRuntime.cancel(requireNotNull(requestId)))
                        reply.flush(force = true)
                        finishPartial()
                        completed = true
                        break
                    }
                    if (step.getInt("pending") == 0) break
                    yield()
                }
                if (!completed) {
                    reply.flush(force = true)
                    finishPartial()
                }
            } catch (error: Throwable) {
                reply.flush(force = true)
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
    fun startPhoneRecording() = startRecording(CaptureSource.PHONE)

    @RequiresPermission(allOf = [Manifest.permission.CHANGE_WIFI_STATE, Manifest.permission.ACCESS_WIFI_STATE])
    fun startAccessoryRecording() = startRecording(CaptureSource.ACCESSORY)

    private fun startRecording(source: CaptureSource) {
        val current = _state.value
        if (!current.speechReady) {
            _state.update { it.copy(error = "Load a Whisper speech model in settings first") }
            return
        }
        if (current.generating || current.loadingSpeech || current.recording || current.transcribing ||
            current.pairing || current.accessoryBusy)
            return
        val accessoryProfile = if (source == CaptureSource.ACCESSORY) {
            runCatching { accessoryProfiles.profiles().firstOrNull() }.getOrNull() ?: run {
                _state.update { it.copy(error = "Pair an accessory before using its microphone") }
                return
            }
        } else {
            null
        }
        speechRecorder.prepare()
        stopRequestedAtMs = 0L
        _state.update {
            it.copy(
                recording = true,
                accessoryBusy = source == CaptureSource.ACCESSORY,
                recordingSeconds = 0f,
                status = if (source == CaptureSource.ACCESSORY) "Waking accessory microphone" else "Listening",
                error = null,
            )
        }
        recordingJob = viewModelScope.launch {
            val composerPrefix = current.composerText.trim()
            val draftBusy = AtomicBoolean()
            val draftEnabled = AtomicBoolean(true)
            var lastDraftSamples = 0
            val latestDraft = AtomicReference<SpeechDraft?>(null)
            val pendingDraft = AtomicReference<PendingDraft?>(null)
            val mergeTranscript = { transcript: String ->
                listOf(composerPrefix, transcript.trim()).filter(String::isNotEmpty).joinToString(" ")
            }
            fun publishPartial(payload: String, draft: Boolean, complete: Boolean = false) {
                val partial = checked(payload)
                if (partial.optBoolean("cancelled")) return
                val text = mergeTranscript(partial.optString("text"))
                val start = if (composerPrefix.isEmpty()) 0 else composerPrefix.length + 1
                _state.update { state ->
                    if ((draft && state.recording) || (!draft && state.transcribing)) {
                        val updated = state.withProvisionalTranscript(text, start, complete)
                        if (updated === state) state else updated.copy(
                            status = if (draft) "Draft transcript · ${partial.optString("language")}" else "Refining transcript",
                        )
                    } else state
                }
            }
            try {
                var reportedTenths = -1
                var preempted = false
                val onSamples: (Int, Int) -> Boolean = { count, speechEnd ->
                    val tenths = count / 1600
                    if (tenths != reportedTenths) {
                        reportedTenths = tenths
                        _state.update {
                            it.copy(
                                recordingSeconds = count / 16000f,
                                status = if (source == CaptureSource.ACCESSORY) "Listening on accessory" else it.status,
                            )
                        }
                    }
                    val cadence = count >= MINIMUM_DRAFT_SAMPLES &&
                        (lastDraftSamples == 0 || count - lastDraftSamples >= DRAFT_INTERVAL_SAMPLES)
                    // A pause after new speech gets a draft at once, so it is usually ready to become the final
                    // transcript by the time the user stops recording.
                    val paused = speechEnd > 0 && count - speechEnd >= PAUSE_SAMPLES &&
                        lastDraftSamples < speechEnd + SPEECH_TAIL_SAMPLES
                    // The draft in progress misses the latest speech, so give the thread to a covering one.
                    if (paused && draftBusy.get() && !preempted) {
                        preempted = true
                        NativeRuntime.cancelTranscription()
                    }
                    val requested = (cadence || paused) && draftEnabled.get() &&
                        draftBusy.compareAndSet(false, true)
                    if (requested) {
                        lastDraftSamples = count
                        preempted = false
                    }
                    requested
                }
                val onSnapshot: (FloatArray) -> Unit = { snapshot ->
                    val generation = NativeRuntime.transcriptionGeneration()
                    val job = viewModelScope.async(speechDispatcher) {
                        try {
                            val payload = NativeRuntime.transcribe(snapshot, "auto", 128, generation) { partial ->
                                publishPartial(partial, true)
                            }
                            val result = checked(payload)
                            if (result.optBoolean("cancelled")) null else {
                                publishPartial(payload, true, complete = true)
                                latestDraft.accumulateAndGet(SpeechDraft(snapshot.size, result)) { old, new ->
                                    if (old == null || new!!.samples >= old.samples) new else old
                                }
                                result
                            }
                        } catch (error: CancellationException) {
                            throw error
                        } catch (error: Throwable) {
                            draftEnabled.set(false)
                            _state.update {
                                it.copy(error = "Live transcript failed: ${error.userMessage()}")
                            }
                            null
                        } finally {
                            draftBusy.set(false)
                        }
                    }
                    pendingDraft.set(PendingDraft(snapshot.size, job))
                }
                val recording = if (source == CaptureSource.PHONE) {
                    speechRecorder.capture(onSamples, onSnapshot)
                } else {
                    accessoryNetwork.connect(requireNotNull(accessoryProfile)).use { lease ->
                        speechRecorder.captureAccessory(
                            AccessoryClient(accessoryProfile, lease.network.socketFactory),
                            onSamples,
                            onSnapshot,
                        )
                    }
                }
                _state.update {
                    it.copy(recording = false, transcribing = true, accessoryBusy = false, status = "Refining transcript")
                }
                val samples = recording.samples
                val covers = { size: Int ->
                    recording.speechEnd > 0 && size >= minOf(samples.size, recording.speechEnd + SPEECH_TAIL_SAMPLES)
                }
                var source = "draft"
                var result = latestDraft.get()?.takeIf { covers(it.samples) }?.result
                if (result == null) {
                    val pending = pendingDraft.get()
                    if (pending != null && covers(pending.samples)) {
                        source = "pending-draft"
                        result = pending.job.await()
                    }
                }
                if (result == null) {
                    source = "final"
                    // Drafts still queued or running are stale; the final pass replaces them.
                    val generation = NativeRuntime.cancelTranscription()
                    result = withContext(speechDispatcher) {
                        checked(NativeRuntime.transcribe(samples, "auto", 128, generation) { payload ->
                            publishPartial(payload, false)
                        })
                    }
                }
                val stoppedAt = stopRequestedAtMs
                // Abandon drafts still running so the speech thread is free for the next recording.
                if (source != "final") NativeRuntime.cancelTranscription()
                Log.i("KidiSpeech", "final source=$source audio_ms=${samples.size / 16} " +
                    "speech_end_ms=${recording.speechEnd / 16} " +
                    "stop_to_text_ms=${if (stoppedAt > 0) SystemClock.elapsedRealtime() - stoppedAt else -1} " +
                    "encode_ms=${result.optDouble("encode_ms")} decode_ms=${result.optDouble("decode_ms")}")
                val transcript = result.optString("text").trim()
                _state.update {
                    it.copy(
                        composerText = mergeTranscript(transcript),
                        provisionalTextStart = null,
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
                _state.update { if (it.accessoryBusy) it.copy(accessoryBusy = false) else it }
                recordingJob = null
            }
        }
    }

    fun stopRecording() {
        if (!_state.value.recording) return
        stopRequestedAtMs = SystemClock.elapsedRealtime()
        _state.update { it.copy(status = "Finishing recording") }
        speechRecorder.stop()
    }

    fun newChat() {
        if (_state.value.generating || _state.value.loadingChat || _state.value.importingImage ||
            _state.value.recording || _state.value.transcribing)
            return
        removePendingImage()
        _state.update { it.copy(loadingChat = true) }
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { chats.select(null) }
                _state.update { it.copy(activeChatId = null, messages = emptyList(), draft = "", composerText = "",
                    provisionalTextStart = null,
                    hasOlderMessages = false, loadingOlderMessages = false, loadingChat = false,
                    status = readyStatus(it), error = null) }
            } catch (error: Exception) {
                _state.update { it.copy(loadingChat = false, error = error.userMessage()) }
            }
        }
    }

    fun openChat(id: String) {
        val current = _state.value
        if (current.generating || current.recording || current.transcribing || current.loadingChat || current.importingImage || id == current.activeChatId) return
        removePendingImage()
        _state.update { it.copy(loadingChat = true) }
        viewModelScope.launch {
            try {
                val saved = withContext(Dispatchers.IO) {
                    requireNotNull(chats.load(id)) { "Chat no longer exists" }.also { chats.select(id) }
                }
                _state.update { it.copy(activeChatId = id, messages = saved.page.messages, hasOlderMessages = saved.page.hasMore,
                    draft = "", composerText = "", provisionalTextStart = null, loadingChat = false, loadingOlderMessages = false, error = null) }
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
            withContext(speechDispatcher) { NativeRuntime.unloadAsr() }
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
        val threads = _state.value.threadCount
        val accelerator = _state.value.chatAccelerator
        _state.update {
            it.copy(
                modelReady = false,
                visionReady = false,
                loadingModel = true,
                progressFile = "",
                chatExecution = "",
                status = "Loading model",
                error = null,
                modelError = null,
            )
        }
        viewModelScope.launch(runtimeDispatcher) {
            val started = SystemClock.elapsedRealtime()
            Log.i("KidiStartup", "gemma_start queue_ms=${started - queued}")
            runCatching {
                checked(NativeRuntime.configure(threads))
                val configured = SystemClock.elapsedRealtime()
                checked(NativeRuntime.load(model.directory.absolutePath, accelerator)).also {
                    Log.i("KidiStartup", "gemma_ready configure_ms=${configured - started} native_ms=${it.optDouble("load_ms")} stages=${it.optJSONObject("stages_ms")}")
                }
            }.onSuccess { loaded ->
                _state.update {
                    it.copy(
                        modelId = model.modelId,
                        modelRevision = model.revision,
                        modelReady = true,
                        visionReady = loaded.optBoolean("vision"),
                        loadingModel = false,
                        progress = 1f,
                        chatExecution = loaded.optString("backend"),
                        // Dictation can run while chat loads; keep its status visible.
                        status = if (it.recording || it.transcribing) it.status else "Ready on device",
                    )
                }
            }.onFailure { error ->
                Log.e("KidiStartup", "gemma_failed elapsed_ms=${SystemClock.elapsedRealtime() - started}")
                val message = "Chat model could not load: ${error.userMessage()}"
                _state.update {
                    it.copy(modelReady = false, loadingModel = false, error = message, modelError = message,
                        status = if (it.recording || it.transcribing) it.status else "Model offline")
                }
            }
        }
    }

    private fun loadSpeech(model: InstalledModel) {
        val queued = SystemClock.elapsedRealtime()
        val int8 = model.modelId == DEFAULT_SPEECH_MODEL_ID
        val threads = _state.value.threadCount
        val accelerator = _state.value.speechAccelerator
        _state.update {
            it.copy(
                speechReady = false,
                loadingSpeech = true,
                speechProgressFile = "",
                speechExecution = "",
                speechModelError = null,
                status = if (int8) "Preparing Whisper Small INT8" else "Loading speech model",
                error = null,
            )
        }
        viewModelScope.launch(speechDispatcher) {
            val started = SystemClock.elapsedRealtime()
            Log.i("KidiStartup", "whisper_start queue_ms=${started - queued}")
            runCatching {
                checked(NativeRuntime.configure(threads))
                val configured = SystemClock.elapsedRealtime()
                checked(NativeRuntime.loadAsr(model.directory.absolutePath, int8, accelerator)).also {
                    Log.i("KidiStartup", "whisper_ready configure_ms=${configured - started} native_ms=${it.optDouble("load_ms")} stages=${it.optJSONObject("stages_ms")}")
                }
            }.onSuccess { loaded ->
                _state.update {
                    it.copy(
                        speechExecution = loaded.optString("backend"),
                        speechModelId = model.modelId,
                        speechModelRevision = model.revision,
                        speechReady = true,
                        loadingSpeech = false,
                        speechProgress = 1f,
                        status = if (it.generating) it.status else readyStatus(it.copy(speechReady = true)),
                    )
                }
            }.onFailure { error ->
                val message = "Speech model could not load: ${error.userMessage()}"
                _state.update {
                    it.copy(
                        speechReady = false,
                        loadingSpeech = false,
                        status = if (it.generating) it.status else readyStatus(it),
                        error = message,
                        speechModelError = message,
                    )
                }
                Log.e("KidiStartup", "whisper_failed elapsed_ms=${SystemClock.elapsedRealtime() - started}")
            }
        }
    }

    private suspend fun finish(result: JSONObject) {
        Log.i("KidiChat", "reply prompt_tokens=${result.optInt("prompt_tokens")} " +
            "reused_tokens=${result.optInt("reused_prompt_tokens")} first_token_ms=${result.optDouble("first_token_ms")} " +
            "prefill_ms=${result.optDouble("prefill_ms")} output_tokens=${result.optInt("output_tokens")} " +
            "decode_ms=${result.optDouble("decode_ms")} decode_tokens=${result.optInt("decode_tokens")}")
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
                    .put("content", message.content)
                    .put("images", JSONArray().also { images ->
                        message.attachments.filter { it.kind == AttachmentKind.IMAGE }.forEach { image ->
                            images.put(requireNotNull(Uri.parse(image.localUri).path))
                        }
                    }),
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
        state.loadingModel -> "Loading model"
        state.speechReady -> "Speech ready"
        else -> "Model offline"
    }

    override fun onCleared() {
        stopRequested.set(true)
        speechRecorder.stop()
        recordingJob?.cancel()
        val id = requestId
        NativeRuntime.cancelTranscription()
        speechExecutor.execute { NativeRuntime.unloadAsr() }
        speechExecutor.shutdown()
        runtimeExecutor.execute {
            if (id != null) runCatching { NativeRuntime.cancel(id) }
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
        const val CHAT_ACCELERATOR_KEY = "chat-accelerator"
        const val SPEECH_ACCELERATOR_KEY = "speech-accelerator"
        const val PHOTO_SOURCE_KEY = "photo-source"
        const val SPEECH_SOURCE_KEY = "speech-source"
        const val MESSAGES_KEY = "messages"
        const val MINIMUM_DRAFT_SAMPLES = 12800
        const val DRAFT_INTERVAL_SAMPLES = 19200
        const val PAUSE_SAMPLES = 4800
        const val SPEECH_TAIL_SAMPLES = 3200
    }
}

private val KidiUiState.chatBusyState: Boolean
    get() = generating || recording || transcribing || loadingModel || loadingChat || importingImage ||
        pairing || accessoryBusy

private fun SharedPreferences.captureSource(key: String): CaptureSource =
    runCatching { CaptureSource.valueOf(getString(key, null) ?: CaptureSource.PHONE.name) }
        .getOrDefault(CaptureSource.PHONE)

private fun defaultThreadCount() = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)