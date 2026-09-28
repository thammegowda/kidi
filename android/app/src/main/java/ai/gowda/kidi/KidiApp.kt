package ai.gowda.kidi

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.Image
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddComment
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Download
import androidx.compose.material.icons.outlined.EditNote
import androidx.compose.material.icons.outlined.Lightbulb
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.AttachFile
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.PlainTooltip
import androidx.compose.material3.TooltipBox
import androidx.compose.material3.TooltipDefaults
import androidx.compose.material3.rememberTooltipState
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.rememberDrawerState
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Slider
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.snapshotFlow
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.launch

@Composable
internal fun KidiApp(viewModel: ChatViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    KidiTheme {
        KidiScreen(state, viewModel)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@SuppressLint("MissingPermission")
@Composable
private fun KidiScreen(state: KidiUiState, viewModel: ChatViewModel) {
    var settingsOpen by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }
    var confirmSpeechDelete by remember { mutableStateOf(false) }
    val snackbar = remember { SnackbarHostState() }
    val context = LocalContext.current
    val microphonePermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) viewModel.startRecording()
        else viewModel.reportError("Microphone permission is required for speech dictation")
    }
    val startRecording = {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
            viewModel.startRecording()
        else
            microphonePermission.launch(Manifest.permission.RECORD_AUDIO)
    }
    LaunchedEffect(state.error) {
        state.error?.let {
            snackbar.showSnackbar(it)
            viewModel.clearError()
        }
    }

    ChatWorkspace(
        state = state,
        snackbar = snackbar,
        onNewChat = viewModel::newChat,
        onSettings = { settingsOpen = true },
        onTextChange = viewModel::setComposerText,
        onSend = viewModel::send,
        onStop = viewModel::stop,
        onRecord = startRecording,
        onStopRecording = viewModel::stopRecording,
        onHistoryOpen = viewModel::refreshHistory,
        onHistoryQuery = viewModel::searchHistory,
        onOpenChat = viewModel::openChat,
        onMoreHistory = viewModel::loadMoreHistory,
        onOlderMessages = viewModel::loadOlderMessages,
    )

    if (settingsOpen) {
        SettingsSheet(
            state = state,
            onDismiss = { settingsOpen = false },
            onModelId = viewModel::setModelId,
            onSpeechModelId = viewModel::setSpeechModelId,
            onThreads = viewModel::setThreadCount,
            onTokens = viewModel::setMaximumTokens,
            onInstall = viewModel::installModel,
            onCancelChat = viewModel::cancelModelInstall,
            onInstallSpeech = viewModel::installSpeechModel,
            onCancelSpeech = viewModel::cancelSpeechInstall,
            onDelete = { confirmDelete = true },
            onDeleteSpeech = { confirmSpeechDelete = true },
        )
    }
    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Remove local model?") },
            text = { Text("This deletes the downloaded checkpoint from this device. Your conversation stays here.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    settingsOpen = false
                    viewModel.removeModel()
                }) { Text("Remove") }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Keep") } },
        )
    }
    if (confirmSpeechDelete) {
        AlertDialog(
            onDismissRequest = { confirmSpeechDelete = false },
            title = { Text("Remove speech model?") },
            text = { Text("This deletes the downloaded Whisper checkpoint from this device.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmSpeechDelete = false
                    viewModel.removeSpeechModel()
                }) { Text("Remove") }
            },
            dismissButton = { TextButton(onClick = { confirmSpeechDelete = false }) { Text("Keep") } },
        )
    }
}

private val KidiUiState.runtimeBusy: Boolean
    get() = generating || recording || transcribing || loadingModel || loadingSpeech || loadingChat

private val KidiUiState.chatBusy: Boolean
    get() = generating || recording || transcribing || loadingModel || loadingChat

private fun modelLabel(modelId: String) = when (modelId) {
    "google/gemma-4-E2B-it-qat-mobile-transformers" -> "Gemma 4 E2B"
    "openai/whisper-tiny" -> "Whisper Tiny"
    "openai/whisper-base" -> "Whisper Base"
    "openai/whisper-small" -> "Whisper Small INT8"
    else -> modelId.substringAfterLast('/')
}

@Composable
internal fun ChatWorkspace(
    state: KidiUiState,
    snackbar: SnackbarHostState,
    onNewChat: () -> Unit,
    onSettings: () -> Unit,
    onTextChange: (String) -> Unit,
    onSend: (String) -> Unit,
    onStop: () -> Unit,
    onRecord: () -> Unit,
    onStopRecording: () -> Unit,
    onHistoryOpen: () -> Unit = {},
    onHistoryQuery: (String) -> Unit = {},
    onOpenChat: (String) -> Unit = {},
    onMoreHistory: () -> Unit = {},
    onOlderMessages: () -> Unit = {},
) {
    val drawer = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    val closeHistory: () -> Unit = { scope.launch { drawer.close() }; Unit }
    val canSwitch = !state.generating && !state.recording && !state.transcribing && !state.loadingChat
    ModalNavigationDrawer(
        drawerState = drawer,
        gesturesEnabled = drawer.isOpen,
        drawerContent = {
            ChatHistoryDrawer(state, drawer.isOpen, canSwitch, onHistoryQuery, onMoreHistory, onHistoryOpen,
                onSelect = { id -> if (canSwitch) { onOpenChat(id); closeHistory() } },
                onNewChat = { if (canSwitch) { onNewChat(); closeHistory() } }, onClose = closeHistory)
        },
    ) {
        Scaffold(
            modifier = Modifier.fillMaxSize().imePadding(),
            containerColor = MaterialTheme.colorScheme.background,
            contentWindowInsets = WindowInsets(0, 0, 0, 0),
            snackbarHost = { SnackbarHost(snackbar) },
            topBar = { KidiHeader(state, onNewChat, onSettings) {
                onHistoryOpen()
                scope.launch { drawer.open() }
            } },
            bottomBar = { Composer(state, onTextChange, onSend, onStop, onRecord, onStopRecording, onSettings) },
        ) { padding ->
            Box(Modifier.fillMaxSize().padding(padding).consumeWindowInsets(padding), contentAlignment = Alignment.TopCenter) {
                Conversation(state, onSettings, onTextChange, Modifier.widthIn(max = 760.dp).fillMaxSize(), onOlderMessages)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun ToolButton(
    icon: ImageVector,
    label: String,
    onClick: () -> Unit,
    enabled: Boolean = true,
    tint: Color = MaterialTheme.colorScheme.onSurfaceVariant,
) {
    TooltipBox(
        positionProvider = TooltipDefaults.rememberPlainTooltipPositionProvider(),
        tooltip = { PlainTooltip { Text(label) } },
        state = rememberTooltipState(),
    ) {
        IconButton(onClick = onClick, enabled = enabled, modifier = Modifier.size(48.dp)) {
            Icon(icon, label, Modifier.size(22.dp), tint = if (enabled) tint else tint.copy(alpha = 0.38f))
        }
    }
}

@Composable
private fun KidiHeader(state: KidiUiState, onNewChat: () -> Unit, onSettings: () -> Unit, onHistory: () -> Unit) {
    Column(Modifier.background(MaterialTheme.colorScheme.surface).statusBarsPadding()) {
        Row(
            modifier = Modifier.fillMaxWidth().heightIn(min = 64.dp).padding(start = 20.dp, end = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onHistory, modifier = Modifier.size(48.dp)) {
                Image(painterResource(R.drawable.kidi_logo), "Chat history", Modifier.size(34.dp))
            }
            Spacer(Modifier.width(10.dp))
            Text("Kidi", Modifier.weight(1f), style = MaterialTheme.typography.titleLarge)
            ToolButton(Icons.Default.AddComment, "New chat", onNewChat, !state.chatBusy)
            ToolButton(Icons.Outlined.Tune, "Model settings", onSettings)
        }
        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
        Row(
            Modifier.fillMaxWidth().clickable(onClick = onSettings).heightIn(min = 44.dp)
                .padding(horizontal = 20.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Icon(
                if (state.modelReady) Icons.Outlined.CheckCircle else Icons.Default.Memory,
                null, Modifier.size(15.dp),
                tint = if (state.modelReady) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(modelLabel(state.modelId), Modifier.weight(1f), style = MaterialTheme.typography.labelMedium,
                maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text(
                when {
                    state.loadingModel -> "Loading"
                    state.modelReady -> "On device"
                    else -> "Offline"
                },
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (state.speechReady) {
                Icon(Icons.Default.Mic, "Speech ready", Modifier.size(15.dp), tint = MaterialTheme.colorScheme.primary)
            }
        }
        if (state.loadingModel || state.loadingSpeech) {
            LinearProgressIndicator(Modifier.fillMaxWidth().height(2.dp))
        } else {
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
        }
    }
}

@Composable
private fun Conversation(
    state: KidiUiState,
    onSettings: () -> Unit,
    onPrompt: (String) -> Unit,
    modifier: Modifier = Modifier,
    onOlderMessages: () -> Unit = {},
) {
    val listState = rememberLazyListState()
    var followResponse by remember { mutableStateOf(true) }
    LaunchedEffect(listState) {
        snapshotFlow { listState.isScrollInProgress to listState.canScrollForward }.distinctUntilChanged().collect {
            (scrolling, canScroll) -> if (scrolling) followResponse = !canScroll
        }
    }
    val itemCount = state.messages.size + (if (state.generating) 1 else 0) + (if (state.hasOlderMessages) 1 else 0)
    LaunchedEffect(state.activeChatId) { followResponse = true }
    LaunchedEffect(state.activeChatId, state.messages.lastOrNull()?.id, state.draft) {
        if (itemCount > 0 && followResponse) listState.scrollToItem(itemCount - 1)
    }
    if (state.messages.isEmpty() && !state.generating) {
        EmptyConversation(state, onSettings, onPrompt, modifier)
        return
    }
    Box(modifier) {
        LazyColumn(
            state = listState,
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(horizontal = 20.dp, vertical = 24.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp),
        ) {
            if (state.hasOlderMessages) item(key = "earlier") {
                TextButton(onClick = onOlderMessages, enabled = !state.loadingOlderMessages && !state.chatBusy,
                    modifier = Modifier.fillMaxWidth()) {
                    Text(if (state.loadingOlderMessages) "Loading messages" else "Earlier messages")
                }
            }
            items(state.messages.size, key = { state.messages[it].id }) { index -> MessageRow(state.messages[index]) }
            if (state.generating) item(key = "draft") {
                MessageRow(ChatMessage(MessageRole.ASSISTANT, state.draft), active = true)
            }
        }
    }
}

@Composable
private fun EmptyConversation(
    state: KidiUiState,
    onSettings: () -> Unit,
    onPrompt: (String) -> Unit,
    modifier: Modifier,
) {
    Column(
        modifier = modifier.verticalScroll(rememberScrollState()).padding(horizontal = 24.dp, vertical = 40.dp),
    ) {
        Image(painterResource(R.drawable.kidi_logo), null, Modifier.size(56.dp), contentScale = ContentScale.Fit)
        Spacer(Modifier.height(20.dp))
        Text("New conversation", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(8.dp))
        Text(modelLabel(state.modelId), style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(32.dp))
        if (!state.modelReady) {
            Text(if (state.loadingModel) "Preparing model" else "No chat model loaded",
                style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(12.dp))
            Button(onClick = onSettings, shape = RoundedCornerShape(8.dp)) {
                Icon(Icons.Default.Memory, null, Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Text("Manage models")
            }
        } else {
            val prompts = listOf(
                Triple(Icons.Outlined.Lightbulb, "Explain a concept", "Explain this concept in simple terms: "),
                Triple(Icons.Outlined.EditNote, "Refine some writing", "Help me improve this writing: "),
                Triple(Icons.Outlined.AutoAwesome, "Explore an idea", "Help me think through this idea: "),
            )
            prompts.forEach { (icon, title, prompt) ->
                Row(
                    Modifier.fillMaxWidth().clickable(enabled = !state.chatBusy) { onPrompt(prompt) }
                        .padding(vertical = 18.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(16.dp),
                ) {
                    Icon(icon, null, Modifier.size(22.dp), tint = MaterialTheme.colorScheme.primary)
                    Text(title, Modifier.weight(1f), style = MaterialTheme.typography.bodyLarge)
                    Icon(Icons.AutoMirrored.Filled.ArrowForward, null, Modifier.size(18.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))
            }
        }
    }
}

@Composable
private fun MessageRow(message: ChatMessage, active: Boolean = false) {
    val user = message.role == MessageRole.USER
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (user) Arrangement.End else Arrangement.Start,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth(if (user) 0.9f else 1f)
                .background(
                    if (user) MaterialTheme.colorScheme.surfaceContainer else Color.Transparent,
                    RoundedCornerShape(8.dp),
                )
                .padding(if (user) 16.dp else 0.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (!user) Image(painterResource(R.drawable.kidi_logo), null, Modifier.size(22.dp))
                Text(message.sender.name, style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Spacer(Modifier.height(10.dp))
            if (!user && message.content.isNotEmpty()) {
                MarkdownReply(message.content, Modifier.fillMaxWidth())
            } else {
                SelectionContainer {
                    Text(message.content.ifEmpty { when {
                        active -> "Thinking..."
                        message.status == MessageStatus.STOPPED -> "Response stopped"
                        message.status == MessageStatus.FAILED -> "Response failed"
                        else -> if (user) "" else "Empty response"
                    } },
                        style = MaterialTheme.typography.bodyLarge)
                }
            }
            message.attachments.forEach { attachment ->
                Row(Modifier.fillMaxWidth().padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Icon(Icons.Default.AttachFile, attachment.kind.name, Modifier.size(18.dp))
                    Text(attachment.name, Modifier.weight(1f), maxLines = 2, overflow = TextOverflow.Ellipsis)
                }
            }
            message.stats?.let { stats ->
                val speed = if (stats.decodeMs > 0 && stats.decodeTokens > 0) {
                    "%.1f tok/s".format(stats.decodeTokens * 1000 / stats.decodeMs)
                } else {
                    "-- tok/s"
                }
                Spacer(Modifier.height(8.dp))
                Text(
                    "${stats.tokens} tokens · ${"%.1f".format(stats.elapsedMs / 1000)} s · $speed",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (active && message.content.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(modifier = Modifier.width(72.dp))
            }
        }
    }
}

@Composable
private fun Composer(
    state: KidiUiState,
    onTextChange: (String) -> Unit,
    onSend: (String) -> Unit,
    onStop: () -> Unit,
    onRecord: () -> Unit,
    onStopRecording: () -> Unit,
    onSettings: () -> Unit,
) {
    val keyboard = LocalSoftwareKeyboardController.current
    val canSend = state.modelReady && !state.chatBusy && state.composerText.isNotBlank()
    val submit = {
        if (canSend) {
            keyboard?.hide()
            onSend(state.composerText)
        }
    }
    Column(
        Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.background).navigationBarsPadding(),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Column(
            Modifier.widthIn(max = 760.dp).fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
        ) {
            Surface(
                shape = RoundedCornerShape(8.dp),
                color = MaterialTheme.colorScheme.surface,
                border = BorderStroke(1.dp, if (state.recording) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.outlineVariant),
            ) {
                Column {
                    if (state.recording || state.transcribing || state.generating) {
                        Row(
                            Modifier.fillMaxWidth().padding(start = 16.dp, end = 12.dp, top = 12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            if (state.recording) Icon(Icons.Default.Mic, null, Modifier.size(16.dp),
                                tint = MaterialTheme.colorScheme.primary)
                            else CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 1.5.dp)
                            Text(
                                when {
                                    state.recording -> "Listening · ${"%.1f".format(state.recordingSeconds)} s"
                                    state.transcribing -> "Refining transcript"
                                    state.stopping -> "Stopping"
                                    else -> "Generating"
                                },
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.primary,
                            )
                        }
                    }
                    TextField(
                        value = state.composerText,
                        onValueChange = onTextChange,
                        modifier = Modifier.fillMaxWidth().testTag("message-composer"),
                        readOnly = state.recording || state.transcribing || state.loadingChat,
                        placeholder = { Text(if (state.recording) "Listening..." else "Message Kidi") },
                        maxLines = 6,
                        textStyle = MaterialTheme.typography.bodyLarge,
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                        keyboardActions = KeyboardActions(onSend = { submit() }),
                        colors = TextFieldDefaults.colors(
                            focusedContainerColor = Color.Transparent, unfocusedContainerColor = Color.Transparent,
                            focusedIndicatorColor = Color.Transparent, unfocusedIndicatorColor = Color.Transparent,
                        ),
                    )
                    Row(
                        Modifier.fillMaxWidth().padding(start = 4.dp, end = 8.dp, bottom = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        ToolButton(
                            if (state.recording) Icons.Default.Stop else Icons.Default.Mic,
                            if (state.recording) "Stop recording" else if (state.speechReady) "Dictate message" else "Set up dictation",
                            {
                                keyboard?.hide()
                                when {
                                    state.recording -> onStopRecording()
                                    state.speechReady -> onRecord()
                                    else -> onSettings()
                                }
                            },
                            enabled = state.recording || !state.runtimeBusy,
                            tint = if (state.recording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            when {
                                state.recording -> modelLabel(state.speechModelId)
                                state.transcribing -> "Final pass"
                                state.loadingModel -> "Preparing chat"
                                state.loadingSpeech -> "Preparing speech"
                                state.modelReady -> "On-device AI"
                                else -> "Model offline"
                            },
                            Modifier.weight(1f), style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                        FilledIconButton(
                            onClick = if (state.generating) onStop else submit,
                            enabled = if (state.generating) !state.stopping else canSend,
                            modifier = Modifier.size(48.dp),
                            shape = RoundedCornerShape(8.dp),
                            colors = IconButtonDefaults.filledIconButtonColors(
                                containerColor = MaterialTheme.colorScheme.primary,
                                contentColor = MaterialTheme.colorScheme.onPrimary,
                            ),
                        ) {
                            Icon(if (state.generating) Icons.Default.Stop else Icons.Default.ArrowUpward,
                                if (state.generating) "Stop generation" else "Send message", Modifier.size(22.dp))
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun SettingsSheet(
    state: KidiUiState,
    onDismiss: () -> Unit,
    onModelId: (String) -> Unit,
    onSpeechModelId: (String) -> Unit,
    onThreads: (Int) -> Unit,
    onTokens: (Int) -> Unit,
    onInstall: () -> Unit,
    onCancelChat: () -> Unit,
    onInstallSpeech: () -> Unit,
    onCancelSpeech: () -> Unit,
    onDelete: () -> Unit,
    onDeleteSpeech: () -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    var selectedTab by rememberSaveable { mutableStateOf(0) }
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        sheetMaxWidth = 600.dp,
        containerColor = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(topStart = 8.dp, topEnd = 8.dp),
    ) {
        Column(Modifier.fillMaxWidth().fillMaxHeight(0.9f)) {
            Row(Modifier.fillMaxWidth().padding(start = 24.dp, end = 12.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Text("Settings", Modifier.weight(1f), style = MaterialTheme.typography.titleLarge)
                ToolButton(Icons.Default.Close, "Close settings", onDismiss)
            }
            TabRow(selectedTabIndex = selectedTab, containerColor = MaterialTheme.colorScheme.surface) {
                listOf("Models", "Inference").forEachIndexed { index, label ->
                    Tab(selected = selectedTab == index, onClick = { selectedTab = index }, text = { Text(label) })
                }
            }
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
                if (selectedTab == 0) {
                    ModelSection(
                        title = "Chat", icon = Icons.AutoMirrored.Filled.Chat,
                        modelId = state.modelId, revision = state.modelRevision,
                        ready = state.modelReady, loading = state.loadingModel,
                        progress = state.progress, file = state.progressFile, busy = state.runtimeBusy,
                        onModelId = onModelId, onInstall = onInstall, onCancel = onCancelChat,
                        onDelete = onDelete, deleteLabel = "Remove local model",
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    ModelSection(
                        title = "Voice", icon = Icons.Default.Mic,
                        modelId = state.speechModelId, revision = state.speechModelRevision,
                        ready = state.speechReady, loading = state.loadingSpeech,
                        progress = state.speechProgress, file = state.speechProgressFile, busy = state.runtimeBusy,
                        onModelId = onSpeechModelId, onInstall = onInstallSpeech, onCancel = onCancelSpeech,
                        onDelete = onDeleteSpeech, deleteLabel = "Remove speech model",
                    )
                } else {
                    InferenceSettings(state, onThreads, onTokens)
                }
                Spacer(Modifier.height(24.dp))
            }
        }
    }
}

@Composable
private fun ModelSection(
    title: String,
    icon: ImageVector,
    modelId: String,
    revision: String?,
    ready: Boolean,
    loading: Boolean,
    progress: Float,
    file: String,
    busy: Boolean,
    onModelId: (String) -> Unit,
    onInstall: () -> Unit,
    onCancel: () -> Unit,
    onDelete: () -> Unit,
    deleteLabel: String,
) {
    var detailsOpen by rememberSaveable { mutableStateOf(false) }
    val downloading = loading && file.isNotEmpty() && progress < 1f
    Column(Modifier.fillMaxWidth().padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Icon(icon, null, Modifier.size(20.dp), tint = MaterialTheme.colorScheme.primary)
            Text(title, Modifier.weight(1f), style = MaterialTheme.typography.titleMedium)
            Text(
                when { loading -> if (downloading) "Downloading" else "Preparing"; ready -> "Ready"; else -> "Not loaded" },
                style = MaterialTheme.typography.labelMedium,
                color = if (ready) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Text(modelLabel(modelId), style = MaterialTheme.typography.titleLarge)
        Text(if (title == "Voice") "${if (modelId == "openai/whisper-small") "INT8" else "FP32"} · CPU · Automatic language · 30 seconds" else "Text generation · CPU",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = { detailsOpen = !detailsOpen }, contentPadding = PaddingValues(0.dp)) {
                Text("Model details")
                Icon(if (detailsOpen) Icons.Default.ExpandLess else Icons.Default.ExpandMore, null, Modifier.size(20.dp))
            }
            Spacer(Modifier.weight(1f))
            if (ready) ToolButton(Icons.Default.DeleteOutline, deleteLabel, onDelete, !busy)
        }
        if (detailsOpen) {
            if (ready || loading) {
                SelectionContainer {
                    Text(modelId, style = MaterialTheme.typography.bodyMedium)
                }
            } else {
                OutlinedTextField(
                    value = modelId, onValueChange = onModelId, modifier = Modifier.fillMaxWidth(),
                    label = { Text("Hugging Face repository") },
                    enabled = !busy, minLines = 1, maxLines = 3, shape = RoundedCornerShape(8.dp),
                    textStyle = MaterialTheme.typography.bodyMedium,
                )
            }
            revision?.let {
                Text("Revision ${it.take(8)}", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        when {
            loading -> {
                if (downloading) {
                    LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(file, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall,
                            maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text("${(progress * 100).toInt()}%", style = MaterialTheme.typography.labelMedium)
                        ToolButton(Icons.Default.Close, "Cancel $title download", onCancel)
                    }
                } else {
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                }
            }
            !ready -> Button(onClick = onInstall, enabled = !busy, shape = RoundedCornerShape(8.dp)) {
                Icon(Icons.Outlined.Download, null, Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Text(if (title == "Voice") "Load speech model" else "Load chat model")
            }
        }
    }
}

@Composable
private fun InferenceSettings(state: KidiUiState, onThreads: (Int) -> Unit, onTokens: (Int) -> Unit) {
    var threads by remember(state.threadCount) { mutableStateOf(state.threadCount.toFloat()) }
    var outputMenu by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().padding(24.dp), verticalArrangement = Arrangement.spacedBy(20.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("CPU threads", Modifier.weight(1f), style = MaterialTheme.typography.titleMedium)
            Text(threads.toInt().toString(), style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.primary)
        }
        Slider(
            value = threads, onValueChange = { threads = it }, onValueChangeFinished = { onThreads(threads.toInt()) },
            valueRange = 1f..8f, steps = 6, enabled = !state.runtimeBusy,
        )
        HorizontalDivider()
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Output tokens", Modifier.weight(1f), style = MaterialTheme.typography.titleMedium)
            Box {
                TextButton(onClick = { outputMenu = true }, enabled = !state.runtimeBusy) {
                    Text(state.maximumTokens.toString())
                    Icon(Icons.Default.ExpandMore, null, Modifier.size(20.dp))
                }
                DropdownMenu(expanded = outputMenu, onDismissRequest = { outputMenu = false }) {
                    (listOf(256, 1024, 4096, 8192) + state.maximumTokens).distinct().sorted().forEach { tokens ->
                        DropdownMenuItem(text = { Text(tokens.toString()) }, onClick = {
                            onTokens(tokens)
                            outputMenu = false
                        })
                    }
                }
            }
        }
        HorizontalDivider()
        Text("Runtime", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row {
            Text("Context window", Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
            Text("9,216 tokens", style = MaterialTheme.typography.bodyMedium)
        }
        Row {
            Text("Backend", Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
            Text("YNNPACK CPU", style = MaterialTheme.typography.bodyMedium)
        }
    }
}