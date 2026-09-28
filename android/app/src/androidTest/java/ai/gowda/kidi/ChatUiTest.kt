package ai.gowda.kidi

import androidx.compose.foundation.layout.padding
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertIsNotDisplayed
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performScrollToNode
import androidx.compose.ui.test.performTextReplacement
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.longClick
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertSame
import org.junit.Rule
import org.junit.Test

class ChatUiTest {
    @get:Rule val compose = createComposeRule()

    @Test
    fun provisionalTranscriptStylesOnlySpeechAndPreservesOffsets() {
        val source = androidx.compose.ui.text.AnnotatedString("Typed prefix spoken draft")
        val draft = ProvisionalTranscriptTransformation(13, androidx.compose.ui.graphics.Color.Gray).filter(source)
        assertEquals(source.text, draft.text.text)
        assertEquals(13, draft.text.spanStyles.single().start)
        assertEquals(source.length, draft.text.spanStyles.single().end)
        assertEquals(14, draft.offsetMapping.originalToTransformed(14))
        val finalized = ProvisionalTranscriptTransformation(null, androidx.compose.ui.graphics.Color.Gray).filter(source)
        assertEquals(source, finalized.text)
        val state = KidiUiState(composerText = source.text, provisionalTextStart = 13)
        assertSame(state, state.withProvisionalTranscript(source.text, 13))
        assertSame(state, state.withProvisionalTranscript("Typed prefix spoken", 13))
        val corrected = state.withProvisionalTranscript("Typed prefix correction", 13, complete = true)
        assertEquals("Typed prefix correction", corrected.composerText)
        assertEquals(13, corrected.provisionalTextStart)
    }

    @Test
    fun responseStatusShowsLiveTokensAndDecodeSpeedUntilCompletion() {
        val state = mutableStateOf(KidiUiState(modelReady = true, generating = true, status = "Preparing response"))
        compose.setContent {
            KidiTheme {
                ChatWorkspace(state = state.value, snackbar = SnackbarHostState(), onNewChat = {}, onSettings = {},
                    onTextChange = {}, onSend = {}, onStop = { state.value = state.value.copy(stopping = true) },
                    onRecord = {}, onStopRecording = {})
            }
        }
        compose.onNodeWithText("Preparing response").assertIsDisplayed()
        compose.onNodeWithText("-- tok/s").assertIsDisplayed()
        compose.runOnIdle {
            state.value = state.value.copy(draft = "Streaming response", status = "Responding", generationTokens = 21,
                generationDecodeTokens = 20, generationDecodeMs = 1000.0)
        }
        compose.onNodeWithText("Responding").assertIsDisplayed()
        compose.onNodeWithText("21 tokens").assertIsDisplayed()
        compose.onNodeWithText("%.1f tok/s".format(20.0)).assertIsDisplayed()
        compose.onNodeWithContentDescription("Stop generation").performClick()
        compose.onNodeWithText("Stopping").assertIsDisplayed()
        compose.runOnIdle { state.value = state.value.copy(generating = false, stopping = false) }
        compose.onNodeWithTag("generation-progress").assertDoesNotExist()
    }

    @Test
    fun modelDownloadRequiresExplicitConfirmation() {
        var downloads = 0
        compose.setContent {
            KidiTheme {
                SettingsSheet(state = KidiUiState(), onDismiss = {}, onModelId = {}, onSpeechModelId = {},
                    onThreads = {}, onTokens = {}, onInstall = { downloads++ }, onCancelChat = {},
                    onInstallSpeech = {}, onCancelSpeech = {}, onDelete = {}, onDeleteSpeech = {})
            }
        }
        compose.runOnIdle { assertEquals(0, downloads) }
        compose.onNodeWithText("Download chat model").performScrollTo().performClick()
        compose.onNodeWithText("About 2.5 GB download; allow 3 GB free storage.").assertIsDisplayed()
        compose.runOnIdle { assertEquals(0, downloads) }
        compose.onNodeWithText("Not now").performClick()
        compose.runOnIdle { assertEquals(0, downloads) }
        compose.onNodeWithText("Download chat model").performClick()
        compose.onNodeWithText("Download", useUnmergedTree = true).performClick()
        compose.runOnIdle { assertEquals(1, downloads) }
    }

    @Test
    fun photoControlsRespectVisionCapabilityAndAllowImageOnlySend() {
        val image = MessageAttachment(kind = AttachmentKind.IMAGE, localUri = "file:///unavailable-test-photo.jpg",
            name = "Photo", mimeType = "image/jpeg", sizeBytes = 123)
        val state = mutableStateOf(KidiUiState(modelReady = true))
        var captured = false
        var sent = false
        compose.setContent {
            KidiTheme {
                ChatWorkspace(state = state.value, snackbar = SnackbarHostState(), onNewChat = {}, onSettings = {},
                    onTextChange = {}, onSend = { sent = true }, onStop = {}, onRecord = {}, onStopRecording = {},
                    onTakePhoto = { captured = true },
                    onPickImage = { state.value = state.value.copy(pendingImages = listOf(image)) },
                    onRemoveImage = { state.value = state.value.copy(pendingImages = emptyList()) })
            }
        }
        compose.onNodeWithContentDescription("Take photo").assertIsNotEnabled()
        compose.onNodeWithContentDescription("Choose photo").assertIsNotEnabled()
        compose.runOnIdle { state.value = state.value.copy(visionReady = true) }
        compose.onNodeWithContentDescription("Take photo").performClick()
        compose.runOnIdle { assertEquals(true, captured) }
        compose.onNodeWithContentDescription("Choose photo").performClick()
        compose.onNodeWithContentDescription("Remove photo").assertIsDisplayed()
        compose.onNodeWithContentDescription("Send message").assertIsEnabled().performClick()
        compose.runOnIdle { assertEquals(true, sent) }
        compose.onNodeWithContentDescription("Remove photo").performClick()
        compose.onNodeWithContentDescription("Send message").assertIsNotEnabled()
        compose.runOnIdle { state.value = state.value.copy(importingImage = true) }
        compose.onNodeWithContentDescription("Choose photo").assertIsNotEnabled()
    }

    @Test
    fun draftsRemainVisibleAndSendRequiresReadyIdleRuntime() {
        val state = mutableStateOf(KidiUiState())
        var sent = ""
        var settingsOpened = false
        compose.setContent {
            KidiTheme {
                ChatWorkspace(
                    state = state.value, snackbar = SnackbarHostState(), onNewChat = {},
                    onSettings = { settingsOpened = true },
                    onTextChange = { state.value = state.value.copy(composerText = it) },
                    onSend = { sent = it }, onStop = {}, onRecord = {},
                    onStopRecording = { state.value = state.value.copy(recording = false, transcribing = true) },
                )
            }
        }
        compose.onNodeWithTag("message-composer").performTextReplacement("Existing draft")
        compose.onNodeWithContentDescription("Send message").assertIsNotEnabled()
        compose.onNodeWithContentDescription("Set up dictation").performClick()
        compose.runOnIdle {
            state.value = state.value.copy(loadingModel = true, loadingSpeech = true)
        }
        compose.onNodeWithTag("message-composer").performTextReplacement("Typing during model loading")
        compose.onNodeWithContentDescription("Send message").assertIsNotEnabled()
        compose.runOnIdle {
            state.value = state.value.copy(modelReady = true, loadingModel = false)
        }
        compose.onNodeWithContentDescription("Send message").assertIsEnabled().performClick()
        compose.runOnIdle { assertEquals("Typing during model loading", sent) }
        compose.runOnIdle {
            assertEquals(true, settingsOpened)
            state.value = state.value.copy(modelReady = true, speechReady = true, loadingSpeech = false, recording = true,
                composerText = "Existing draft with live speech", provisionalTextStart = 15, recordingSeconds = 4.2f)
        }
        compose.onNodeWithText("Existing draft with live speech").assertIsDisplayed()
        val composerHeight = compose.onNodeWithTag("message-composer").fetchSemanticsNode().boundsInRoot.height
        compose.runOnIdle {
            state.value = state.value.copy(recordingSeconds = 5.2f,
                composerText = "Existing draft\nwith several\nlines of live\nspeech that scrolls\ninside the composer")
        }
        assertEquals(composerHeight, compose.onNodeWithTag("message-composer").fetchSemanticsNode().boundsInRoot.height, 0f)
        compose.runOnIdle { state.value = state.value.copy(composerText = "Short correction") }
        assertEquals(composerHeight, compose.onNodeWithTag("message-composer").fetchSemanticsNode().boundsInRoot.height, 0f)
        compose.onNodeWithContentDescription("Send message").assertIsNotEnabled()
        compose.onNodeWithContentDescription("Stop recording").assertIsEnabled().performClick()
        compose.onNodeWithText("Refining transcript").assertIsDisplayed()
        assertEquals(composerHeight, compose.onNodeWithTag("message-composer").fetchSemanticsNode().boundsInRoot.height, 0f)
        compose.onNodeWithContentDescription("Send message").assertIsNotEnabled()
        compose.runOnIdle {
            state.value = state.value.copy(transcribing = false, composerText = "Corrected transcript", provisionalTextStart = null)
        }
        compose.onNodeWithContentDescription("Send message").assertIsEnabled().performClick()
        compose.runOnIdle { assertEquals("Corrected transcript", sent) }
    }

    @Test
    fun modelAndInferenceTabsKeepActionsSeparate() {
        val state = mutableStateOf(KidiUiState(modelReady = true, speechReady = true))
        var speechRemoved = false
        compose.setContent {
            KidiTheme {
                SettingsSheet(
                    state = state.value, onDismiss = {}, onModelId = {}, onSpeechModelId = {},
                    onThreads = {}, onTokens = { state.value = state.value.copy(maximumTokens = it) },
                    onInstall = {}, onCancelChat = {}, onInstallSpeech = {}, onCancelSpeech = {},
                    onDelete = {}, onDeleteSpeech = { speechRemoved = true },
                )
            }
        }
        compose.onNodeWithText("Gemma 4 E2B").assertIsDisplayed()
        compose.onNodeWithText("Whisper Small INT8").performScrollTo().assertIsDisplayed()
        compose.onNodeWithContentDescription("Remove speech model").performScrollTo().performClick()
        compose.runOnIdle { assertEquals(true, speechRemoved) }
        compose.onNodeWithText("Inference").performClick()
        compose.onNodeWithText("Output tokens").performScrollTo()
        compose.onNodeWithText("1024").performClick()
        compose.onNodeWithText("256").performClick()
        compose.runOnIdle { assertEquals(256, state.value.maximumTokens) }
    }

    @Test
    fun markdownCopyIsAvailableOnlyAfterSelection() {
        compose.setContent {
            KidiTheme { MarkdownReply("A **selectable** response.", Modifier.padding(48.dp)) }
        }
        onView(withText("A selectable response.")).perform(longClick())
        val device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
        assertNotNull(device.wait(Until.findObject(By.text("Copy")), 5000))
    }

    @Test
    fun logoOpensSearchableHistoryAndScrollingRequestsAnotherPage() {
        val state = mutableStateOf(KidiUiState(history = (0 until 30).map {
            ChatSummary("chat-$it", "Conversation $it", "Preview $it", 1_000_000L + it)
        }, historyHasMore = true))
        var opened = false
        var moreRequested = false
        var selected = ""
        compose.setContent {
            KidiTheme {
                ChatWorkspace(state = state.value, snackbar = SnackbarHostState(), onNewChat = {}, onSettings = {},
                    onTextChange = {}, onSend = {}, onStop = {}, onRecord = {}, onStopRecording = {},
                    onHistoryOpen = { opened = true },
                    onHistoryQuery = { query -> state.value = state.value.copy(historyQuery = query,
                        history = listOf(ChatSummary("old-chat", "Older astronomy chat", "Found in all chats", 1)), historyHasMore = false) },
                    onOpenChat = { selected = it },
                    onMoreHistory = { moreRequested = true; state.value = state.value.copy(historyHasMore = false) })
            }
        }
        compose.onNodeWithContentDescription("Chat history").performClick()
        compose.onNodeWithText("Search chats").assertIsDisplayed()
        compose.onNodeWithTag("chat-history-list").performScrollToNode(hasText("Conversation 29"))
        compose.runOnIdle { assertEquals(true, opened); assertEquals(true, moreRequested) }
        compose.onNodeWithTag("chat-history-search").performTextReplacement("astro")
        compose.onNodeWithText("Older astronomy chat").performClick()
        compose.runOnIdle { assertEquals("astro", state.value.historyQuery); assertEquals("old-chat", selected) }
        compose.onNodeWithText("Search chats").assertIsNotDisplayed()
    }
}