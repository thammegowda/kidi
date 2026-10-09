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
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performScrollToNode
import androidx.compose.ui.test.performTextReplacement
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.UiController
import androidx.test.espresso.ViewAction
import androidx.test.espresso.action.ViewActions.longClick
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertSame
import org.junit.Rule
import org.junit.Test
import org.hamcrest.Matchers.containsString
import org.hamcrest.Matcher
import androidx.test.espresso.matcher.ViewMatchers.isAssignableFrom
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import android.view.View
import android.widget.TextView
import android.text.Spanned
import io.noties.markwon.ext.latex.JLatexAsyncDrawableSpan
import org.junit.Assert.assertTrue

class ChatUiTest {
    @get:Rule val compose = createComposeRule()

    @Test
    fun streamingMathKeepsCompletedEquationsStable() {
        val state = mutableStateOf(KidiUiState(modelReady = true, generating = true))
        compose.setContent {
            KidiTheme {
                ChatWorkspace(state = state.value, snackbar = SnackbarHostState(), onNewChat = {}, onSettings = {},
                    onTextChange = {}, onSend = {}, onStop = {}, onRecord = {}, onStopRecording = {})
            }
        }
        fun reply(): TextView {
            var textView: TextView? = null
            onView(withText(containsString("Math response"))).perform(object : ViewAction {
                override fun getConstraints(): Matcher<View> = isAssignableFrom(TextView::class.java)
                override fun getDescription(): String = "Capture the streamed math reply"
                override fun perform(uiController: UiController, view: View) { textView = view as TextView }
            })
            return requireNotNull(textView)
        }
        fun spans(view: TextView) = (view.text as Spanned)
            .getSpans(0, view.text.length, JLatexAsyncDrawableSpan::class.java)
        fun awaitRendering(view: TextView, count: Int) {
            compose.waitUntil(5000) {
                var rendered = false
                compose.runOnIdle { rendered = spans(view).size == count && spans(view).all { it.getDrawable().hasResult() } }
                rendered
            }
        }
        val inline = "Math response **bold** \$x^2\$ and \$x^2\$"
        val block = "$inline\n\n\$\$\n\\frac{1}{2}\n\$\$"
        for (stopped in listOf(false, true)) {
            compose.runOnIdle {
                state.value = state.value.copy(generating = true, messages = emptyList(), draft = inline)
            }
            val streamingView = reply()
            awaitRendering(streamingView, 2)
            val inlineSpans = spans(streamingView)
            assertNotSame(inlineSpans[0], inlineSpans[1])
            for (draft in listOf("$inline\n\n\$\$\n\\frac{1}{", "$inline\n\n\$\$\n\\frac{1}{2}")) {
                compose.runOnIdle { state.value = state.value.copy(draft = draft) }
                compose.runOnIdle {
                    assertEquals(2, spans(streamingView).size)
                    inlineSpans.zip(spans(streamingView)).forEach { (before, after) -> assertSame(before, after) }
                    assertTrue(streamingView.text.contains("Math response bold"))
                    assertTrue(streamingView.text.contains("\$\$"))
                }
            }
            compose.runOnIdle { state.value = state.value.copy(draft = block) }
            awaitRendering(streamingView, 3)
            val completedSpans = spans(streamingView)
            val bounds = completedSpans.map { android.graphics.Rect(it.getDrawable().bounds) }
            for (suffix in listOf("\n\nMore", " text", " while generating.", " Incomplete \\(\\frac{1}{")) {
                compose.runOnIdle { state.value = state.value.copy(draft = state.value.draft + suffix) }
                compose.runOnIdle {
                    assertEquals(3, spans(streamingView).size)
                    completedSpans.zip(spans(streamingView)).forEach { (before, after) -> assertSame(before, after) }
                    assertEquals(bounds, spans(streamingView).map { it.getDrawable().bounds })
                }
            }
            compose.runOnIdle {
                val message = ChatMessage(MessageRole.ASSISTANT, state.value.draft)
                state.value = state.value.copy(generating = false, draft = "",
                    messages = listOf(if (stopped) message.copy(status = MessageStatus.STOPPED) else message))
            }
            awaitRendering(reply(), 3)
        }
    }

    @Test
    fun mathRendersOfflineInLightAndDarkThemes() {
        val dark = mutableStateOf(false)
        val source = mutableStateOf("Inline \$x^2\$.\n\n\$\$\n\\frac{1}{2}\n\$\$")
        var textView: TextView? = null
        compose.setContent {
            MaterialTheme(colorScheme = if (dark.value) darkColorScheme() else lightColorScheme()) {
                MarkdownReply(source.value, Modifier.padding(16.dp))
            }
        }
        onView(isAssignableFrom(TextView::class.java)).perform(object : ViewAction {
            override fun getConstraints(): Matcher<View> = isAssignableFrom(TextView::class.java)
            override fun getDescription(): String = "Capture the math-rendering TextView"
            override fun perform(uiController: UiController, view: View) { textView = view as TextView }
        })
        fun spans() = (textView!!.text as Spanned).getSpans(0, textView!!.text.length, JLatexAsyncDrawableSpan::class.java)
        fun awaitRendering() {
            compose.waitUntil(5000) {
                var ready = false
                compose.runOnIdle { ready = spans().size == 2 && spans().all { it.getDrawable().hasResult() } }
                ready
            }
            compose.runOnIdle {
                spans().forEach { span ->
                    assertEquals(textView!!.currentTextColor, span.color())
                    assertTrue(span.getDrawable().getResult().bounds.width() > 0)
                    assertTrue(span.getDrawable().getResult().bounds.height() > 0)
                }
            }
        }
        awaitRendering()
        val retained = spans()
        compose.runOnIdle { source.value += "\n\nMore text after the equations." }
        compose.runOnIdle {
            assertEquals(retained.size, spans().size)
            retained.zip(spans()).forEach { (before, after) -> assertSame(before, after) }
        }
        compose.runOnIdle { dark.value = true }
        awaitRendering()
        compose.runOnIdle { source.value = "Incomplete \\(\\frac{1}{" }
        compose.runOnIdle { assertEquals(0, spans().size) }
    }

    @Test
    fun legalFooterOpensBundledPoliciesAndReturnsToChat() {
        compose.setContent {
            KidiTheme {
                ChatWorkspace(state = KidiUiState(), snackbar = SnackbarHostState(), onNewChat = {}, onSettings = {},
                    onTextChange = {}, onSend = {}, onStop = {}, onRecord = {}, onStopRecording = {})
            }
        }
        compose.onNodeWithText(AI_NOTICE).assertIsDisplayed()
        val noticeBounds = compose.onNodeWithText(AI_NOTICE).fetchSemanticsNode().boundsInRoot
        val privacyBounds = compose.onNodeWithText("Privacy").fetchSemanticsNode().boundsInRoot
        val termsBounds = compose.onNodeWithText("Terms").fetchSemanticsNode().boundsInRoot
        assertEquals(noticeBounds.center.y, privacyBounds.center.y, 1f)
        assertEquals(noticeBounds.center.y, termsBounds.center.y, 1f)
        compose.onNodeWithText("Privacy").performClick()
        compose.onNodeWithText("Privacy Policy").assertIsDisplayed()
        compose.waitUntil(5000) { compose.onAllNodesWithTag("legal-document-content").fetchSemanticsNodes().isNotEmpty() }
        onView(withText(containsString("Kidi Android Privacy Policy"))).check(matches(isDisplayed()))
        compose.onNodeWithText("Terms of Use").performClick()
        compose.waitUntil(5000) { compose.onAllNodesWithTag("legal-document-content").fetchSemanticsNodes().isNotEmpty() }
        onView(withText(containsString("Kidi is experimental software"))).check(matches(isDisplayed()))
        compose.onNodeWithContentDescription("Close legal document").performClick()
        compose.onNodeWithTag("legal-document-content").assertDoesNotExist()
        compose.onNodeWithText("Terms").performClick()
        compose.onNodeWithText("Terms of Use").assertIsDisplayed()
        compose.onNodeWithContentDescription("Close legal document").performClick()
        compose.onNodeWithTag("message-composer").assertIsDisplayed()
    }

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
        val phaseBounds = compose.onNodeWithText("Responding").fetchSemanticsNode().boundsInRoot
        val tokenBounds = compose.onNodeWithText("21 tokens").fetchSemanticsNode().boundsInRoot
        val speedBounds = compose.onNodeWithText("%.1f tok/s".format(20.0)).fetchSemanticsNode().boundsInRoot
        assertEquals(phaseBounds.center.y, tokenBounds.center.y, 1f)
        assertEquals(phaseBounds.center.y, speedBounds.center.y, 1f)
        org.junit.Assert.assertTrue(phaseBounds.right <= tokenBounds.left)
        org.junit.Assert.assertTrue(tokenBounds.right <= speedBounds.left)
        compose.onNodeWithContentDescription("Stop generation").performClick()
        compose.onNodeWithText("Stopping").assertIsDisplayed()
        compose.runOnIdle { state.value = state.value.copy(generating = false, stopping = false) }
        compose.onNodeWithTag("generation-progress").assertDoesNotExist()
    }

    @Test
    fun modelDownloadFailureIsVisibleInsideSettings() {
        val message = "Chat download failed: connection timed out. Retry to resume."
        val state = mutableStateOf(KidiUiState(error = message, modelError = message))
        compose.setContent {
            KidiTheme {
                SettingsSheet(state = state.value, onDismiss = {}, onModelId = {}, onSpeechModelId = {},
                    onThreads = {}, onTokens = {}, onInstall = {}, onCancelChat = {}, onInstallSpeech = {},
                    onCancelSpeech = {}, onDelete = {}, onDeleteSpeech = {})
            }
        }
        compose.onNodeWithTag("Chat-model-error").performScrollTo().assertIsDisplayed()
        compose.onNodeWithText(message).assertIsDisplayed()
        compose.runOnIdle { state.value = state.value.copy(error = null, speechReady = true) }
        compose.onNodeWithText(message).assertIsDisplayed()
        compose.onNodeWithText("Download chat model").performScrollTo().assertIsEnabled()
    }

    @Test
    fun verificationAndRetriesRemainVisibleAndCancellable() {
        val state = mutableStateOf(KidiUiState(loadingModel = true, progress = 1f,
            progressFile = "model.safetensors", progressPhase = ModelDownloadPhase.VERIFYING))
        var cancelled = false
        compose.setContent {
            KidiTheme {
                SettingsSheet(state = state.value, onDismiss = {}, onModelId = {}, onSpeechModelId = {},
                    onThreads = {}, onTokens = {}, onInstall = {}, onCancelChat = { cancelled = true },
                    onInstallSpeech = {}, onCancelSpeech = {}, onDelete = {}, onDeleteSpeech = {})
            }
        }
        compose.onNodeWithText("Verifying").assertIsDisplayed()
        compose.onNodeWithContentDescription("Cancel Chat download").performScrollTo().assertIsEnabled()
        compose.runOnIdle { state.value = state.value.copy(progressPhase = ModelDownloadPhase.RETRYING) }
        compose.onNodeWithText("Retrying").assertIsDisplayed()
        compose.onNodeWithContentDescription("Cancel Chat download").performClick()
        compose.runOnIdle { assertEquals(true, cancelled) }
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
    fun pairedAccessoryCanBeUnpairedFromSettings() {
        var unpaired = false
        compose.setContent {
            KidiTheme {
                SettingsSheet(
                    state = KidiUiState(pairedAccessoryName = "Kidi Lab"),
                    onDismiss = {},
                    onModelId = {},
                    onSpeechModelId = {},
                    onThreads = {},
                    onTokens = {},
                    onInstall = {},
                    onCancelChat = {},
                    onInstallSpeech = {},
                    onCancelSpeech = {},
                    onDelete = {},
                    onDeleteSpeech = {},
                    onUnpairAccessory = { unpaired = true },
                )
            }
        }

        compose.onNodeWithText("Accessory").performClick()
        compose.onNodeWithText("Kidi Lab").assertIsDisplayed()
        compose.onNodeWithTag("unpair-accessory").performClick()
        compose.runOnIdle { assertEquals(true, unpaired) }
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
        compose.onNodeWithContentDescription("Send message").assertDoesNotExist()
        compose.onNodeWithContentDescription("Dictate message").assertIsNotEnabled()
        compose.onNodeWithText("On-device AI").assertDoesNotExist()
        val stopBounds = compose.onNodeWithContentDescription("Stop recording").fetchSemanticsNode().boundsInRoot
        val micBounds = compose.onNodeWithContentDescription("Dictate message").fetchSemanticsNode().boundsInRoot
        assertTrue(stopBounds.left >= micBounds.right)
        compose.onNodeWithContentDescription("Stop recording").assertIsEnabled().performClick()
        compose.onNodeWithText("Refining transcript").assertIsDisplayed()
        compose.onNodeWithContentDescription("Stop recording").assertDoesNotExist()
        compose.onNodeWithContentDescription("Send message").assertIsNotEnabled()
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