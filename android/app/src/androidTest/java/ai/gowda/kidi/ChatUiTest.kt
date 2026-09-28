package ai.gowda.kidi

import androidx.compose.foundation.layout.padding
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
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
import org.junit.Rule
import org.junit.Test

class ChatUiTest {
    @get:Rule val compose = createComposeRule()

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
        compose.onNode(hasSetTextAction()).performTextReplacement("Existing draft")
        compose.onNodeWithContentDescription("Send message").assertIsNotEnabled()
        compose.onNodeWithContentDescription("Set up dictation").performClick()
        compose.runOnIdle {
            state.value = state.value.copy(loadingModel = true, loadingSpeech = true)
        }
        compose.onNode(hasSetTextAction()).performTextReplacement("Typing during model loading")
        compose.onNodeWithContentDescription("Send message").assertIsNotEnabled()
        compose.runOnIdle {
            state.value = state.value.copy(modelReady = true, loadingModel = false)
        }
        compose.onNodeWithContentDescription("Send message").assertIsEnabled().performClick()
        compose.runOnIdle { assertEquals("Typing during model loading", sent) }
        compose.runOnIdle {
            assertEquals(true, settingsOpened)
            state.value = state.value.copy(modelReady = true, speechReady = true, loadingSpeech = false, recording = true,
                composerText = "Existing draft with live speech", recordingSeconds = 4.2f)
        }
        compose.onNodeWithText("Existing draft with live speech").assertIsDisplayed()
        compose.onNodeWithContentDescription("Send message").assertIsNotEnabled()
        compose.onNodeWithContentDescription("Stop recording").assertIsEnabled().performClick()
        compose.onNodeWithText("Refining transcript").assertIsDisplayed()
        compose.onNodeWithContentDescription("Send message").assertIsNotEnabled()
        compose.runOnIdle {
            state.value = state.value.copy(transcribing = false, composerText = "Corrected transcript")
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
}