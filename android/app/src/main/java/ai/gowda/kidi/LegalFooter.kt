package ai.gowda.kidi

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

internal const val AI_NOTICE = "Experimental AI. Verify answers."

private enum class LegalDocument(val title: String, val filename: String) {
    PRIVACY("Privacy Policy", "PRIVACY.md"),
    TERMS("Terms of Use", "TERMS.md"),
}

@Composable
internal fun LegalFooter(modifier: Modifier = Modifier) {
    var selected by rememberSaveable { mutableStateOf<String?>(null) }
    val keyboard = LocalSoftwareKeyboardController.current
    Row(modifier.fillMaxWidth().heightIn(min = 24.dp).testTag("legal-footer"),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(AI_NOTICE, Modifier.weight(1f), style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
        Text("Privacy", Modifier.clickable(role = Role.Button) {
            keyboard?.hide(); selected = LegalDocument.PRIVACY.name
        }.padding(vertical = 6.dp), style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.primary, maxLines = 1)
        Text("Terms", Modifier.clickable(role = Role.Button) {
            keyboard?.hide(); selected = LegalDocument.TERMS.name
        }.padding(vertical = 6.dp), style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.primary, maxLines = 1)
    }
    LegalDocument.entries.find { it.name == selected }?.let { document ->
        key(document) {
            LegalDocumentDialog(document, onNavigate = { selected = it.name }, onDismiss = { selected = null })
        }
    }
}

@Composable
private fun LegalDocumentDialog(document: LegalDocument, onNavigate: (LegalDocument) -> Unit, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val text by produceState<String?>(null, context, document) {
        value = withContext(Dispatchers.IO) {
            try {
                context.assets.open("legal/${document.filename}").bufferedReader().use { it.readText() }
            } catch (_: IOException) {
                "Unable to open this document. Please try again after updating Kidi."
            }
        }
    }
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.padding(16.dp).widthIn(max = 600.dp).fillMaxWidth().fillMaxHeight(0.9f),
            shape = RoundedCornerShape(8.dp), color = MaterialTheme.colorScheme.surface) {
            Column {
                Row(Modifier.fillMaxWidth().padding(start = 24.dp, end = 8.dp, top = 8.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Text(document.title, Modifier.weight(1f), style = MaterialTheme.typography.titleLarge)
                    ToolButton(Icons.Default.Close, "Close legal document", onDismiss)
                }
                HorizontalDivider()
                Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(24.dp)) {
                    text?.let { content ->
                        MarkdownReply(content, Modifier.fillMaxWidth().testTag("legal-document-content"), onLink = { link ->
                            val destination = LegalDocument.entries.find { it.filename == link.removePrefix("./") }
                            if (destination != null) onNavigate(destination)
                            destination != null
                        })
                    } ?: CircularProgressIndicator()
                }
                HorizontalDivider()
                val other = if (document == LegalDocument.PRIVACY) LegalDocument.TERMS else LegalDocument.PRIVACY
                TextButton(onClick = { onNavigate(other) }, modifier = Modifier.align(Alignment.End).padding(8.dp)) {
                    Text(other.title)
                }
            }
        }
    }
}