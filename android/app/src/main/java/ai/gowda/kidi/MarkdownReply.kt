package ai.gowda.kidi

import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.util.TypedValue
import android.widget.TextView
import android.widget.Toast
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.res.ResourcesCompat
import io.noties.markwon.AbstractMarkwonPlugin
import io.noties.markwon.Markwon
import io.noties.markwon.MarkwonConfiguration
import io.noties.markwon.core.MarkwonTheme

@Composable
internal fun MarkdownReply(text: String, modifier: Modifier = Modifier, onLink: ((String) -> Boolean)? = null) {
    val context = LocalContext.current
    val handleLink by rememberUpdatedState(onLink)
    val colors = MaterialTheme.colorScheme
    val foreground = colors.onSurface.toArgb()
    val codeBackground = colors.surfaceContainer.toArgb()
    val linkColor = colors.primary.toArgb()
    val textSize = with(LocalDensity.current) { MaterialTheme.typography.bodyLarge.fontSize.toPx() }
    val markwon = remember(context, codeBackground, linkColor) {
        Markwon.builder(context).bufferType(TextView.BufferType.SPANNABLE).usePlugin(object : AbstractMarkwonPlugin() {
            override fun configureTheme(builder: MarkwonTheme.Builder) {
                builder.codeBackgroundColor(codeBackground).codeBlockBackgroundColor(codeBackground)
                    .linkColor(linkColor).headingBreakHeight(0)
                    .headingTextSizeMultipliers(floatArrayOf(1.4f, 1.25f, 1.15f, 1.1f, 1f, 1f))
            }

            override fun configureConfiguration(builder: MarkwonConfiguration.Builder) {
                builder.linkResolver { view, link ->
                    if (handleLink?.invoke(link) == true) return@linkResolver
                    val uri = Uri.parse(link)
                    if (uri.scheme in listOf("https", "http", "mailto")) {
                        try {
                            view.context.startActivity(Intent(Intent.ACTION_VIEW, uri))
                        } catch (_: ActivityNotFoundException) {
                            Toast.makeText(view.context, "No app available to open this link", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
        }).build()
    }
    AndroidView(
        modifier = modifier,
        factory = {
            TextView(it).apply {
                typeface = ResourcesCompat.getFont(it, R.font.lato_regular)
                includeFontPadding = false
                setTextIsSelectable(true)
                setLineSpacing(0f, 1.2f)
            }
        },
        update = { view ->
            view.setTextColor(foreground)
            view.setTextSize(TypedValue.COMPLEX_UNIT_PX, textSize)
            val version = text to markwon
            if (view.tag != version) {
                markwon.setMarkdown(view, text)
                view.setTextIsSelectable(true)
                view.isLongClickable = true
                view.tag = version
            }
        },
    )
}