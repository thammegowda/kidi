package ai.gowda.kidi

import android.graphics.BitmapFactory
import android.net.Uri
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@Composable
internal fun PhotoPreview(attachment: MessageAttachment, modifier: Modifier = Modifier) {
    var expanded by remember { mutableStateOf(false) }
    val result by produceState<Result<ImageBitmap>?>(null, attachment.localUri, expanded) {
        value = withContext(Dispatchers.IO) {
            runCatching {
                val file = requireNotNull(Uri.parse(attachment.localUri).path)
                val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(file, options)
                val maximum = if (expanded) 2048 else 512
                options.inSampleSize = 1
                while (maxOf(options.outWidth, options.outHeight) / options.inSampleSize > maximum) options.inSampleSize *= 2
                options.inJustDecodeBounds = false
                requireNotNull(BitmapFactory.decodeFile(file, options)).asImageBitmap()
            }
        }
    }
    Box(modifier.clickable(enabled = result?.isSuccess == true) { expanded = true }, contentAlignment = Alignment.Center) {
        val bitmap = result?.getOrNull()
        if (bitmap != null) Image(bitmap, attachment.name, Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
        else if (result == null) CircularProgressIndicator(Modifier.size(24.dp), strokeWidth = 2.dp)
        else Text("Photo unavailable", style = MaterialTheme.typography.bodySmall)
    }
    if (expanded) Dialog(onDismissRequest = { expanded = false }, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.padding(16.dp).widthIn(max = 760.dp).fillMaxWidth().fillMaxHeight(0.85f)) {
            Column {
                Row(Modifier.fillMaxWidth().padding(start = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(attachment.name, Modifier.weight(1f), style = MaterialTheme.typography.titleMedium)
                    ToolButton(Icons.Default.Close, "Close photo", { expanded = false })
                }
                result?.getOrNull()?.let { Image(it, attachment.name, Modifier.fillMaxWidth().weight(1f), contentScale = ContentScale.Fit) }
            }
        }
    }
}