package ai.gowda.kidi

import android.content.Context
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File

internal class CameraCaptureStore(private val context: Context) {
    private val captures = File(context.cacheDir, "camera")

    fun captureUri(): Uri {
        check(captures.isDirectory || captures.mkdirs()) { "Unable to create camera storage" }
        val file = File.createTempFile("capture-", ".jpg", captures)
        return FileProvider.getUriForFile(context, "${context.packageName}.images", file)
    }

    fun removeCapture(uri: Uri) {
        if (uri.authority != "${context.packageName}.images") return
        val file = File(captures, uri.lastPathSegment ?: return).canonicalFile
        if (file.parentFile == captures.canonicalFile) file.delete()
    }
}
