package ai.gowda.kidi

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorSpace
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

internal class ImageStore(private val context: Context) {
    private val images = File(context.filesDir, "images")
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

    fun import(uri: Uri): MessageAttachment {
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        val bitmap = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            require(info.size.width.toLong() * info.size.height in 1..64L * 1024 * 1024) { "Image is too large" }
            val factor = minOf(1.0, 2048.0 / maxOf(info.size.width, info.size.height))
            decoder.setTargetSize(maxOf(1, (info.size.width * factor).toInt()), maxOf(1, (info.size.height * factor).toInt()))
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB))
        }
        val opaque = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Canvas(opaque).apply { drawColor(Color.WHITE); drawBitmap(bitmap, 0f, 0f, null) }
        bitmap.recycle()
        val id = UUID.randomUUID().toString()
        val file = File(images, "$id.jpg")
        try {
            check(images.isDirectory || images.mkdirs()) { "Unable to create image storage" }
            FileOutputStream(file).use {
                check(opaque.compress(Bitmap.CompressFormat.JPEG, 92, it)) { "Unable to save image" }
                it.fd.sync()
            }
            return MessageAttachment(id = id, kind = AttachmentKind.IMAGE, localUri = Uri.fromFile(file).toString(),
                name = "Photo", mimeType = "image/jpeg", sizeBytes = file.length(), width = opaque.width, height = opaque.height)
        } catch (error: Exception) {
            file.delete()
            throw error
        } finally {
            opaque.recycle()
        }
    }

    fun removeDraft(attachment: MessageAttachment) {
        val uri = Uri.parse(attachment.localUri)
        if (uri.scheme != "file") return
        val file = File(uri.path ?: return).canonicalFile
        if (file.parentFile == images.canonicalFile) file.delete()
    }
}