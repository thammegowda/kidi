package ai.gowda.kidi

import android.content.Context
import android.os.storage.StorageManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

internal data class InstalledModel(
    val modelId: String,
    val revision: String,
    val directory: File,
)

internal data class ModelDownloadProgress(
    val file: String,
    val completedBytes: Long,
    val totalBytes: Long,
)

internal class ModelRepository(context: Context) {
    private data class RemoteFile(val name: String, val size: Long, val sha256: String?)
    private data class Revision(val modelId: String, val sha: String, val files: List<RemoteFile>)

    private val root = File(context.getExternalFilesDir(null) ?: context.filesDir, "models")
    private val storageManager = context.getSystemService(StorageManager::class.java)
    private val prepareMutex = Mutex()

    fun installed(): InstalledModel? = installed(CURRENT_FILE, CHAT_FILES + "model.yaml")

    fun installedSpeech(): InstalledModel? = installed(SPEECH_CURRENT_FILE, SPEECH_FILES)

    private fun installed(currentName: String, requiredFiles: List<String>): InstalledModel? {
        val current = File(root, currentName)
        if (!current.isFile) return null
        return runCatching {
            val value = JSONObject(current.readText())
            val modelId = value.getString("model_id")
            val revision = value.getString("revision")
            val directory = modelDirectory(modelId, revision)
            if (!File(directory, READY_FILE).isFile || requiredFiles.any { !File(directory, it).isFile }) return null
            InstalledModel(modelId, revision, directory)
        }.getOrNull()
    }

    suspend fun prepare(modelId: String, progress: (ModelDownloadProgress) -> Unit): InstalledModel =
        prepareModel(modelId, CHAT_FILES, CURRENT_FILE, ::createManifest, progress)

    suspend fun prepareSpeech(modelId: String, progress: (ModelDownloadProgress) -> Unit): InstalledModel =
        prepareModel(modelId, SPEECH_FILES, SPEECH_CURRENT_FILE, ::validateSpeechModel, progress)

    private suspend fun prepareModel(
        modelId: String,
        requiredFiles: List<String>,
        currentName: String,
        configure: (File) -> Unit,
        progress: (ModelDownloadProgress) -> Unit,
    ): InstalledModel =
        prepareMutex.withLock {
            withContext(Dispatchers.IO) {
                require(MODEL_ID.matches(modelId)) { "Enter a Hugging Face model ID such as $DEFAULT_MODEL_ID" }
                val revision = resolve(modelId, requiredFiles)
                val directory = modelDirectory(modelId, revision.sha)
                directory.mkdirs()

                val totalBytes = revision.files.sumOf(RemoteFile::size)
                val ready = readReady(directory)
                if (ready == revision.sha && revision.files.all { File(directory, it.name).length() == it.size }) {
                    progress(ModelDownloadProgress("Model ready", totalBytes, totalBytes))
                    writeCurrent(currentName, modelId, revision.sha)
                    return@withContext InstalledModel(modelId, revision.sha, directory)
                }

                val existingBytes = revision.files.sumOf { file ->
                    val target = File(directory, file.name)
                    val partial = File(directory, "${file.name}.part")
                    when {
                        target.length() == file.size -> file.size
                        partial.length() in 1..file.size -> partial.length()
                        else -> 0L
                    }
                }
                val requiredBytes = totalBytes - existingBytes + DOWNLOAD_HEADROOM_BYTES
                val storageUuid = storageManager.getUuidForPath(directory)
                val allocatableBytes = storageManager.getAllocatableBytes(storageUuid)
                require(allocatableBytes >= requiredBytes) {
                    "Not enough storage. Free at least ${formatBytes(requiredBytes - allocatableBytes)} and retry."
                }
                if (directory.usableSpace < requiredBytes) storageManager.allocateBytes(storageUuid, requiredBytes)

                var completedBytes = 0L
                for (file in revision.files) {
                    coroutineContext.ensureActive()
                    download(modelId, revision.sha, directory, file) { fileBytes ->
                        progress(ModelDownloadProgress(file.name, completedBytes + fileBytes, totalBytes))
                    }
                    completedBytes += file.size
                    progress(ModelDownloadProgress(file.name, completedBytes, totalBytes))
                }
                configure(directory)
                writeAtomically(File(directory, READY_FILE), JSONObject().put("revision", revision.sha).toString())
                writeCurrent(currentName, modelId, revision.sha)
                InstalledModel(modelId, revision.sha, directory)
            }
        }

    fun removeInstalled() {
        removeInstalled(CURRENT_FILE, CHAT_FILES + "model.yaml")
    }

    fun removeInstalledSpeech() {
        removeInstalled(SPEECH_CURRENT_FILE, SPEECH_FILES)
    }

    private fun removeInstalled(currentName: String, requiredFiles: List<String>) {
        installed(currentName, requiredFiles)?.directory?.deleteRecursively()
        File(root, currentName).delete()
    }

    private fun resolve(modelId: String, requiredFiles: List<String>): Revision {
        val path = modelId.split('/').joinToString("/") { java.net.URLEncoder.encode(it, Charsets.UTF_8.name()) }
        val metadata = JSONObject(read(URL("https://huggingface.co/api/models/$path/revision/main?blobs=true")))
        val revision = metadata.getString("sha")
        require(REVISION.matches(revision)) { "Hugging Face returned an invalid model revision" }
        val siblings = metadata.getJSONArray("siblings")
        val available = buildMap {
            for (index in 0 until siblings.length()) {
                val item = siblings.getJSONObject(index)
                put(item.getString("rfilename"), item)
            }
        }
        val files = requiredFiles.map { name ->
            val item = requireNotNull(available[name]) { "Model is missing $name" }
            val size = item.getLong("size")
            require(size > 0) { "Model file $name is empty" }
            RemoteFile(name, size, item.optJSONObject("lfs")?.optString("sha256")?.takeIf(SHA256::matches))
        }
        return Revision(modelId, revision, files)
    }

    private suspend fun download(
        modelId: String,
        revision: String,
        directory: File,
        metadata: RemoteFile,
        progress: (Long) -> Unit,
    ) {
        val target = File(directory, metadata.name)
        if (target.length() == metadata.size && verify(target, metadata.sha256)) return
        if (target.exists()) target.delete()

        val partial = File(directory, "${metadata.name}.part")
        if (partial.length() > metadata.size) partial.delete()
        var offset = partial.length()
        progress(offset)
        var connection = connect(fileUrl(modelId, revision, metadata.name), offset)
        if (offset > 0 && connection.responseCode != HttpURLConnection.HTTP_PARTIAL) {
            connection.disconnect()
            partial.delete()
            offset = 0
            connection = connect(fileUrl(modelId, revision, metadata.name), 0)
        }
        try {
            require(connection.responseCode == if (offset > 0) HttpURLConnection.HTTP_PARTIAL else HttpURLConnection.HTTP_OK) {
                "Download ${metadata.name} failed with HTTP ${connection.responseCode}"
            }
            if (offset > 0) {
                val start = CONTENT_RANGE.find(connection.getHeaderField("Content-Range").orEmpty())
                    ?.groupValues?.get(1)?.toLongOrNull()
                require(start == offset) { "Invalid resume response for ${metadata.name}" }
            }
            connection.inputStream.use { input ->
                FileOutputStream(partial, offset > 0).use { output ->
                    val buffer = ByteArray(BUFFER_BYTES)
                    var downloaded = offset
                    while (true) {
                        coroutineContext.ensureActive()
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        downloaded += count
                        require(downloaded <= metadata.size) { "Download exceeded expected size for ${metadata.name}" }
                        progress(downloaded)
                    }
                    output.fd.sync()
                }
            }
        } finally {
            connection.disconnect()
        }
        require(partial.length() == metadata.size) { "Download was truncated for ${metadata.name}" }
        require(verify(partial, metadata.sha256)) { "Checksum mismatch for ${metadata.name}" }
        move(partial, target)
    }

    private fun createManifest(directory: File) {
        val config = JSONObject(File(directory, "config.json").readText())
        val model = JSONObject(config.getJSONObject("text_config").toString())
        require(config.optString("model_type") == "gemma4" && !model.optBoolean("enable_moe_block")) {
            "Expected a dense Gemma 4 checkpoint"
        }
        val quantization = config.optJSONObject("quantization_config")
        require(quantization == null || quantization.optString("quant_method") == "gemma") {
            "Only native Gemma QAT or original floating-point checkpoints are supported"
        }
        val tokenizerConfig = JSONObject(File(directory, "tokenizer_config.json").readText())
        require(tokenizerConfig.optString("chat_template").isNotBlank() || File(directory, "chat_template.jinja").isFile) {
            "Model has no chat template"
        }
        model.put("type", "gemma4_text")
        if (quantization != null) model.put("quantization_config", quantization)
        val manifest = JSONObject()
            .put("format_version", 1)
            .put("weights_file", "model.safetensors")
            .put("tokenizer_file", "tokenizer.json")
            .put("model", model)
            .put("decode", JSONObject().put("maximum_new_tokens", 8192).put("context_size", 16384))
        writeAtomically(File(directory, "model.yaml"), manifest.toString())
    }

    private fun validateSpeechModel(directory: File) {
        val config = JSONObject(File(directory, "config.json").readText())
        val architectures = config.optJSONArray("architectures") ?: JSONArray()
        val signature = listOf(
            "d_model",
            "encoder_layers",
            "decoder_layers",
            "encoder_attention_heads",
            "decoder_attention_heads",
            "encoder_ffn_dim",
            "decoder_ffn_dim",
        ).map(config::optInt)
        require(
            config.optString("model_type") == "whisper" &&
                (0 until architectures.length()).any {
                    architectures.optString(it) == "WhisperForConditionalGeneration"
                } &&
                signature in SUPPORTED_WHISPER_SIGNATURES &&
                config.optInt("num_mel_bins") == 80 &&
                config.optInt("vocab_size") == 51865,
        ) { "Expected a supported Whisper Tiny, Base, or Small checkpoint" }
    }

    private fun connect(source: URL, offset: Long): HttpURLConnection {
        var url = source
        repeat(MAX_REDIRECTS) {
            val connection = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = CONNECT_TIMEOUT_MS
                readTimeout = READ_TIMEOUT_MS
                instanceFollowRedirects = false
                setRequestProperty("Accept-Encoding", "identity")
                if (offset > 0) setRequestProperty("Range", "bytes=$offset-")
            }
            val code = connection.responseCode
            if (code !in 300..399) return connection
            val location = connection.getHeaderField("Location")
                ?: throw IllegalStateException("Model download redirect has no location")
            url = URL(url, location)
            connection.disconnect()
        }
        throw IllegalStateException("Too many model download redirects")
    }

    private fun read(url: URL): String {
        val connection = connect(url, 0)
        try {
            require(connection.responseCode == HttpURLConnection.HTTP_OK) {
                "Hugging Face lookup failed with HTTP ${connection.responseCode}"
            }
            return connection.inputStream.bufferedReader().use { it.readText() }
        } finally {
            connection.disconnect()
        }
    }

    private fun verify(file: File, expected: String?): Boolean {
        if (expected == null) return true
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(BUFFER_BYTES)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) } == expected
    }

    private fun readReady(directory: File): String? = runCatching {
        JSONObject(File(directory, READY_FILE).readText()).getString("revision")
    }.getOrNull()

    private fun writeCurrent(currentName: String, modelId: String, revision: String) {
        root.mkdirs()
        writeAtomically(
            File(root, currentName),
            JSONObject().put("model_id", modelId).put("revision", revision).toString(),
        )
    }

    private fun writeAtomically(target: File, content: String) {
        target.parentFile?.mkdirs()
        val temporary = File(target.parentFile, ".${target.name}.tmp")
        temporary.writeText(content)
        move(temporary, target)
    }

    private fun move(source: File, target: File) {
        try {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private fun modelDirectory(modelId: String, revision: String) =
        File(root, "${modelId.replace('/', '-')}/$revision")

    private fun fileUrl(modelId: String, revision: String, name: String) =
        URL("https://huggingface.co/$modelId/resolve/$revision/$name")

    private fun formatBytes(bytes: Long) = "%.1f GB".format(bytes.coerceAtLeast(0) / 1_000_000_000.0)

    private companion object {
        val MODEL_ID = Regex("[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
        val REVISION = Regex("[a-f0-9]{40}")
        val SHA256 = Regex("[a-f0-9]{64}")
        val CONTENT_RANGE = Regex("bytes (\\d+)-\\d+/\\d+")
        val CHAT_FILES = listOf(
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
        val SPEECH_FILES = listOf(
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "preprocessor_config.json",
            "generation_config.json",
        )
        val SUPPORTED_WHISPER_SIGNATURES = setOf(
            listOf(384, 4, 4, 6, 6, 1536, 1536),
            listOf(512, 6, 6, 8, 8, 2048, 2048),
            listOf(768, 12, 12, 12, 12, 3072, 3072),
        )
        const val DEFAULT_MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
        const val READY_FILE = "ready.json"
        const val CURRENT_FILE = "current.json"
        const val SPEECH_CURRENT_FILE = "speech-current.json"
        const val BUFFER_BYTES = 1024 * 1024
        const val DOWNLOAD_HEADROOM_BYTES = 256L * 1024 * 1024
        const val CONNECT_TIMEOUT_MS = 30_000
        const val READ_TIMEOUT_MS = 120_000
        const val MAX_REDIRECTS = 8
    }
}