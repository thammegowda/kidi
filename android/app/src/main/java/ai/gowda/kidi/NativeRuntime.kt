package ai.gowda.kidi

internal const val DEFAULT_ACCELERATOR = "auto"
internal val ACCELERATOR_LABELS =
    mapOf(DEFAULT_ACCELERATOR to "Auto", "cpu" to "CPU", "gpu" to "GPU (experimental)", "npu" to "NPU")

internal object NativeRuntime {
    fun interface PartialListener {
        fun onPartial(payload: String)
    }

    init {
        System.loadLibrary("kidi_android")
    }

    external fun configure(threads: Int): String
    external fun deviceInfo(): String
    external fun setDiagnosticLogging(enabled: Boolean): String
    /** Points native code at packaged/staged NPU libraries and the persistent compiled-graph cache. */
    external fun setDataDirectory(directory: String, dspDirectory: String = ""): String
    /** Loads the chat model; [accelerator] is auto, cpu, gpu, or npu. */
    external fun load(directory: String, accelerator: String = DEFAULT_ACCELERATOR): String
    external fun enqueue(messagesJson: String, maximumTokens: Int): String
    external fun step(): String
    external fun cancel(requestId: Long): String
    external fun loadAsr(
        directory: String,
        int8: Boolean = false,
        accelerator: String = DEFAULT_ACCELERATOR,
    ): String
    external fun transcribe(samples: FloatArray, language: String, maximumTokens: Int, listener: PartialListener? = null): String
    external fun unload()
    external fun unloadAsr()

}