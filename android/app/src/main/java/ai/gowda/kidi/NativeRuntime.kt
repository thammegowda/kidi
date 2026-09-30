package ai.gowda.kidi

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
    external fun load(directory: String, accelerator: String = "auto"): String
    external fun enqueue(messagesJson: String, maximumTokens: Int): String
    external fun step(): String
    external fun cancel(requestId: Long): String
    external fun loadAsr(directory: String, int8: Boolean = false, accelerator: String = "auto"): String
    external fun transcribe(samples: FloatArray, language: String, maximumTokens: Int, listener: PartialListener? = null): String
    external fun unload()
    external fun unloadAsr()

}