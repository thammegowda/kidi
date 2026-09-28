package ai.gowda.kidi

internal object NativeRuntime {
    fun interface PartialListener {
        fun onPartial(payload: String)
    }

    init {
        System.loadLibrary("kidi_android")
    }

    external fun configure(threads: Int): String
    external fun load(directory: String): String
    external fun enqueue(messagesJson: String, maximumTokens: Int): String
    external fun step(): String
    external fun cancel(requestId: Long): String
    external fun loadAsr(directory: String, int8: Boolean = false): String
    external fun transcribe(samples: FloatArray, language: String, maximumTokens: Int, listener: PartialListener? = null): String
    external fun unload()
    external fun unloadAsr()

}