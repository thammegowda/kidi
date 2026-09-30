package ai.gowda.kidi

import android.app.ActivityManager
import android.content.Intent
import android.os.Bundle
import android.os.Process
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import java.util.concurrent.Executors
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

private data class BenchmarkResult(
    val accelerator: String,
    val backend: String = "",
    val loadMs: Double = 0.0,
    val promptTokens: Int = 0,
    val prefillMs: Double = 0.0,
    val firstTokenMs: Double = 0.0,
    val decodeTokens: Int = 0,
    val decodeMs: Double = 0.0,
    val processMemoryMiB: Double = 0.0,
    val availableMemoryMiB: Double = 0.0,
    val totalMemoryMiB: Double = 0.0,
    val error: String? = null,
) {
    val prefillTps: Double get() = if (prefillMs > 0) promptTokens * 1000.0 / prefillMs else 0.0
    val decodeTps: Double get() = if (decodeMs > 0) decodeTokens * 1000.0 / decodeMs else 0.0
}

private data class BenchmarkState(
    val running: Boolean = false,
    val current: String = "",
    val results: List<BenchmarkResult> = emptyList(),
    val error: String? = null,
)

class BenchmarkActivity : ComponentActivity() {
    private val runtimeExecutor = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val state = MutableStateFlow(BenchmarkState())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            KidiTheme {
                val benchmark by state.collectAsState()
                BenchmarkScreen(benchmark, ::runBenchmark, ::returnToChat)
            }
        }
        if (intent.getBooleanExtra("autorun", false)) runBenchmark()
    }

    override fun onDestroy() {
        NativeRuntime.unload()
        NativeRuntime.unloadAsr()
        runtimeExecutor.close()
        super.onDestroy()
    }

    private fun runBenchmark() {
        if (state.value.running) return
        lifecycleScope.launch {
            state.value = BenchmarkState(running = true, current = "Preparing")
            val outcome = runCatching { withContext(runtimeExecutor) { benchmark() } }
            outcome.onFailure { error ->
                state.update { it.copy(running = false, current = "", error = error.message ?: "Benchmark failed") }
            }
        }
    }

    private fun benchmark() {
        NativeRuntime.unload()
        NativeRuntime.unloadAsr()
        checked(NativeRuntime.configure(4))
        checked(NativeRuntime.setDiagnosticLogging(true))
        try {
            runAcceleratorBenchmarks()
        } finally {
            checked(NativeRuntime.setDiagnosticLogging(false))
        }
    }

    private fun runAcceleratorBenchmarks() {
        checked(NativeRuntime.setDataDirectory(filesDir.absolutePath, applicationInfo.nativeLibraryDir))
        val model = ModelRepository(this).installed() ?: error("Install the chat model before benchmarking")
        val paragraph = "Rivers move water sediment and nutrients, connect wetlands, support wildlife and " +
            "communities, and respond to dams restoration forests floods seasons and changing climate. "
        val warmup = messages("Warm up this runtime. " + buildString { repeat(7) { append(paragraph) } })
        val measured = messages("Write a long numbered analysis. " + buildString { repeat(7) { append(paragraph) } })
        val results = mutableListOf<BenchmarkResult>()
        for (accelerator in listOf("cpu", "gpu", "npu")) {
            state.update { it.copy(current = "Running ${accelerator.uppercase()}", results = results.toList()) }
            val result = runCatching {
                val loaded = checked(NativeRuntime.load(model.directory.absolutePath, accelerator))
                try {
                    generate(warmup, 16)
                    val completed = generate(measured, 64)
                    val memory = memorySnapshot()
                    BenchmarkResult(
                        accelerator = accelerator,
                        backend = loaded.getString("backend"),
                        loadMs = loaded.getDouble("load_ms"),
                        promptTokens = completed.getInt("prompt_tokens"),
                        prefillMs = completed.getDouble("prefill_ms"),
                        firstTokenMs = completed.getDouble("first_token_ms"),
                        decodeTokens = completed.getInt("decode_tokens"),
                        decodeMs = completed.getDouble("decode_ms"),
                        processMemoryMiB = memory.first,
                        availableMemoryMiB = memory.second,
                        totalMemoryMiB = memory.third,
                    )
                } finally {
                    NativeRuntime.unload()
                }
            }.getOrElse { error ->
                BenchmarkResult(accelerator = accelerator, error = error.message ?: "Unavailable")
            }
            results += result
            Log.i("KidiBenchmark", "accelerator=${result.accelerator} backend=${result.backend} " +
                "load_ms=${result.loadMs} prompt_tokens=${result.promptTokens} prefill_ms=${result.prefillMs} " +
                "prefill_tps=${result.prefillTps} first_token_ms=${result.firstTokenMs} " +
                "decode_tokens=${result.decodeTokens} decode_ms=${result.decodeMs} decode_tps=${result.decodeTps} " +
                "process_memory_mib=${result.processMemoryMiB} available_memory_mib=${result.availableMemoryMiB} " +
                "total_memory_mib=${result.totalMemoryMiB} " +
                "error=${result.error.orEmpty()}")
            state.update { it.copy(results = results.toList()) }
        }
        state.value = BenchmarkState(results = results)
    }

    private fun checked(source: String): JSONObject {
        val result = JSONObject(source)
        if (result.has("error")) error(result.getString("error"))
        return result
    }

    private fun memorySnapshot(): Triple<Double, Double, Double> {
        val manager = getSystemService(ActivityManager::class.java)
        val process = manager.getProcessMemoryInfo(intArrayOf(Process.myPid())).single()
        val system = ActivityManager.MemoryInfo().also(manager::getMemoryInfo)
        val mib = 1024.0 * 1024.0
        return Triple(process.totalPss / 1024.0, system.availMem / mib, system.totalMem / mib)
    }

    private fun messages(prompt: String) =
        JSONArray().put(JSONObject().put("role", "user").put("content", prompt))

    private fun generate(messages: JSONArray, maximumTokens: Int): JSONObject {
        checked(NativeRuntime.enqueue(messages.toString(), maximumTokens))
        repeat(1024) {
            val step = checked(NativeRuntime.step())
            val events = step.getJSONArray("events")
            for (index in 0 until events.length()) {
                val event = events.getJSONObject(index)
                if (event.has("completed")) return event.getJSONObject("completed")
            }
            if (step.getInt("pending") == 0) error("Generation ended without completion metrics")
        }
        error("Generation did not finish")
    }

    private fun returnToChat() {
        NativeRuntime.unload()
        NativeRuntime.unloadAsr()
        startActivity(Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK or Intent.FLAG_ACTIVITY_NEW_TASK)
        })
        finish()
    }
}

@Composable
private fun BenchmarkScreen(state: BenchmarkState, onRun: () -> Unit, onReturn: () -> Unit) {
    Scaffold { padding ->
        LazyColumn(
            Modifier.fillMaxSize().padding(padding).padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            item {
                Text("Accelerator benchmark", style = MaterialTheme.typography.headlineMedium)
                Spacer(Modifier.height(8.dp))
                Text("Runs identical warmup and measured chat requests through the real app serving path.")
            }
            if (state.running) item {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    CircularProgressIndicator()
                    Text(state.current)
                }
            }
            state.results.forEach { result ->
                item { BenchmarkResultCard(result, state.results.firstOrNull { it.accelerator == "cpu" }) }
            }
            state.error?.let { error ->
                item { Text(error, color = MaterialTheme.colorScheme.error) }
            }
            item {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = onRun, enabled = !state.running, modifier = Modifier.fillMaxWidth()) {
                        Text(if (state.results.isEmpty()) "Start benchmark" else "Run again")
                    }
                    OutlinedButton(onClick = onReturn, enabled = !state.running, modifier = Modifier.fillMaxWidth()) {
                        Text("Return to chat")
                    }
                    Text("The chat and speech models reload when you return.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}

@Composable
private fun BenchmarkResultCard(result: BenchmarkResult, cpu: BenchmarkResult?) {
    Surface(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.medium, tonalElevation = 2.dp) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(result.accelerator.uppercase(), style = MaterialTheme.typography.titleLarge)
            if (result.error != null) {
                Text(result.error, color = MaterialTheme.colorScheme.error)
            } else {
                Text(result.backend, color = MaterialTheme.colorScheme.onSurfaceVariant)
                HorizontalDivider()
                Metric("Model load", "${"%.0f".format(result.loadMs)} ms")
                Metric("Prompt", "${result.promptTokens} tokens")
                Metric("Prefill rate", "${"%.1f".format(result.prefillTps)} tok/s")
                Metric("Prefill time", "${result.prefillMs.toInt()} ms")
                Metric("Time to first token", "${"%.0f".format(result.firstTokenMs)} ms")
                Metric("Incremental decode", "${"%.1f".format(result.decodeTps)} tok/s")
                Metric("Measured decode", "${result.decodeTokens} tokens · ${result.decodeMs.toInt()} ms")
                Metric("App memory", "${"%.0f".format(result.processMemoryMiB)} MiB PSS")
                Metric("Memory headroom", "${"%.0f".format(result.availableMemoryMiB)} / " +
                    "${"%.0f".format(result.totalMemoryMiB)} MiB")
                if (cpu != null && result.accelerator != "cpu" && cpu.decodeTps > 0)
                    Text("${"%.2f".format(result.decodeTps / cpu.decodeTps)}× CPU decode",
                        color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.titleMedium)
            }
        }
    }
}

@Composable
private fun Metric(label: String, value: String) {
    Row(Modifier.fillMaxWidth()) {
        Text(label, Modifier.weight(1f))
        Text(value, fontFamily = FontFamily.Monospace)
    }
}
