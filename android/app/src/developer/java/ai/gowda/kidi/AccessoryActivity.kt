package ai.gowda.kidi

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

private data class AccessoryState(
    val invitation: PairingInvitation? = null,
    val profiles: List<AccessoryProfile> = emptyList(),
    val busy: Boolean = false,
    val status: String = "",
    val error: String = "",
)

class AccessoryActivity : ComponentActivity() {
    private lateinit var profiles: AccessoryProfileStore
    private lateinit var pairer: AccessoryBlePairer
    private lateinit var network: AccessoryNetwork
    private val state = MutableStateFlow(AccessoryState())
    private var permissionAction: (() -> Unit)? = null
    private val permissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
        val action = permissionAction
        permissionAction = null
        if (result.values.all { it }) action?.invoke()
        else state.update { it.copy(error = "Required accessory permission was denied") }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        profiles = AccessoryProfileStore(this)
        pairer = AccessoryBlePairer(this, profiles)
        network = AccessoryNetwork(this)
        refreshProfiles()
        acceptIntent(intent)
        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                val current by state.collectAsState()
                Surface(Modifier.fillMaxSize()) {
                    AccessoryScreen(
                        state = current,
                        onPair = ::requestPair,
                        onTest = ::requestTest,
                        onUnpair = ::requestUnpair,
                    )
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        acceptIntent(intent)
    }

    private fun acceptIntent(intent: Intent?) {
        if (intent?.action != Intent.ACTION_VIEW || intent.data?.scheme != "kidi") return
        runCatching {
            AccessoryProtocol.parsePairingUri(intent.data.toString(), System.currentTimeMillis() / 1000)
        }.onSuccess { invitation ->
            state.update { it.copy(invitation = invitation, status = "Invitation verified", error = "") }
        }.onFailure { error ->
            state.update { it.copy(invitation = null, error = error.message ?: "Invalid pairing invitation") }
        }
    }

    private fun refreshProfiles() {
        runCatching(profiles::profiles)
            .onSuccess { values -> state.update { it.copy(profiles = values) } }
            .onFailure { error ->
                state.update { it.copy(error = error.message ?: "Unable to read accessory profiles") }
            }
    }

    private fun requestPair() = withPermissions(pairingPermissions(), ::pair)

    @SuppressLint("MissingPermission")
    private fun pair() {
        val invitation = state.value.invitation ?: return
        if (state.value.busy) return
        state.update { it.copy(busy = true, status = "Pairing with ${invitation.deviceName}", error = "") }
        lifecycleScope.launch {
            runCatching {
                pairer.pair(invitation, "${Build.MANUFACTURER} ${Build.MODEL}".trim().take(64))
            }.onSuccess { profile ->
                refreshProfiles()
                state.update {
                    it.copy(invitation = null, busy = false, status = "Paired with ${profile.name}", error = "")
                }
            }.onFailure { error ->
                state.update { it.copy(busy = false, error = error.message ?: "Accessory pairing failed") }
            }
        }
    }

    private fun requestTest(profile: AccessoryProfile) =
        withPermissions(networkPermissions()) { test(profile) }

    @SuppressLint("MissingPermission")
    private fun test(profile: AccessoryProfile) {
        if (state.value.busy) return
        state.update { it.copy(busy = true, status = "Connecting to ${profile.name}", error = "") }
        lifecycleScope.launch {
            runCatching {
                network.connect(profile).use { lease ->
                    AccessoryClient(profile, lease.network.socketFactory).capturePhoto("diagnostic").size
                }
            }.onSuccess { bytes ->
                state.update {
                    it.copy(busy = false, status = "${profile.name} returned a ${bytes / 1024} KiB JPEG", error = "")
                }
            }.onFailure { error ->
                state.update { it.copy(busy = false, error = error.message ?: "Accessory diagnostic failed") }
            }
        }
    }

    private fun requestUnpair(profile: AccessoryProfile) =
        withPermissions(networkPermissions()) { unpair(profile) }

    @SuppressLint("MissingPermission")
    private fun unpair(profile: AccessoryProfile) {
        if (state.value.busy) return
        state.update { it.copy(busy = true, status = "Removing ${profile.name}", error = "") }
        lifecycleScope.launch {
            runCatching {
                network.connect(profile).use { lease ->
                    AccessoryClient(profile, lease.network.socketFactory).unpair()
                }
                profiles.remove(profile.deviceId)
            }.onSuccess {
                refreshProfiles()
                state.update { it.copy(busy = false, status = "Accessory unpaired", error = "") }
            }.onFailure { error ->
                state.update { it.copy(busy = false, error = error.message ?: "Unable to unpair accessory") }
            }
        }
    }

    private fun withPermissions(required: Array<String>, action: () -> Unit) {
        val missing = required.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) action()
        else {
            permissionAction = action
            permissions.launch(missing.toTypedArray())
        }
    }

    private fun pairingPermissions() = when {
        Build.VERSION.SDK_INT >= 31 ->
            arrayOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT)
        else -> arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
    }

    private fun networkPermissions() = when {
        Build.VERSION.SDK_INT >= 33 -> arrayOf(Manifest.permission.NEARBY_WIFI_DEVICES)
        else -> arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
    }
}

@Composable
private fun AccessoryScreen(
    state: AccessoryState,
    onPair: () -> Unit,
    onTest: (AccessoryProfile) -> Unit,
    onUnpair: (AccessoryProfile) -> Unit,
) {
    Scaffold { padding ->
        Column(
            Modifier
                .padding(padding)
                .padding(20.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text("Kidi accessory tools", style = MaterialTheme.typography.headlineMedium)
            Text("Developer-only BLE pairing and local Wi-Fi diagnostics.")
            state.invitation?.let { invitation ->
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text("Pair ${invitation.deviceName}", style = MaterialTheme.typography.titleMedium)
                        Text("Device ${invitation.deviceId}")
                        Button(onClick = onPair, enabled = !state.busy) { Text("Pair accessory") }
                    }
                }
            } ?: Text("Open a kidi://pair/v1 invitation to pair another accessory.")
            HorizontalDivider()
            Text("Saved accessories", style = MaterialTheme.typography.titleLarge)
            if (state.profiles.isEmpty()) Text("No accessories paired.")
            for (profile in state.profiles) {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(profile.name, style = MaterialTheme.typography.titleMedium)
                        Text("${profile.deviceId} · ${profile.host}")
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = { onTest(profile) }, enabled = !state.busy) {
                                Text("Test camera")
                            }
                            OutlinedButton(onClick = { onUnpair(profile) }, enabled = !state.busy) {
                                Text("Unpair")
                            }
                        }
                    }
                }
            }
            if (state.status.isNotBlank()) Text(state.status, color = MaterialTheme.colorScheme.primary)
            if (state.error.isNotBlank()) Text(state.error, color = MaterialTheme.colorScheme.error)
        }
    }
}
