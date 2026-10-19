package ai.gowda.kidi

import android.Manifest
import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.wifi.WifiManager
import android.net.wifi.WifiNetworkSpecifier
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeout
import java.io.Closeable
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

internal class AccessoryNetwork(private val context: Context) {
    @RequiresPermission(allOf = [Manifest.permission.CHANGE_WIFI_STATE, Manifest.permission.ACCESS_WIFI_STATE])
    suspend fun connect(profile: AccessoryProfile): Lease = withTimeout(CONNECTION_TIMEOUT_MS) {
        val connectivity = context.getSystemService(ConnectivityManager::class.java)
        val wifi = context.getSystemService(WifiManager::class.java)
        val active = connectivity.getNetworkCapabilities(connectivity.activeNetwork)
        val primaryWifi = active?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true &&
            active.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
        if (primaryWifi && (
                android.os.Build.VERSION.SDK_INT < 31 ||
                    !wifi.isStaConcurrencyForLocalOnlyConnectionsSupported
                )
        ) {
            throw UnsupportedOperationException(
                "This phone cannot keep its current Wi-Fi while connecting to the accessory",
            )
        }
        val specifier = WifiNetworkSpecifier.Builder()
            .setSsid(profile.softApSsid)
            .setWpa2Passphrase(profile.softApPassword)
            .build()
        val request = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
            .removeCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .setNetworkSpecifier(specifier)
            .build()
        suspendCancellableCoroutine { continuation ->
            val callback = object : ConnectivityManager.NetworkCallback() {
                override fun onAvailable(network: Network) {
                    if (continuation.isActive) continuation.resume(Lease(connectivity, this, network))
                    else runCatching { connectivity.unregisterNetworkCallback(this) }
                }

                override fun onUnavailable() {
                    if (continuation.isActive) {
                        continuation.resumeWithException(
                            IllegalStateException("Accessory Wi-Fi request was denied or unavailable"),
                        )
                    }
                }
            }
            continuation.invokeOnCancellation {
                runCatching { connectivity.unregisterNetworkCallback(callback) }
            }
            connectivity.requestNetwork(request, callback)
        }
    }

    internal class Lease(
        private val connectivity: ConnectivityManager,
        private val callback: ConnectivityManager.NetworkCallback,
        val network: Network,
    ) : Closeable {
        override fun close() {
            runCatching { connectivity.unregisterNetworkCallback(callback) }
        }
    }

    private companion object {
        const val CONNECTION_TIMEOUT_MS = 30_000L
    }
}
