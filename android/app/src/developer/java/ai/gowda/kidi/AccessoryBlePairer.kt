package ai.gowda.kidi

import android.Manifest
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.BluetoothStatusCodes
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.os.Build
import android.os.ParcelUuid
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeout
import java.io.ByteArrayOutputStream
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

internal class AccessoryBlePairer(
    private val context: Context,
    private val profiles: AccessoryProfileStore,
) {
    @RequiresPermission(allOf = [Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT])
    suspend fun pair(invitation: PairingInvitation, controllerName: String): AccessoryProfile {
        val session = AccessoryPairingSession(invitation, controllerName)
        val device = withTimeout(SCAN_TIMEOUT_MS) { scan(invitation) }
        val response = withTimeout(EXCHANGE_TIMEOUT_MS) { exchange(device, session.request) }
        return session.acceptResponse(response).also(profiles::save)
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    private suspend fun scan(invitation: PairingInvitation): BluetoothDevice = suspendCancellableCoroutine { continuation ->
        val adapter = context.getSystemService(BluetoothManager::class.java).adapter
            ?: return@suspendCancellableCoroutine continuation.resumeWithException(
                UnsupportedOperationException("Bluetooth is unavailable"),
            )
        if (!adapter.isEnabled) {
            return@suspendCancellableCoroutine continuation.resumeWithException(
                IllegalStateException("Bluetooth is disabled"),
            )
        }
        val scanner = adapter.bluetoothLeScanner
            ?: return@suspendCancellableCoroutine continuation.resumeWithException(
                IllegalStateException("Bluetooth scanner is unavailable"),
            )
        val complete = AtomicBoolean()
        lateinit var callback: ScanCallback
        fun finish(result: Result<BluetoothDevice>) {
            if (!complete.compareAndSet(false, true)) return
            scanner.stopScan(callback)
            result.fold(continuation::resume, continuation::resumeWithException)
        }
        callback = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, result: ScanResult) {
                val data = result.scanRecord?.getServiceData(ParcelUuid(AccessoryProtocol.BLE_SERVICE_UUID)) ?: return
                if (
                    data.size == ADVERTISEMENT_BYTES &&
                    data[0] == AccessoryProtocol.VERSION.toByte() &&
                    data.copyOfRange(1, data.size).toHex() == invitation.deviceId
                ) {
                    finish(Result.success(result.device))
                }
            }

            override fun onScanFailed(errorCode: Int) {
                finish(Result.failure(IllegalStateException("Accessory BLE scan failed: $errorCode")))
            }
        }
        continuation.invokeOnCancellation {
            if (complete.compareAndSet(false, true)) scanner.stopScan(callback)
        }
        scanner.startScan(
            listOf(ScanFilter.Builder().setServiceUuid(ParcelUuid(AccessoryProtocol.BLE_SERVICE_UUID)).build()),
            ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build(),
            callback,
        )
    }

    @Suppress("DEPRECATION")
    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    private suspend fun exchange(device: BluetoothDevice, request: ByteArray): ByteArray =
        suspendCancellableCoroutine { continuation ->
            val complete = AtomicBoolean()
            val framed = frameBleMessage(request)
            val assembler = BleMessageAssembler()
            var mtu = DEFAULT_MTU
            var requestOffset = 0
            var requestCharacteristic: BluetoothGattCharacteristic? = null
            fun fail(gatt: BluetoothGatt, message: String) {
                if (!complete.compareAndSet(false, true)) return
                gatt.disconnect()
                gatt.close()
                continuation.resumeWithException(IllegalStateException(message))
            }
            fun succeed(gatt: BluetoothGatt, response: ByteArray) {
                if (!complete.compareAndSet(false, true)) return
                gatt.disconnect()
                gatt.close()
                continuation.resume(response)
            }
            fun writeNext(gatt: BluetoothGatt) {
                val characteristic = requestCharacteristic
                    ?: return fail(gatt, "Pair request characteristic is unavailable")
                if (requestOffset >= framed.size) return
                val end = minOf(framed.size, requestOffset + maxOf(1, mtu - ATT_HEADER_BYTES))
                val chunk = framed.copyOfRange(requestOffset, end)
                requestOffset = end
                val accepted = if (Build.VERSION.SDK_INT >= 33) {
                    gatt.writeCharacteristic(
                        characteristic,
                        chunk,
                        BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT,
                    ) == BluetoothStatusCodes.SUCCESS
                } else {
                    characteristic.writeType = BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT
                    characteristic.value = chunk
                    gatt.writeCharacteristic(characteristic)
                }
                if (!accepted) fail(gatt, "Unable to write accessory pairing request")
            }
            fun enableResponse(gatt: BluetoothGatt) {
                val service = gatt.getService(AccessoryProtocol.BLE_SERVICE_UUID)
                    ?: return fail(gatt, "Accessory BLE service is missing")
                requestCharacteristic = service.getCharacteristic(AccessoryProtocol.BLE_PAIR_REQUEST_UUID)
                    ?: return fail(gatt, "Accessory pair request characteristic is missing")
                val response = service.getCharacteristic(AccessoryProtocol.BLE_PAIR_RESPONSE_UUID)
                    ?: return fail(gatt, "Accessory pair response characteristic is missing")
                if (!gatt.setCharacteristicNotification(response, true)) {
                    return fail(gatt, "Unable to enable accessory pairing indications")
                }
                val descriptor = response.getDescriptor(CLIENT_CONFIGURATION_UUID)
                    ?: return fail(gatt, "Accessory indication descriptor is missing")
                val accepted = if (Build.VERSION.SDK_INT >= 33) {
                    gatt.writeDescriptor(descriptor, BluetoothGattDescriptor.ENABLE_INDICATION_VALUE) ==
                        BluetoothStatusCodes.SUCCESS
                } else {
                    descriptor.value = BluetoothGattDescriptor.ENABLE_INDICATION_VALUE
                    gatt.writeDescriptor(descriptor)
                }
                if (!accepted) fail(gatt, "Unable to configure accessory pairing indications")
            }
            fun receive(gatt: BluetoothGatt, value: ByteArray) {
                runCatching { assembler.append(value) }
                    .onSuccess { it?.let { response -> succeed(gatt, response) } }
                    .onFailure { fail(gatt, it.message ?: "Invalid accessory BLE response") }
            }
            val callback = object : BluetoothGattCallback() {
                override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
                    if (status != BluetoothGatt.GATT_SUCCESS) {
                        fail(gatt, "Accessory BLE connection failed: $status")
                    } else if (newState == BluetoothProfile.STATE_CONNECTED) {
                        if (!gatt.discoverServices()) fail(gatt, "Unable to discover accessory BLE service")
                    } else if (newState == BluetoothProfile.STATE_DISCONNECTED && !complete.get()) {
                        fail(gatt, "Accessory disconnected during pairing")
                    }
                }

                override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
                    if (status != BluetoothGatt.GATT_SUCCESS) {
                        fail(gatt, "Accessory BLE discovery failed: $status")
                    } else if (!gatt.requestMtu(REQUESTED_MTU)) {
                        enableResponse(gatt)
                    }
                }

                override fun onMtuChanged(gatt: BluetoothGatt, requestedMtu: Int, status: Int) {
                    if (status == BluetoothGatt.GATT_SUCCESS) mtu = requestedMtu
                    enableResponse(gatt)
                }

                override fun onDescriptorWrite(gatt: BluetoothGatt, descriptor: BluetoothGattDescriptor, status: Int) {
                    if (status != BluetoothGatt.GATT_SUCCESS) {
                        fail(gatt, "Accessory indication setup failed: $status")
                    } else {
                        writeNext(gatt)
                    }
                }

                override fun onCharacteristicWrite(
                    gatt: BluetoothGatt,
                    characteristic: BluetoothGattCharacteristic,
                    status: Int,
                ) {
                    if (status != BluetoothGatt.GATT_SUCCESS) {
                        fail(gatt, "Accessory pairing write failed: $status")
                    } else {
                        writeNext(gatt)
                    }
                }

                override fun onCharacteristicChanged(
                    gatt: BluetoothGatt,
                    characteristic: BluetoothGattCharacteristic,
                    value: ByteArray,
                ) = receive(gatt, value)

                @Deprecated("Used on Android 12 and earlier")
                override fun onCharacteristicChanged(
                    gatt: BluetoothGatt,
                    characteristic: BluetoothGattCharacteristic,
                ) = receive(gatt, characteristic.value)
            }
            val gatt = device.connectGatt(context, false, callback, BluetoothDevice.TRANSPORT_LE)
            continuation.invokeOnCancellation {
                if (complete.compareAndSet(false, true)) {
                    gatt.disconnect()
                    gatt.close()
                }
            }
        }

    private fun ByteArray.toHex() = joinToString("") { "%02x".format(it) }

    private companion object {
        val CLIENT_CONFIGURATION_UUID: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
        const val ADVERTISEMENT_BYTES = 17
        const val DEFAULT_MTU = 23
        const val REQUESTED_MTU = 517
        const val ATT_HEADER_BYTES = 3
        const val SCAN_TIMEOUT_MS = 20_000L
        const val EXCHANGE_TIMEOUT_MS = 30_000L
    }
}

internal fun frameBleMessage(message: ByteArray): ByteArray {
    require(message.size in 1..AccessoryProtocol.MAXIMUM_BLE_MESSAGE_BYTES) { "BLE message length is invalid" }
    return byteArrayOf((message.size ushr 8).toByte(), message.size.toByte()) + message
}

internal class BleMessageAssembler {
    private val bytes = ByteArrayOutputStream()
    private var expected: Int? = null

    fun append(chunk: ByteArray): ByteArray? {
        require(chunk.isNotEmpty()) { "Empty BLE fragment" }
        bytes.write(chunk)
        val current = bytes.toByteArray()
        if (expected == null && current.size >= 2) {
            expected = ((current[0].toInt() and 0xff) shl 8) or (current[1].toInt() and 0xff)
            require(expected!! in 1..AccessoryProtocol.MAXIMUM_BLE_MESSAGE_BYTES) {
                "BLE message length is invalid"
            }
        }
        val length = expected ?: return null
        require(current.size <= length + 2) { "BLE response contains trailing bytes" }
        return if (current.size == length + 2) current.copyOfRange(2, current.size) else null
    }
}
