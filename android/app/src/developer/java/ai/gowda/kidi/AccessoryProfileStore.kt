package ai.gowda.kidi

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import androidx.core.content.edit
import org.json.JSONObject
import java.security.KeyStore
import java.security.SecureRandom
import java.util.Base64
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

internal class AccessoryProfileStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)

    fun profiles(): List<AccessoryProfile> = preferences.all
        .filterKeys { it.startsWith(PROFILE_PREFIX) }
        .map { (key, value) ->
            val deviceId = key.removePrefix(PROFILE_PREFIX)
            decode(deviceId, requireNotNull(value as? String))
        }
        .sortedBy(AccessoryProfile::name)

    fun profile(deviceId: String): AccessoryProfile? =
        preferences.getString(PROFILE_PREFIX + deviceId, null)?.let { decode(deviceId, it) }

    fun save(profile: AccessoryProfile) {
        preferences.edit(commit = true) {
            putString(PROFILE_PREFIX + profile.deviceId, encode(profile))
        }
    }

    fun remove(deviceId: String) {
        preferences.edit(commit = true) { remove(PROFILE_PREFIX + deviceId) }
    }

    private fun encode(profile: AccessoryProfile): String {
        val cleartext = JSONObject()
            .put("v", AccessoryProtocol.VERSION)
            .put("device_id", profile.deviceId)
            .put("name", profile.name)
            .put("host", profile.host)
            .put("control_port", profile.controlPort)
            .put("media_port", profile.mediaPort)
            .put("control_certificate_sha256", profile.controlCertificateSha256.base64())
            .put("media_certificate_sha256", profile.mediaCertificateSha256.base64())
            .put("controller_token", profile.controllerToken.base64())
            .put("softap_ssid", profile.softApSsid)
            .put("softap_password", profile.softApPassword)
            .toString()
            .encodeToByteArray()
        require(cleartext.size <= MAXIMUM_PROFILE_BYTES) { "Accessory profile exceeds its storage bound" }
        val iv = ByteArray(12).also(SecureRandom()::nextBytes)
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, key(), GCMParameterSpec(128, iv))
        cipher.updateAAD(profile.deviceId.encodeToByteArray())
        return (iv + cipher.doFinal(cleartext)).base64()
    }

    private fun decode(deviceId: String, encoded: String): AccessoryProfile {
        val encrypted = encoded.unbase64()
        require(encrypted.size > 12 + 16) { "Encrypted accessory profile is truncated" }
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, encrypted.copyOfRange(0, 12)))
        cipher.updateAAD(deviceId.encodeToByteArray())
        val cleartext = cipher.doFinal(encrypted.copyOfRange(12, encrypted.size))
        require(cleartext.size <= MAXIMUM_PROFILE_BYTES) { "Accessory profile exceeds its storage bound" }
        val objectValue = JSONObject(cleartext.decodeToString())
        require(objectValue.getInt("v") == AccessoryProtocol.VERSION && objectValue.getString("device_id") == deviceId) {
            "Stored accessory profile identity is invalid"
        }
        return AccessoryProfile(
            deviceId = deviceId,
            name = objectValue.getString("name"),
            host = objectValue.getString("host"),
            controlPort = objectValue.getInt("control_port"),
            mediaPort = objectValue.getInt("media_port"),
            controlCertificateSha256 = objectValue.getString("control_certificate_sha256").unbase64(),
            mediaCertificateSha256 = objectValue.getString("media_certificate_sha256").unbase64(),
            controllerToken = objectValue.getString("controller_token").unbase64(),
            softApSsid = objectValue.getString("softap_ssid"),
            softApPassword = objectValue.getString("softap_password"),
        )
    }

    private fun key(): SecretKey {
        val store = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
        (store.getKey(KEY_ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, ANDROID_KEYSTORE)
        generator.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setRandomizedEncryptionRequired(true)
                .build(),
        )
        return generator.generateKey()
    }

    private fun ByteArray.base64() = Base64.getEncoder().encodeToString(this)
    private fun String.unbase64() = Base64.getDecoder().decode(this)

    private companion object {
        const val PREFERENCES = "kidi-accessory-profiles-v1"
        const val PROFILE_PREFIX = "profile."
        const val ANDROID_KEYSTORE = "AndroidKeyStore"
        const val KEY_ALIAS = "kidi-accessory-profile-key-v1"
        const val TRANSFORMATION = "AES/GCM/NoPadding"
        const val MAXIMUM_PROFILE_BYTES = 4096
    }
}
