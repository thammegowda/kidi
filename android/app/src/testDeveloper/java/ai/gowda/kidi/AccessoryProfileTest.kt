package ai.gowda.kidi

import org.junit.Assert.assertThrows
import org.junit.Test

class AccessoryProfileTest {
    @Test
    fun validatesNetworkSplitRangesAndSecrets() {
        validProfile()
        assertThrows(IllegalArgumentException::class.java) {
            validProfile().copy(controlPort = 443)
        }
        assertThrows(IllegalArgumentException::class.java) {
            validProfile().copy(mediaPort = 8443)
        }
        assertThrows(IllegalArgumentException::class.java) {
            validProfile().copy(controllerToken = ByteArray(31))
        }
        assertThrows(IllegalArgumentException::class.java) {
            validProfile().copy(softApPassword = "short")
        }
    }

    private fun validProfile() = AccessoryProfile(
        deviceId = "00112233445566778899aabbccddeeff",
        name = "Kidi accessory",
        host = "192.168.4.1",
        controlCertificateSha256 = ByteArray(32) { 1 },
        mediaCertificateSha256 = ByteArray(32) { 2 },
        controllerToken = ByteArray(32) { 3 },
        softApSsid = "kidi-test",
        softApPassword = "prototype-password",
    )
}
