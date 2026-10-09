package ai.gowda.kidi

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class BleMessageAssemblerTest {
    @Test
    fun reassemblesEveryByteBoundary() {
        val message = ByteArray(700) { (it and 0xff).toByte() }
        val framed = frameBleMessage(message)
        val assembler = BleMessageAssembler()
        var result: ByteArray? = null
        for (byte in framed) result = assembler.append(byteArrayOf(byte))
        assertArrayEquals(message, result)
    }

    @Test
    fun waitsForCompleteFrame() {
        val framed = frameBleMessage(byteArrayOf(1, 2, 3))
        val assembler = BleMessageAssembler()
        assertNull(assembler.append(framed.copyOfRange(0, 4)))
        assertArrayEquals(byteArrayOf(1, 2, 3), assembler.append(framed.copyOfRange(4, 5)))
    }

    @Test
    fun rejectsEmptyOversizedAndTrailingData() {
        assertThrows(IllegalArgumentException::class.java) { frameBleMessage(ByteArray(0)) }
        assertThrows(IllegalArgumentException::class.java) {
            frameBleMessage(ByteArray(AccessoryProtocol.MAXIMUM_BLE_MESSAGE_BYTES + 1))
        }
        assertThrows(IllegalArgumentException::class.java) {
            BleMessageAssembler().append(byteArrayOf(0, 1, 7, 8))
        }
    }
}
