package ai.gowda.kidi

import org.junit.Assert.assertEquals
import org.junit.Test

class ReplyUpdateBufferTest {
    @Test
    fun batchesTokenDeltasAndFlushesFinalTextWithoutWaiting() {
        var now = 0L
        val updates = mutableListOf<String>()
        val buffer = ReplyUpdateBuffer({ now }, updates::add)
        buffer.append("First")
        buffer.flush()
        now = 99L
        buffer.append(" equation")
        buffer.flush()
        assertEquals(emptyList<String>(), updates)
        now = 100L
        buffer.flush()
        assertEquals(listOf("First equation"), updates)
        now = 120L
        buffer.append(" ")
        buffer.append("\$x^2\$")
        buffer.flush()
        assertEquals(1, updates.size)
        now = 200L
        buffer.flush()
        assertEquals(listOf("First equation", "First equation \$x^2\$"), updates)
        now = 201L
        buffer.append(". Final token")
        buffer.flush(force = true)
        assertEquals("First equation \$x^2\$. Final token", updates.last())
        buffer.flush(force = true)
        assertEquals(3, updates.size)
    }
}