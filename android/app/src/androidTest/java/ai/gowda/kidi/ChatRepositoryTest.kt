package ai.gowda.kidi

import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.UUID

class ChatRepositoryTest {
    @Test
    fun modelContextKeepsStoredHistoryButSkipsUnansweredTurns() {
        val history = listOf(
            ChatMessage(MessageRole.USER, "Unanswered"),
            ChatMessage(MessageRole.ASSISTANT, "", status = MessageStatus.FAILED),
            ChatMessage(MessageRole.USER, "Question"),
            ChatMessage(MessageRole.ASSISTANT, "First agent"),
            ChatMessage(MessageRole.ASSISTANT, "Second agent", sender = ChatParticipant.agent("other/model")),
            ChatMessage(MessageRole.USER, "Continue"),
        )
        assertEquals(listOf("Question", "First agent\n\nSecond agent", "Continue"), history.textGenerationContext().map { it.content })
        assertEquals(6, history.size)
    }

    @Test
    fun historyPagesSearchAndLegacyMigrationSurviveReopening() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = "chat-test-${UUID.randomUUID()}.db"
        val legacy = """[{"role":"USER","content":"Legacy conversation"}]"""
        try {
            val legacyId = ChatRepository(context, name).use { repository ->
                val restored = requireNotNull(repository.restore(legacy))
                assertEquals("Legacy conversation", restored.page.messages.single().content)
                assertEquals(restored.id, repository.restore(legacy)?.id)
                for (index in 1..65) {
                    val id = repository.createThread(listOf(ChatParticipant.USER, ChatParticipant.LEGACY_AGENT), "chat-$index")
                    repository.append(id, ChatMessage(MessageRole.USER, "Question $index", createdAt = index.toLong()))
                    repository.append(id, ChatMessage(MessageRole.ASSISTANT,
                        if (index == 1) "Hidden astronomy answer" else "Reply $index", createdAt = index.toLong()))
                }
                val first = repository.recent()
                assertEquals(30, first.chats.size)
                assertTrue(first.hasMore)
                val second = repository.recent(after = first.chats.last())
                assertEquals(30, second.chats.size)
                assertTrue(second.hasMore)
                val third = repository.recent(after = second.chats.last())
                assertEquals(6, third.chats.size)
                assertFalse(third.hasMore)
                assertEquals(66, (first.chats + second.chats + third.chats).map { it.id }.toSet().size)
                assertEquals(listOf("chat-1"), repository.recent("astro").chats.map { it.id })
                assertTrue(repository.recent("\" OR *").chats.isEmpty())
                repository.select("chat-1")
                restored.id
            }
            ChatRepository(context, name).use { repository ->
                assertEquals("chat-1", repository.restore(legacy)?.id)
                assertEquals(legacyId, repository.recent().chats.first().id)
                val messages = requireNotNull(repository.load("chat-1")).page.messages
                assertEquals("Hidden astronomy answer", messages.last().content)
                repository.select(null)
                assertNull(repository.restore(legacy))
                assertTrue(repository.load(legacyId) != null)
            }
        } finally {
            context.deleteDatabase(name)
        }
    }

    @Test
    fun membersAttachmentsMessagePagesAndDeletionStayConsistent() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = "chat-test-${UUID.randomUUID()}.db"
        val agent = ChatParticipant("researcher", ParticipantKind.AGENT, "Researcher", "model/research")
        try {
            ChatRepository(context, name).use { repository ->
                val thread = repository.createThread(listOf(ChatParticipant.USER, agent, ChatParticipant.LEGACY_AGENT))
                val attachments = AttachmentKind.entries.map { kind ->
                    MessageAttachment(kind = kind, localUri = "content://test/${kind.name}", name = kind.name,
                        mimeType = "application/octet-stream", sizeBytes = 42)
                }
                val attached = repository.append(thread, ChatMessage(MessageRole.USER, "Files", attachments = attachments))
                repeat(60) { repository.append(thread, ChatMessage(MessageRole.ASSISTANT, "Response $it", sender = agent)) }
                val latest = repository.messages(thread)
                assertEquals(50, latest.messages.size)
                assertTrue(latest.hasMore)
                assertEquals(agent, latest.messages.last().sender)
                val earlier = repository.messages(thread, before = latest.messages.first().sequence)
                assertEquals(11, earlier.messages.size)
                assertFalse(earlier.hasMore)
                assertEquals(attached.attachments, earlier.messages.first().attachments)
                val outsider = ChatParticipant("outsider", ParticipantKind.AGENT, "Other")
                repository.createThread(listOf(ChatParticipant.USER, outsider))
                var rejected = false
                try {
                    repository.append(thread, ChatMessage(MessageRole.ASSISTANT, "Must not be stored", sender = outsider))
                } catch (_: android.database.sqlite.SQLiteConstraintException) {
                    rejected = true
                }
                assertTrue(rejected)
                assertTrue(repository.recent("stored").chats.isEmpty())
                val failed = repository.append(thread, ChatMessage(MessageRole.ASSISTANT, "", sender = agent, status = MessageStatus.FAILED))
                assertEquals(failed, repository.messages(thread).messages.last())
                val stopped = repository.append(thread, ChatMessage(MessageRole.ASSISTANT, "", sender = agent, status = MessageStatus.STOPPED))
                assertEquals(stopped, repository.messages(thread).messages.last())
                repository.select(thread)
                repository.deleteThread(thread)
                assertNull(repository.load(thread))
                assertTrue(repository.recent("response").chats.isEmpty())
                repository.readableDatabase.rawQuery("SELECT COUNT(*) FROM attachments", null).use {
                    it.moveToFirst()
                    assertEquals(0, it.getInt(0))
                }
            }
        } finally {
            context.deleteDatabase(name)
        }
    }
}