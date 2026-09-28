package ai.gowda.kidi

import java.util.UUID

internal enum class MessageRole { USER, ASSISTANT }
internal enum class ParticipantKind { USER, AGENT }
internal enum class MessageStatus { COMPLETE, STOPPED, FAILED }
internal enum class AttachmentKind { IMAGE, DOCUMENT, AUDIO, VIDEO }

internal data class ChatParticipant(val id: String, val kind: ParticipantKind, val name: String, val modelId: String? = null) {
    companion object {
        val USER = ChatParticipant("local-user", ParticipantKind.USER, "You")
        val LEGACY_AGENT = ChatParticipant("legacy-agent", ParticipantKind.AGENT, "Kidi")
        fun agent(modelId: String) = ChatParticipant("agent:$modelId", ParticipantKind.AGENT, "Kidi", modelId)
    }
}

internal data class MessageAttachment(
    val id: String = UUID.randomUUID().toString(),
    val kind: AttachmentKind,
    val localUri: String,
    val name: String,
    val mimeType: String,
    val sizeBytes: Long,
    val width: Int? = null,
    val height: Int? = null,
    val durationMs: Long? = null,
)

internal data class MessageStats(val tokens: Int, val elapsedMs: Double, val decodeMs: Double, val decodeTokens: Int)

internal data class ChatMessage(
    val role: MessageRole,
    val content: String,
    val stats: MessageStats? = null,
    val id: String = UUID.randomUUID().toString(),
    val sender: ChatParticipant = if (role == MessageRole.USER) ChatParticipant.USER else ChatParticipant.LEGACY_AGENT,
    val createdAt: Long = System.currentTimeMillis(),
    val status: MessageStatus = MessageStatus.COMPLETE,
    val attachments: List<MessageAttachment> = emptyList(),
    val sequence: Long = 0,
)

internal data class ChatSummary(val id: String, val title: String, val preview: String, val updatedAt: Long)
internal data class MessagePage(val messages: List<ChatMessage>, val hasMore: Boolean)
internal data class SavedChat(val id: String, val page: MessagePage)
internal data class ChatPage(val chats: List<ChatSummary>, val hasMore: Boolean)

internal fun List<ChatMessage>.textGenerationContext(): List<ChatMessage> {
    val result = mutableListOf<ChatMessage>()
    var pendingUser: ChatMessage? = null
    for (message in takeLast(50)) {
        require(message.attachments.isEmpty()) { "This model currently supports text-only messages" }
        if (message.role == MessageRole.USER) {
            pendingUser = message
        } else if (message.status != MessageStatus.FAILED && message.content.isNotBlank()) {
            pendingUser?.let { result.add(it); pendingUser = null }
            if (result.isEmpty()) continue
            if (result.last().role == MessageRole.ASSISTANT) {
                val previous = result.last()
                result[result.lastIndex] = previous.copy(content = "${previous.content}\n\n${message.content}")
            } else result.add(message)
        }
    }
    pendingUser?.let { result.add(it) }
    return result
}