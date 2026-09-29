package ai.gowda.kidi

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.json.JSONArray
import java.util.UUID

internal class ChatRepository(context: Context, name: String = "chat-threads.db") : SQLiteOpenHelper(context, name, null, 1) {
    init {
        setWriteAheadLoggingEnabled(true)
    }

    override fun onConfigure(db: SQLiteDatabase) {
        db.setForeignKeyConstraintsEnabled(true)
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""CREATE TABLE threads (
            id TEXT PRIMARY KEY NOT NULL, title TEXT, preview TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)""")
        db.execSQL("CREATE INDEX threads_recent ON threads(updated_at DESC, id DESC)")
        db.execSQL("""CREATE TABLE participants (
            id TEXT PRIMARY KEY NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('USER','AGENT')),
            name TEXT NOT NULL, model_id TEXT)""")
        db.execSQL("""CREATE TABLE thread_participants (
            thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
            participant_id TEXT NOT NULL REFERENCES participants(id),
            PRIMARY KEY(thread_id, participant_id))""")
        db.execSQL("""CREATE TABLE messages (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
            thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE, sender_id TEXT NOT NULL,
            body TEXT NOT NULL, created_at INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('COMPLETE','STOPPED','FAILED')),
            tokens INTEGER, elapsed_ms REAL, decode_ms REAL, decode_tokens INTEGER,
            FOREIGN KEY(thread_id, sender_id) REFERENCES thread_participants(thread_id, participant_id))""")
        db.execSQL("CREATE INDEX messages_thread ON messages(thread_id, sequence DESC)")
        db.execSQL("""CREATE TABLE attachments (
            id TEXT PRIMARY KEY NOT NULL, message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            position INTEGER NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('IMAGE','DOCUMENT','AUDIO','VIDEO')),
            local_uri TEXT NOT NULL, name TEXT NOT NULL, mime_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
            width INTEGER CHECK(width > 0), height INTEGER CHECK(height > 0), duration_ms INTEGER CHECK(duration_ms >= 0),
            UNIQUE(message_id, position))""")
        db.execSQL("CREATE VIRTUAL TABLE message_search USING fts4(body, tokenize=unicode61)")
        db.execSQL("""CREATE TRIGGER messages_insert AFTER INSERT ON messages BEGIN
            INSERT INTO message_search(docid, body) VALUES(new.sequence, new.body); END""")
        db.execSQL("""CREATE TRIGGER messages_delete AFTER DELETE ON messages BEGIN
            DELETE FROM message_search WHERE docid = old.sequence; END""")
        db.execSQL("""CREATE TABLE app_state (
            id INTEGER PRIMARY KEY CHECK(id = 1), active_thread_id TEXT REFERENCES threads(id) ON DELETE SET NULL,
            legacy_migrated INTEGER NOT NULL DEFAULT 0)""")
        db.execSQL("INSERT INTO app_state(id) VALUES(1)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        error("Unsupported chat database migration: $oldVersion to $newVersion")
    }

    private fun <T> transaction(action: (SQLiteDatabase) -> T): T {
        val db = writableDatabase
        db.beginTransaction()
        try {
            val result = action(db)
            db.setTransactionSuccessful()
            return result
        } finally {
            db.endTransaction()
        }
    }

    fun createThread(members: List<ChatParticipant>, id: String = UUID.randomUUID().toString()): String = transaction { db ->
        require(members.any { it.kind == ParticipantKind.USER } && members.any { it.kind == ParticipantKind.AGENT })
        val now = System.currentTimeMillis()
        db.insertOrThrow("threads", null, ContentValues().apply {
            put("id", id)
            put("created_at", now)
            put("updated_at", now)
        })
        members.forEach { addParticipant(id, it) }
        id
    }

    fun addParticipant(threadId: String, participant: ChatParticipant) = transaction { db ->
        db.execSQL("INSERT OR IGNORE INTO participants(id, kind, name, model_id) VALUES(?, ?, ?, ?)",
            arrayOf(participant.id, participant.kind.name, participant.name, participant.modelId))
        db.rawQuery("SELECT kind, model_id FROM participants WHERE id = ?", arrayOf(participant.id)).use {
            check(it.moveToFirst() && it.getString(0) == participant.kind.name && it.getString(1) == participant.modelId) {
                "Participant identity cannot change kind or model"
            }
        }
        db.execSQL("INSERT OR IGNORE INTO thread_participants(thread_id, participant_id) VALUES(?, ?)",
            arrayOf(threadId, participant.id))
    }

    fun append(threadId: String, message: ChatMessage) = transaction { db ->
        require((message.role == MessageRole.USER) == (message.sender.kind == ParticipantKind.USER))
        db.rawQuery("SELECT kind FROM participants WHERE id = ?", arrayOf(message.sender.id)).use {
            require(it.moveToFirst() && it.getString(0) == message.sender.kind.name) { "Invalid message sender" }
        }
        require(message.content.isNotBlank() || message.attachments.isNotEmpty() || message.role == MessageRole.ASSISTANT)
        val sequence = db.insertOrThrow("messages", null, ContentValues().apply {
            put("id", message.id)
            put("thread_id", threadId)
            put("sender_id", message.sender.id)
            put("body", message.content)
            put("created_at", message.createdAt)
            put("status", message.status.name)
            message.stats?.let {
                put("tokens", it.tokens)
                put("elapsed_ms", it.elapsedMs)
                put("decode_ms", it.decodeMs)
                put("decode_tokens", it.decodeTokens)
            }
        })
        message.attachments.forEachIndexed { index, attachment ->
            require(android.net.Uri.parse(attachment.localUri).scheme in setOf("content", "file"))
            require(attachment.name.isNotBlank() && attachment.mimeType.isNotBlank())
            db.insertOrThrow("attachments", null, ContentValues().apply {
                put("id", attachment.id)
                put("message_id", message.id)
                put("position", index)
                put("kind", attachment.kind.name)
                put("local_uri", attachment.localUri)
                put("name", attachment.name)
                put("mime_type", attachment.mimeType)
                put("size_bytes", attachment.sizeBytes)
                put("width", attachment.width)
                put("height", attachment.height)
                put("duration_ms", attachment.durationMs)
            })
        }
        val text = message.content.ifBlank { message.attachments.firstOrNull()?.name ?: when (message.status) {
            MessageStatus.STOPPED -> "Response stopped"
            MessageStatus.FAILED -> "Response failed"
            MessageStatus.COMPLETE -> "Empty response"
        } }
        db.execSQL("UPDATE threads SET title = COALESCE(title, ?), preview = ?, updated_at = ? WHERE id = ?",
            arrayOf<Any>(text.lineSequence().first().take(120), text.replace(Regex("\\s+"), " ").take(160), message.createdAt, threadId))
        message.copy(sequence = sequence)
    }

    fun select(id: String?) {
        writableDatabase.execSQL("UPDATE app_state SET active_thread_id = ? WHERE id = 1", arrayOf(id))
    }

    fun deleteThread(id: String) {
        writableDatabase.delete("threads", "id = ?", arrayOf(id))
    }

    fun restore(legacyMessages: String?): SavedChat? = transaction { db ->
        val migrated = db.rawQuery("SELECT legacy_migrated FROM app_state WHERE id = 1", null).use {
            it.moveToFirst()
            it.getInt(0) != 0
        }
        if (!migrated) {
            val array = JSONArray(legacyMessages ?: "[]")
            if (array.length() > 0) {
                val id = createThread(listOf(ChatParticipant.USER, ChatParticipant.LEGACY_AGENT))
                for (index in 0 until array.length()) {
                    val item = array.getJSONObject(index)
                    val message = ChatMessage(MessageRole.valueOf(item.getString("role")), item.getString("content"))
                    if (message.content.isNotBlank()) append(id, message)
                }
                select(id)
            }
            db.execSQL("UPDATE app_state SET legacy_migrated = 1 WHERE id = 1")
        }
        db.rawQuery("SELECT active_thread_id FROM app_state WHERE id = 1", null).use {
            it.moveToFirst()
            if (it.isNull(0)) null else load(it.getString(0))
        }
    }

    fun load(id: String): SavedChat? = readableDatabase.rawQuery("SELECT id FROM threads WHERE id = ?", arrayOf(id)).use {
        if (it.moveToFirst()) SavedChat(id, messages(id)) else null
    }

    fun messages(threadId: String, before: Long? = null): MessagePage {
        val arguments = mutableListOf(threadId)
        val boundary = if (before == null) "" else "AND sequence < ?".also { arguments += before.toString() }
        val rows = readableDatabase.rawQuery("""SELECT m.id, m.sequence, m.body, m.created_at, m.status,
            m.tokens, m.elapsed_ms, m.decode_ms, m.decode_tokens, p.id, p.kind, p.name, p.model_id
            FROM messages m JOIN participants p ON p.id = m.sender_id
            WHERE m.thread_id = ? $boundary ORDER BY m.sequence DESC LIMIT 51""", arguments.toTypedArray()).use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    val sender = ChatParticipant(cursor.getString(9), ParticipantKind.valueOf(cursor.getString(10)),
                        cursor.getString(11), cursor.getString(12))
                    add(ChatMessage(
                        role = if (sender.kind == ParticipantKind.USER) MessageRole.USER else MessageRole.ASSISTANT,
                        content = cursor.getString(2), id = cursor.getString(0), sequence = cursor.getLong(1),
                        createdAt = cursor.getLong(3), status = MessageStatus.valueOf(cursor.getString(4)), sender = sender,
                        stats = if (cursor.isNull(5)) null else MessageStats(cursor.getInt(5), cursor.getDouble(6), cursor.getDouble(7), cursor.getInt(8)),
                    ))
                }
            }
        }
        val page = rows.take(50).reversed()
        if (page.isEmpty()) return MessagePage(emptyList(), false)
        val attachments = mutableMapOf<String, MutableList<MessageAttachment>>()
        readableDatabase.rawQuery("SELECT * FROM attachments WHERE message_id IN (${page.joinToString { "?" }}) ORDER BY position",
            page.map { it.id }.toTypedArray()).use { cursor ->
            while (cursor.moveToNext()) {
                attachments.getOrPut(cursor.getString(1)) { mutableListOf() }.add(MessageAttachment(
                    id = cursor.getString(0), kind = AttachmentKind.valueOf(cursor.getString(3)), localUri = cursor.getString(4),
                    name = cursor.getString(5), mimeType = cursor.getString(6), sizeBytes = cursor.getLong(7),
                    width = if (cursor.isNull(8)) null else cursor.getInt(8), height = if (cursor.isNull(9)) null else cursor.getInt(9),
                    durationMs = if (cursor.isNull(10)) null else cursor.getLong(10),
                ))
            }
        }
        return MessagePage(page.map { it.copy(attachments = attachments[it.id].orEmpty()) }, rows.size > 50)
    }

    fun recent(query: String = "", after: ChatSummary? = null): ChatPage {
        val arguments = mutableListOf<String>()
        val conditions = mutableListOf("EXISTS (SELECT 1 FROM messages WHERE thread_id = threads.id)")
        if (query.isNotBlank()) {
            val words = Regex("[\\p{L}\\p{N}_]+").findAll(query).map { "\"${it.value}*\"" }.toList()
            if (words.isEmpty()) return ChatPage(emptyList(), false)
            conditions += "id IN (SELECT thread_id FROM messages JOIN message_search ON message_search.docid = messages.sequence WHERE message_search MATCH ?)"
            arguments += words.joinToString(" AND ")
        }
        if (after != null) {
            conditions += "(updated_at < ? OR (updated_at = ? AND id < ?))"
            arguments += listOf(after.updatedAt.toString(), after.updatedAt.toString(), after.id)
        }
        val rows = readableDatabase.rawQuery(
            "SELECT id, COALESCE(title, 'New chat'), preview, updated_at FROM threads WHERE ${conditions.joinToString(" AND ")} ORDER BY updated_at DESC, id DESC LIMIT 31",
            arguments.toTypedArray(),
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    add(ChatSummary(cursor.getString(0), cursor.getString(1), cursor.getString(2), cursor.getLong(3)))
                }
            }
        }
        return ChatPage(rows.take(30), rows.size > 30)
    }

}