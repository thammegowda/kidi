package ai.gowda.kidi

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddComment
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.ListItemDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.flow.distinctUntilChanged
import java.text.DateFormat
import java.util.Date

@Composable
internal fun ChatHistoryDrawer(
    state: KidiUiState,
    visible: Boolean,
    enabled: Boolean,
    onQuery: (String) -> Unit,
    onMore: () -> Unit,
    onRetry: () -> Unit,
    onSelect: (String) -> Unit,
    onNewChat: () -> Unit,
    onClose: () -> Unit,
) {
    val listState = rememberLazyListState()
    val dates = remember { DateFormat.getDateInstance(DateFormat.MEDIUM) }
    LaunchedEffect(visible, state.historyQuery) {
        if (visible) listState.scrollToItem(0)
    }
    LaunchedEffect(visible, state.history.size, state.historyHasMore, state.historyLoading, state.historyError) {
        if (visible && state.historyHasMore && !state.historyLoading && state.historyError == null) {
            snapshotFlow { listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: -1 }
                .distinctUntilChanged().collect { last ->
                    if (last >= state.history.size - 5) onMore()
                }
        }
    }
    ModalDrawerSheet(modifier = Modifier.widthIn(max = 360.dp), drawerContainerColor = MaterialTheme.colorScheme.surface) {
        Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Chats", Modifier.weight(1f), style = MaterialTheme.typography.titleLarge)
            ToolButton(Icons.Default.Close, "Close chat history", onClose)
        }
        OutlinedTextField(
            value = state.historyQuery, onValueChange = onQuery,
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp).testTag("chat-history-search"),
            label = { Text("Search chats") }, singleLine = true, shape = RoundedCornerShape(8.dp),
            leadingIcon = { Icon(Icons.Default.Search, null) },
            trailingIcon = {
                if (state.historyQuery.isNotEmpty()) ToolButton(Icons.Default.Close, "Clear chat search", { onQuery("") })
            },
        )
        TextButton(onClick = onNewChat, enabled = enabled,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp).fillMaxWidth()) {
            Icon(Icons.Default.AddComment, null, Modifier.size(20.dp))
            Spacer(Modifier.width(12.dp))
            Text("New chat", Modifier.weight(1f))
        }
        HorizontalDivider()
        LazyColumn(state = listState, modifier = Modifier.weight(1f).fillMaxWidth().testTag("chat-history-list")) {
            items(state.history.size, key = { state.history[it].id }) { index ->
                val chat = state.history[index]
                val selected = chat.id == state.activeChatId
                ListItem(
                    modifier = Modifier.fillMaxWidth().selectable(selected = selected, enabled = enabled,
                        role = Role.Button, onClick = { onSelect(chat.id) }),
                    headlineContent = { Text(chat.title, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                    supportingContent = {
                        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text(chat.preview, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(dates.format(Date(chat.updatedAt)), style = MaterialTheme.typography.labelSmall)
                        }
                    },
                    colors = ListItemDefaults.colors(containerColor = if (selected) MaterialTheme.colorScheme.secondaryContainer else Color.Transparent),
                )
            }
            if (state.historyLoading) item {
                Box(Modifier.fillMaxWidth().padding(24.dp), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(Modifier.size(24.dp), strokeWidth = 2.dp)
                }
            }
            if (state.historyError != null) item {
                Column(Modifier.fillMaxWidth().padding(20.dp)) {
                    Text(state.historyError, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium)
                    ToolButton(Icons.Default.Refresh, "Retry chat history", onRetry)
                }
            } else if (!state.historyLoading && state.history.isEmpty()) item {
                Text(if (state.historyQuery.isBlank()) "No chats yet" else "No matching chats",
                    Modifier.padding(20.dp), style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}