package ai.gowda.kidi

import android.text.Spannable
import android.text.SpannableStringBuilder
import android.text.Spanned
import io.noties.markwon.ext.latex.JLatexAsyncDrawableSpan
import io.noties.markwon.ext.latex.JLatexMathBlock
import io.noties.markwon.ext.latex.JLatexMathNode
import io.noties.markwon.inlineparser.InlineProcessor
import io.noties.markwon.inlineparser.MarkwonInlineParser
import org.commonmark.node.Node
import org.commonmark.node.Block
import org.commonmark.node.Paragraph
import org.commonmark.node.Text
import org.commonmark.parser.block.AbstractBlockParser
import org.commonmark.parser.block.AbstractBlockParserFactory
import org.commonmark.parser.block.BlockContinue
import org.commonmark.parser.block.BlockStart
import org.commonmark.parser.block.MatchedBlockParser
import org.commonmark.parser.block.ParserState

private val pricePrefix = Regex("^[0-9][0-9,.]*(?:\\s+[A-Za-z]{2,}|[;:])")

internal fun reuseMathSpans(previous: Spannable, next: Spanned): Spanned {
    val available = previous.getSpans(0, previous.length, JLatexAsyncDrawableSpan::class.java)
        .groupBy { it.javaClass to it.getDrawable().destination }
        .mapValues { ArrayDeque(it.value) }
    val result = SpannableStringBuilder(next)
    result.getSpans(0, result.length, JLatexAsyncDrawableSpan::class.java).forEach { span ->
        val retained = available[span.javaClass to span.getDrawable().destination]?.removeFirstOrNull()
        if (retained != null) {
            val start = result.getSpanStart(span)
            val end = result.getSpanEnd(span)
            val flags = result.getSpanFlags(span)
            result.removeSpan(span)
            result.setSpan(retained, start, end, flags)
            previous.removeSpan(retained)
        }
    }
    return result
}

internal fun mathInlineParserFactory(): MarkwonInlineParser.FactoryBuilder =
    MarkwonInlineParser.factoryBuilderNoDefaults().apply {
        addInlineProcessor(MathInlineProcessor('$'))
        addInlineProcessor(MathInlineProcessor('\\'))
        includeDefaults()
    }

internal class ClosedMathBlockParser private constructor(private val fence: String) : AbstractBlockParser() {
    private val math = JLatexMathBlock()
    private val content = StringBuilder()
    private var closed = false
    private var openingLine = true

    override fun getBlock(): Block = math

    override fun tryContinue(state: ParserState): BlockContinue {
        val remainder = state.line.subSequence(state.nextNonSpaceIndex, state.line.length).toString()
        if (state.indent < 4 && remainder.trimEnd(' ', '\t') == fence) {
            closed = true
            return BlockContinue.finished()
        }
        return BlockContinue.atIndex(state.index)
    }

    override fun addLine(line: CharSequence) {
        if (openingLine) openingLine = false else content.append(line).append('\n')
    }

    override fun closeBlock() {
        if (closed) {
            math.latex(content.toString())
        } else {
            val literal = Paragraph().apply { appendChild(Text("$fence\n$content")) }
            math.insertAfter(literal)
            math.unlink()
        }
    }

    class Factory : AbstractBlockParserFactory() {
        override fun tryStart(state: ParserState, matchedBlockParser: MatchedBlockParser): BlockStart? {
            val fence = state.line.subSequence(state.nextNonSpaceIndex, state.line.length).toString().trimEnd(' ', '\t')
            if (state.indent >= 4 || fence.length < 2 || fence.any { it != '$' }) return BlockStart.none()
            return BlockStart.of(ClosedMathBlockParser(fence)).atIndex(state.line.length)
        }
    }
}

private class MathInlineProcessor(private val trigger: Char) : InlineProcessor() {
    override fun specialCharacter(): Char = trigger

    override fun parse(): Node? {
        val opening = when {
            trigger == '$' && input.startsWith("\$\$", index) -> "\$\$"
            trigger == '$' -> "\$"
            input.startsWith("\\(", index) -> "\\("
            input.startsWith("\\[", index) -> "\\["
            else -> return null
        }
        val closing = when (opening) { "\\(" -> "\\)"; "\\[" -> "\\]"; else -> opening }
        val contentStart = index + opening.length
        var end = input.indexOf(closing, contentStart)
        while (end >= 0 && escaped(end)) end = input.indexOf(closing, end + closing.length)
        val content = if (end >= 0) input.substring(contentStart, end) else ""
        val singleDollar = opening == "\$"
        val valid = content.isNotBlank() && content.length <= 4096 && (!singleDollar ||
            (!content.first().isWhitespace() && !content.last().isWhitespace() && '\n' !in content && '`' !in content &&
                !pricePrefix.containsMatchIn(content) &&
                input.getOrNull(end + 1)?.isDigit() != true))
        if (!valid) {
            index = contentStart
            return text(opening)
        }
        index = end + closing.length
        return JLatexMathNode().apply { latex(content.trim()) }
    }

    private fun escaped(position: Int): Boolean {
        var start = position
        while (start > 0 && input[start - 1] == '\\') start--
        return (position - start) % 2 == 1
    }
}