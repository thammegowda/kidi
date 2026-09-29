package ai.gowda.kidi

import io.noties.markwon.ext.latex.JLatexMathNode
import io.noties.markwon.ext.latex.JLatexMathBlock
import org.commonmark.node.AbstractVisitor
import org.commonmark.node.CustomBlock
import org.commonmark.node.CustomNode
import org.commonmark.node.Text
import org.commonmark.parser.Parser
import org.junit.Assert.assertEquals
import org.junit.Test

class MarkdownMathTest {
    private val parser = Parser.builder().inlineParserFactory(mathInlineParserFactory().build())
        .customBlockParserFactory(ClosedMathBlockParser.Factory()).build()

    private fun equations(markdown: String): List<String> {
        val result = mutableListOf<String>()
        parser.parse(markdown).accept(object : AbstractVisitor() {
            override fun visit(customBlock: CustomBlock) {
                if (customBlock is JLatexMathBlock) result.add(customBlock.latex().trim())
                visitChildren(customBlock)
            }

            override fun visit(customNode: CustomNode) {
                if (customNode is JLatexMathNode) result.add(customNode.latex())
                visitChildren(customNode)
            }
        })
        return result
    }

    @Test
    fun recognizesModelMathDelimiters() {
        assertEquals(listOf("x^2", "\\frac{1}{2}", "a+b", "\\sum_i x_i"),
            equations("Inline \$x^2\$, \$\$\\frac{1}{2}\$\$, \\(a+b\\), and \\[\\sum_i x_i\\]."))
    }

    @Test
    fun leavesCodePricesAndEscapedDollarsAlone() {
        assertEquals(emptyList<String>(), equations("Price \$5 and \$10; escaped \\\$x\$; `\$x^2\$`.\n\n" +
            "```latex\n\\(x\\)\n\$x\$\n```\n\n    \$y\$\n"))
        assertEquals(listOf("x+1"), equations("Price \$5 and \$10; then \$x+1\$."))
    }

    @Test
    fun incompleteStreamingMathIsNotParsedUntilClosed() {
        assertEquals(emptyList<String>(), equations("Partial \\(\\frac{1}{2}"))
        assertEquals(listOf("\\frac{1}{2}"), equations("Partial \\(\\frac{1}{2}\\)"))
        assertEquals(emptyList<String>(), equations("Partial \$x+"))
        assertEquals(listOf("x+1"), equations("Partial \$x+1\$"))
    }

    @Test
    fun displayMathWaitsForMatchingFenceAndPreservesUnfinishedText() {
        val unfinished = "\$\$\n\\frac{1}{2}\n"
        assertEquals(emptyList<String>(), equations(unfinished))
        assertEquals(unfinished, (parser.parse(unfinished).firstChild.firstChild as Text).literal)
        assertEquals(emptyList<String>(), equations("$unfinished\$\$\$"))
        assertEquals(listOf("\\frac{1}{2}"), equations("$unfinished\$\$"))
        assertEquals(listOf("x+1"), equations("> \$\$\n> x+1\n> \$\$"))
        assertEquals(emptyList<String>(), equations("```latex\n\$\$\nx+1\n\$\$\n```"))
    }
}