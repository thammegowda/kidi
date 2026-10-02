import {parse} from '../src/web/libs/acorn/acorn.mjs';
import {full} from '../src/web/libs/acorn-walk/walk.mjs';

export function unsignedHeapIndices(source) {
    const replacements = new Map();
    const tokens = [];
    const tree = parse(source, {ecmaVersion: 'latest', sourceType: 'module', onToken: tokens});
    const shifts = tokens.filter(token => token.value === '>>');
    full(tree, node => {
        if (node.type !== 'MemberExpression' || !node.computed) return;
        const heap = node.object.type === 'SequenceExpression' ? node.object.expressions.at(-1) : node.object;
        if (heap.type !== 'Identifier' || !/^HEAP(?:U?8|U?16|U?32|U?64|F32|F64)$/.test(heap.name)) return;
        full(node.property, expression => {
            if (expression.type !== 'BinaryExpression' || expression.operator !== '>>') return;
            const token = shifts.find(token => token.start >= expression.left.end && token.end <= expression.right.start);
            if (!token) throw new Error('Cannot locate generated heap-index shift');
            replacements.set(token.start, token.end);
        });
    });
    for (const [start, end] of [...replacements].sort((left, right) => right[0] - left[0]))
        source = source.slice(0, start) + '>>>' + source.slice(end);
    return source;
}