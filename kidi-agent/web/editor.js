// The editor pane: LiteApp's tabbed CodeMirror editor, plus diffs against what the agent changed in this
// conversation ("session" base, from changes.diff).
import {Editor} from './liteapp/ide/editor.mjs';
import {call} from './rpc.js';
import {theme} from './theme.js';
import {setLayout} from './layout.js';

const editor = new Editor({
  call,
  tabs: document.getElementById('editor-tabs'),
  area: document.getElementById('editor-area'),
  welcome: document.getElementById('welcome'),
  theme,
  onReveal: () => setLayout('code'),
  loadDiff: async (path, base) => {
    const result = await call('changes.diff', {path, base});
    return {old: result.old, new: result.new, label: 'before this conversation ↔ now'};
  },
});

export const openFile = path => editor.openFile(path);
export const openDiff = (path, base, mine) => editor.openDiff(path, base, mine);
export const showDiff = (path, diff, options) => editor.showDiff(path, diff, options);
export const unifiedDiff = (parent, path, oldText, newText) => editor.unifiedDiff(parent, path, oldText, newText);
export const filesChanged = paths => editor.filesChanged(paths);
