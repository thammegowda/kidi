// The terminal panel below the editor (LiteApp's xterm.js panel over the workspace terminal service).
import {TerminalPanel} from './liteapp/ide/terminal.mjs';
import {call} from './rpc.js';
import {theme} from './theme.js';
import {setLayout} from './layout.js';

const terminal = new TerminalPanel({
  call,
  panel: document.getElementById('terminal-panel'),
  host: document.getElementById('terminal-host'),
  splitter: document.getElementById('terminal-splitter'),
  theme,
  onReveal: () => setLayout('code'),
});

document.getElementById('restart-terminal').addEventListener('click', () => terminal.restart());
document.getElementById('close-terminal').addEventListener('click', () => terminal.toggle(false));
document.getElementById('toggle-terminal').addEventListener('click', () => terminal.toggle());
addEventListener('keydown', event => {
  if (event.ctrlKey && event.key === '`') {
    event.preventDefault();
    terminal.toggle();
  }
});
