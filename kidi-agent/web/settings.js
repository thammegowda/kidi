// Settings dialog: edits the system prompt of this server's conversations.
import {call} from './rpc.js';
import {notice} from './chat.js';

const dialog = document.getElementById('settings');
const prompt = document.getElementById('system-prompt');
let defaultPrompt = '';

document.getElementById('open-settings').addEventListener('click', async () => {
  try {
    const settings = await call('settings.get');
    prompt.value = settings.system_prompt;
    defaultPrompt = settings.default_system_prompt;
    dialog.returnValue = '';  // Escape keeps the previous value, which may be 'save'
    dialog.showModal();
  } catch (error) {
    notice(error.message, 'error');
  }
});

document.getElementById('settings-default').addEventListener('click', () => { prompt.value = defaultPrompt; });

dialog.addEventListener('close', async () => {
  if (dialog.returnValue !== 'save') return;
  try {
    await call('settings.update', {system_prompt: prompt.value});
    notice('System prompt updated.', 'session');
  } catch (error) {
    notice(`Settings not saved: ${error.message}`, 'error');
  }
});
