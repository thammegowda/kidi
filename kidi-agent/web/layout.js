// Chat layout: the conversation fills the window. Code layout: files | editor | chat. Remembered per browser.
const button = document.getElementById('layout-toggle');
let layout = localStorage.getItem('kidi-layout') === 'chat' ? 'chat' : 'code';

function apply() {
  document.body.classList.toggle('layout-chat', layout === 'chat');
  button.textContent = layout === 'chat' ? 'Code view' : 'Chat view';
  button.title = layout === 'chat' ? 'Show files, editor and terminal' : 'Show only the conversation';
}

export function setLayout(value) {
  if (value === layout) return;
  layout = value;
  localStorage.setItem('kidi-layout', layout);
  apply();
}

button.addEventListener('click', () => setLayout(layout === 'chat' ? 'code' : 'chat'));
apply();
