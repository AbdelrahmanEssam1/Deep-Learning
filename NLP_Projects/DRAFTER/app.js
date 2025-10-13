const messagesEl = document.getElementById('messages');
const documentEl = document.getElementById('document');
const chatForm = document.getElementById('chatForm');
const inputEl = document.getElementById('text');
const saveBtn = document.getElementById('saveBtn');
const filenameEl = document.getElementById('filename');
const saveStatus = document.getElementById('saveStatus');

function addBubble(text, cls) {
  const div = document.createElement('div');
  div.className = `bubble ${cls}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function fetchState() {
  const res = await fetch('/api/state');
  const data = await res.json();
  documentEl.textContent = data.document || '';
  return data;
}

async function sendMessage(text) {
  addBubble(text, 'user');
  const res = await fetch('/api/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  const data = await res.json();
  (data.ai_messages || []).forEach(t => addBubble(t, 'ai'));
  (data.tool_results || []).forEach(t => addBubble(t, 'tool'));
  documentEl.textContent = data.document || '';
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  await sendMessage(text);
});

saveBtn.addEventListener('click', async () => {
  const filename = filenameEl.value.trim();
  saveStatus.textContent = 'Saving...';
  const res = await fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename })
  });
  const data = await res.json();
  if (data.ok) {
    saveStatus.textContent = `Saved to ${data.path}`;
  } else {
    saveStatus.textContent = `Error: ${data.error}`;
  }
});

(async function init() {
  addBubble('Welcome to Drafter! How can I help with your document?', 'ai');
  await fetchState();
})();


