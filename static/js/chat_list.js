// Doctor's conversation list (templates/doctor_messages.html): search, live previews,
// unread badges and most-recent-first ordering.
import { connectChat, formatChatTimestamp } from './chat_firebase.js';
import { ref, onValue } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-database.js";

const listEl = document.getElementById('chatList');
const search = document.getElementById('chatSearch');
const noMatch = document.getElementById('chatNoMatch');
const boot = window.CHAT_BOOT;

if (listEl) {
  const items = () => [...listEl.querySelectorAll('.chat-list-item')];

  function applySearch() {
    const q = search.value.trim().toLowerCase();
    let shown = 0;
    items().forEach(li => {
      const match = !q || li.dataset.name.includes(q);
      li.hidden = !match;
      if (match) shown++;
    });
    noMatch.hidden = shown > 0;
  }
  search.addEventListener('input', applySearch);

  function resort() {
    const sorted = items().sort((a, b) => (Number(b.dataset.ts) - Number(a.dataset.ts)) || a.dataset.name.localeCompare(b.dataset.name));
    sorted.forEach(li => listEl.appendChild(li));
  }

  function refreshTimes() {
    listEl.querySelectorAll('.chat-list-time').forEach(t => { t.textContent = formatChatTimestamp(Number(t.dataset.ts)); });
  }
  setInterval(refreshTimes, 60000);

  // Unread counts per conversation come from the sidebar badge listener (chat_badge.js)
  document.addEventListener('chat-unread', (e) => {
    const perChat = e.detail.perChat;
    if (!perChat) return;
    items().forEach(li => {
      const badge = li.querySelector('[data-unread]');
      const n = Number(perChat[li.dataset.chatId]) || 0;
      badge.textContent = n > 99 ? '99+' : String(n);
      badge.hidden = !n || li.classList.contains('is-active');
    });
  });

  connectChat(boot).then(db => {
    items().forEach(li => {
      onValue(ref(db, `chats/${li.dataset.chatId}/last_message`), snap => {
        const last = snap.val();
        if (!last) return;
        const preview = li.querySelector('.chat-list-preview');
        preview.textContent = (last.sender_uid === boot.me ? 'You: ' : '') + (last.text || '');
        li.dataset.ts = last.timestamp || 0;
        const time = li.querySelector('.chat-list-time');
        time.dataset.ts = last.timestamp || 0;
        time.textContent = formatChatTimestamp(last.timestamp);
        resort();
      }, () => { /* not readable (rules not deployed): keep the server-rendered list */ });
    });
  }).catch(() => {});
}
