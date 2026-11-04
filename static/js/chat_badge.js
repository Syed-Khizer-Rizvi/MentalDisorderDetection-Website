// Live unread count on the sidebar Chat/Messages link. Listens to chat_unread/{me}; if the
// browser can't read Firebase (rules not deployed), it polls /api/chat/unread instead.
// Also broadcasts per-conversation counts as a 'chat-unread' event for the doctor's list.
import { connectChat } from './chat_firebase.js';
import { ref, onValue } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-database.js";

const boot = window.CHAT_BOOT;
const badges = document.querySelectorAll('[data-chat-badge]');

function render(total, perChat) {
  badges.forEach(b => {
    b.textContent = total > 99 ? '99+' : String(total);
    b.hidden = !total;
  });
  document.dispatchEvent(new CustomEvent('chat-unread', { detail: { total, perChat } }));
}

let polling = null;
function startPolling() {
  if (polling) return;
  const tick = async () => {
    try {
      const res = await fetch(boot.unreadUrl, { credentials: 'same-origin' });
      if (res.ok) render((await res.json()).unread_total || 0, null);
    } catch (e) { /* offline; try again next tick */ }
  };
  polling = setInterval(tick, 15000);
}

if (boot && badges.length) {
  connectChat(boot).then(db => {
    onValue(ref(db, `chat_unread/${boot.me}`), snap => {
      const perChat = snap.val() || {};
      const total = Object.values(perChat).reduce((sum, n) => sum + (Number(n) || 0), 0);
      render(total, perChat);
    }, startPolling);
  }).catch(startPolling);
}
