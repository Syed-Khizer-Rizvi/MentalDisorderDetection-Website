// Shared Firebase connection for the chat pages and the sidebar badge (ES module).
// The browser only *reads* chats/{id} and chat_unread/{uid}; Flask does every write.
// It signs in with a custom token from /api/chat/token so the Firebase user always matches the
// Flask session, which is what database.rules.json checks.
import { initializeApp, getApps, getApp } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-app.js";
import { getAuth, signInWithCustomToken, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-auth.js";
import { getDatabase } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-database.js";

let connection = null;

export function connectChat(boot) {
  if (!connection) {
    connection = (async () => {
      const app = getApps().length ? getApp() : initializeApp(boot.config);
      const auth = getAuth(app);
      // Wait for the persisted session to load before deciding whether to sign in again
      await new Promise(resolve => { const stop = onAuthStateChanged(auth, () => { stop(); resolve(); }); });
      if (!auth.currentUser || auth.currentUser.uid !== boot.me) {
        const res = await fetch(boot.tokenUrl, { credentials: 'same-origin' });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Could not start chat');
        await signInWithCustomToken(auth, data.token);
      }
      return getDatabase(app);
    })();
  }
  return connection;
}

// Same wording as format_chat_timestamp() in app.py
export function formatChatTimestamp(ts, now = new Date()) {
  if (!ts) return '';
  const when = new Date(ts);
  const seconds = (now - when) / 1000;
  if (seconds >= 0 && seconds < 60) return 'Just now';
  if (seconds >= 0 && seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (sameDay(when, now)) return formatTime(when);
  const yesterday = new Date(now); yesterday.setDate(now.getDate() - 1);
  if (sameDay(when, yesterday)) return 'Yesterday';
  return formatDay(when, now);
}

export function formatTime(d) {
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

export function formatDay(d, now = new Date()) {
  const opts = { month: 'short', day: 'numeric' };
  if (d.getFullYear() !== now.getFullYear()) opts.year = 'numeric';
  return d.toLocaleDateString('en-US', opts);
}

export function dayLabel(d, now = new Date()) {
  if (sameDay(d, now)) return 'Today';
  const yesterday = new Date(now); yesterday.setDate(now.getDate() - 1);
  if (sameDay(d, yesterday)) return 'Yesterday';
  return formatDay(d, now);
}

export function sameDay(a, b) {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
}
