// One patient <-> doctor conversation (templates/_chat_panel.html).
// Reads messages live from Firebase (chats/{id}/messages); sends and read receipts go through Flask.
// If the browser isn't allowed to read Firebase (database.rules.json not deployed), it falls back to
// polling /api/chat/messages, which Flask access-checks the same way.
import { connectChat, formatTime, dayLabel, sameDay } from './chat_firebase.js';
import { ref, get, query, orderByKey, startAfter, onChildAdded, onChildChanged } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-database.js";

const C = window.CHAT;
const boot = window.CHAT_BOOT;
const $ = (id) => document.getElementById(id);
const list = $('chatMessages'), form = $('chatForm'), input = $('chatInput'), sendBtn = $('chatSend');
const pill = $('chatNewPill'), notice = $('chatNotice'), countEl = $('chatCount'), muteBtn = $('chatMute');
const bubbleTpl = $('chatBubbleTpl'), dayTpl = $('chatDayTpl');

const GROUP_GAP = 5 * 60 * 1000;   // consecutive messages within 5 min are grouped
const messages = [];               // sorted by id (Firebase push ids sort chronologically)
const byId = new Map();            // id -> {msg, el}
let loaded = false, accessLost = false, unseenBelow = 0;
const pendingRead = new Set();

// ---------- rendering ----------
function isMine(m) { return m.sender_uid === C.me; }

function bubbleFor(m) {
  const row = bubbleTpl.content.firstElementChild.cloneNode(true);
  row.dataset.id = m.id;
  row.classList.add(isMine(m) ? 'chat-row-mine' : 'chat-row-theirs');
  row.querySelector('.chat-text').textContent = m.text;
  const when = m.timestamp ? new Date(m.timestamp) : new Date();
  const time = row.querySelector('.chat-time');
  time.textContent = formatTime(when);
  time.dateTime = when.toISOString();
  time.title = when.toLocaleString();
  updateTicks(row, m);
  return row;
}

function updateTicks(row, m) {
  if (!isMine(m)) return;
  const seen = Boolean((m.read_by || {})[C.other_uid]);
  row.querySelector('.chat-ticks').textContent = seen ? '✓✓' : '✓';
  row.querySelector('.chat-ticks').classList.toggle('is-seen', seen);
  row.querySelector('.chat-status').textContent = seen ? 'Seen' : 'Sent';
}

function dayFor(m) {
  const el = dayTpl.content.firstElementChild.cloneNode(true);
  el.querySelector('span').textContent = dayLabel(new Date(m.timestamp || Date.now()));
  return el;
}

function groupWithPrevious(prev, m) {
  return prev && prev.sender_uid === m.sender_uid && m.timestamp - prev.timestamp < GROUP_GAP
    && sameDay(new Date(prev.timestamp), new Date(m.timestamp));
}

function appendRendered(m, prev) {
  if (!prev || !sameDay(new Date(prev.timestamp || Date.now()), new Date(m.timestamp || Date.now()))) list.appendChild(dayFor(m));
  const el = bubbleFor(m);
  if (groupWithPrevious(prev, m)) el.classList.add('chat-row-cont');
  list.appendChild(el);
  return el;
}

function rerenderAll() {
  list.querySelectorAll('.chat-row, .chat-day').forEach(el => el.remove());
  messages.forEach((m, i) => { byId.get(m.id).el = appendRendered(m, messages[i - 1]); });
}

function addMessage(m, { animate = true } = {}) {
  const existing = byId.get(m.id);
  if (existing) {                       // e.g. the send response and the listener both deliver it
    existing.msg.read_by = { ...(existing.msg.read_by || {}), ...(m.read_by || {}) };
    updateTicks(existing.el, existing.msg);
    return false;
  }
  const atBottom = nearBottom();
  const last = messages[messages.length - 1];
  const entry = { msg: m, el: null };
  byId.set(m.id, entry);
  if (!last || m.id > last.id) {
    messages.push(m);
    entry.el = appendRendered(m, last);
  } else {                              // arrived out of order: insert and redraw
    messages.push(m);
    messages.sort((a, b) => (a.id < b.id ? -1 : 1));
    rerenderAll();
  }
  if (!animate) entry.el.classList.add('no-anim');
  $('chatEmpty').hidden = true;

  if (!isMine(m) && !(m.read_by || {})[C.me]) pendingRead.add(m.id);
  if (loaded) {
    if (isMine(m) || atBottom) scrollToBottom();
    else { unseenBelow++; showPill(); }
    if (!isMine(m)) playSound();
  }
  flushRead();
  return true;
}

function updateMessage(m) {
  const entry = byId.get(m.id);
  if (!entry) return;
  entry.msg.read_by = m.read_by || {};
  updateTicks(entry.el, entry.msg);
}

function finishInitialLoad() {
  loaded = true;
  $('chatLoading').hidden = true;
  $('chatEmpty').hidden = messages.length > 0;
  scrollToBottom();
  flushRead();
}

// ---------- scrolling ----------
function nearBottom() { return list.scrollHeight - list.scrollTop - list.clientHeight < 80; }
// Instant, not smooth: a smooth scroll still in progress would make the next message of a
// burst think the reader had scrolled away and stop following
function scrollToBottom() {
  list.scrollTop = list.scrollHeight;
  unseenBelow = 0; pill.hidden = true;
  flushRead();
}
function showPill() {
  pill.textContent = unseenBelow > 1 ? `${unseenBelow} new messages ↓` : 'New message ↓';
  pill.hidden = false;
}
pill.addEventListener('click', () => scrollToBottom());
list.addEventListener('scroll', () => { if (nearBottom()) { unseenBelow = 0; pill.hidden = true; flushRead(); } });

// ---------- read receipts ----------
let readTimer = null;
function flushRead() {
  clearTimeout(readTimer);
  readTimer = setTimeout(async () => {
    // Only count as read when the person can actually see the newest messages
    if (!pendingRead.size || accessLost || document.visibilityState !== 'visible' || !nearBottom()) return;
    const ids = [...pendingRead];
    pendingRead.clear();
    try {
      const res = await fetch(C.readUrl, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'same-origin',
        body: JSON.stringify({ chat_id: C.id, message_ids: ids })
      });
      if (res.status === 403) lostAccess();
      else if (!res.ok) ids.forEach(id => pendingRead.add(id));
    } catch (e) { ids.forEach(id => pendingRead.add(id)); }
  }, 250);
}
document.addEventListener('visibilitychange', flushRead);
window.addEventListener('focus', flushRead);

// ---------- sound ----------
let muted = false;
try { muted = localStorage.getItem('chatMuted') === '1'; } catch (e) { /* storage blocked */ }
function renderMute() {
  muteBtn.textContent = muted ? '🔕' : '🔔';
  muteBtn.setAttribute('aria-pressed', String(muted));
  muteBtn.title = muted ? 'Unmute notification sound' : 'Mute notification sound';
}
renderMute();
muteBtn.addEventListener('click', () => {
  muted = !muted;
  try { localStorage.setItem('chatMuted', muted ? '1' : '0'); } catch (e) { /* ignore */ }
  renderMute();
});
let audio = null;
function playSound() {
  if (muted) return;
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    const osc = audio.createOscillator(), gain = audio.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, audio.currentTime);
    osc.frequency.exponentialRampToValueAtTime(660, audio.currentTime + 0.12);
    gain.gain.setValueAtTime(0.0001, audio.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.08, audio.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.25);
    osc.connect(gain).connect(audio.destination);
    osc.start(); osc.stop(audio.currentTime + 0.26);
  } catch (e) { /* audio unavailable */ }
}

// ---------- composer ----------
function syncComposer() {
  const len = input.value.length;
  sendBtn.disabled = accessLost || !input.value.trim();
  countEl.hidden = len < 1800;
  countEl.textContent = `${len}/2000`;
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
}
input.addEventListener('input', syncComposer);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!sendBtn.disabled) form.requestSubmit();
  }
});

function showNotice(text, kind = 'error') {
  notice.textContent = text;
  notice.className = `chat-notice chat-notice-${kind}`;
  notice.hidden = !text;
}

function lostAccess() {
  if (accessLost) return;
  accessLost = true;
  stopPolling();
  showNotice("You no longer have access to this conversation. The patient may have been assigned to another doctor.");
  input.disabled = true;
  input.placeholder = 'Conversation closed';
  syncComposer();
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || accessLost) return;
  sendBtn.disabled = true;
  showNotice('');
  try {
    const res = await fetch(C.sendUrl, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'same-origin',
      body: JSON.stringify({ chat_id: C.id, text })
    });
    const data = await res.json().catch(() => ({}));
    if (res.status === 403) { lostAccess(); return; }
    if (!res.ok || !data.success) throw new Error(data.error || 'Message not sent');
    input.value = '';
    addMessage(data.message);
    scrollToBottom();
  } catch (err) {
    showNotice(`${err.message}. Please try again.`);
  } finally {
    syncComposer();
    input.focus();
  }
});

// ---------- data: Firebase live, or Flask polling ----------
let pollTimer = null;
function stopPolling() { clearInterval(pollTimer); pollTimer = null; }
async function poll() {
  const after = messages.length ? messages[messages.length - 1].id : '';
  const res = await fetch(`${C.pollUrl}?chat_id=${encodeURIComponent(C.id)}&after=${encodeURIComponent(after)}`, { credentials: 'same-origin' });
  if (res.status === 403) { lostAccess(); return; }
  if (!res.ok) return;
  (await res.json()).messages.forEach(m => addMessage(m, { animate: loaded }));
}
async function startPolling() {
  if (pollTimer || accessLost) return;
  try { await poll(); } catch (e) { /* retry below */ }
  if (!loaded) finishInitialLoad();
  pollTimer = setInterval(() => poll().catch(() => {}), 4000);
}

async function startLive() {
  const db = await connectChat(boot);
  const messagesRef = ref(db, `chats/${C.id}/messages`);
  const snap = await get(messagesRef);            // history first, so we can scroll once
  snap.forEach(child => { addMessage({ id: child.key, ...child.val() }, { animate: false }); });
  finishInitialLoad();
  const lastKey = messages.length ? messages[messages.length - 1].id : null;
  const newOnes = lastKey ? query(messagesRef, orderByKey(), startAfter(lastKey)) : messagesRef;
  // A cancelled listener means access was revoked (e.g. reassignment); polling confirms with a 403
  onChildAdded(newOnes, child => addMessage({ id: child.key, ...child.val() }), () => startPolling());
  onChildChanged(messagesRef, child => updateMessage({ id: child.key, ...child.val() }), () => {});
}

syncComposer();
startLive().catch(() => startPolling());
