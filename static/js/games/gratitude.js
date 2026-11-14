// Gratitude Jar: notes are saved to users/{uid}/gratitude (private) via /api/gratitude/*.
// Each note's colour, tilt and place in the jar come from its id, so the jar looks the same every visit.
(function () {
  const COLORS = ['#FDBA74', '#FCA5A5', '#F9A8D4', '#C4B5FD', '#93C5FD', '#6EE7B7', '#FDE68A', '#A5F3FC'];
  const PER_ROW = 6, ROWS = 7, NOTE = 30;
  const urls = window.GRATITUDE_URLS;
  const $ = (id) => document.getElementById(id);
  const notesEl = $('gjNotes');
  let entries = [];

  function hash(str) { let h = 0; for (const ch of str) h = (h * 31 + ch.charCodeAt(0)) >>> 0; return h; }

  function noteButton(entry, slot, falling) {
    const h = hash(entry.id);
    const area = notesEl.getBoundingClientRect();
    const width = area.width || 192, height = area.height || 222;
    const col = slot % PER_ROW, row = Math.floor(slot / PER_ROW);
    const step = (width - NOTE) / (PER_ROW - 1);
    const x = col * step + ((h % 7) - 3);
    const y = height - NOTE - row * (NOTE - 2) - (h % 5);
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'gj-note' + (falling && !Game.reducedMotion ? ' is-falling' : '');
    b.style.left = `${Math.max(0, x)}px`;
    b.style.top = `${Math.max(0, y)}px`;
    b.style.setProperty('--c', COLORS[h % COLORS.length]);
    b.style.setProperty('--r', `${(h % 41) - 20}deg`);
    b.setAttribute('aria-label', `Gratitude note from ${entry.date}`);
    b.addEventListener('click', () => openNote(entry));
    return b;
  }

  function render(newId) {
    notesEl.innerHTML = '';
    const capacity = PER_ROW * ROWS;
    const shown = entries.slice(-capacity);   // the newest notes sit on top
    shown.forEach((e, i) => notesEl.appendChild(noteButton(e, i, e.id === newId)));
    const hiddenCount = entries.length - shown.length;
    $('gjMore').textContent = hiddenCount > 0 ? `+${hiddenCount} more at the bottom of the jar` : '';
    $('gjViewAll').disabled = !entries.length;
  }

  function setStats(stats) {
    $('gjMessage').textContent = stats.message + (stats.count ? ` · ${stats.count} note${stats.count === 1 ? '' : 's'}` : '');
  }

  function openNote(entry) {
    $('gjNoteDate').textContent = entry.date;
    $('gjNoteText').textContent = entry.text;
    $('gjNoteDialog').showModal();
  }

  $('gjViewAll').addEventListener('click', () => {
    const list = $('gjList');
    list.innerHTML = '';
    [...entries].reverse().forEach((e) => {
      const li = document.createElement('li');
      li.style.setProperty('--c', COLORS[hash(e.id) % COLORS.length]);
      const p = document.createElement('p'); p.textContent = e.text;
      const t = document.createElement('time'); t.textContent = e.date;
      li.append(p, t);
      list.appendChild(li);
    });
    $('gjAllDialog').showModal();
  });
  document.querySelectorAll('.gm-dialog').forEach((d) => {
    d.querySelector('[data-close]').addEventListener('click', () => d.close());
    d.addEventListener('click', (e) => { if (e.target === d) d.close(); });
  });

  const text = $('gjText');
  text.addEventListener('input', () => { $('gjCount').textContent = text.value.length; });
  $('gjForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const value = text.value.trim();
    const err = $('gjError');
    err.hidden = true;
    if (!value) { err.textContent = "Write something you're grateful for first."; err.hidden = false; text.focus(); return; }
    const btn = $('gjAdd');
    btn.disabled = true;
    try {
      const res = await fetch(urls.add, { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
                                          body: JSON.stringify({ text: value }) });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.error || 'Could not save your note.');
      entries.push(data.entry);
      render(data.entry.id);
      setStats(data.stats);
      text.value = '';
      $('gjCount').textContent = '0';
      Game.toast('Added to your jar 🌼');
    } catch (ex) {
      err.textContent = ex.message;
      err.hidden = false;
    } finally {
      btn.disabled = false;
    }
  });

  (async function load() {
    try {
      const res = await fetch(urls.list, { credentials: 'same-origin' });
      const data = await res.json();
      if (!data.success) throw new Error();
      entries = data.entries;
      render();
      setStats(data.stats);
    } catch (e) {
      $('gjMessage').textContent = "Couldn't load your jar. Refresh to try again.";
    }
  })();
  window.addEventListener('resize', () => render());
})();
