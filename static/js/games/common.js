// Shared helpers for the wellness games (templates/games/*.html).
// Game.complete() records a finished session: POST /api/games/complete -> game_stats + activity log.
(function () {
  const config = window.GAME_CONFIG || {};
  const toastEl = document.getElementById('gmToast');
  let toastTimer;

  const Game = {
    key: config.key,
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    startedAt: null,

    start() { this.startedAt = performance.now(); },
    elapsed() { return this.startedAt ? (performance.now() - this.startedAt) / 1000 : 0; },

    formatTime(seconds) {
      const s = Math.max(0, Math.round(seconds));
      return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    },

    toast(message) {
      if (!toastEl) return;
      toastEl.textContent = message;
      toastEl.hidden = false;
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => { toastEl.hidden = true; }, 3200);
    },

    // extra: { score, level } (both optional). Resolves to the server reply, or null if it couldn't be saved.
    async complete(extra) {
      const body = Object.assign({ game: this.key, duration: Math.round(this.elapsed()) }, extra || {});
      try {
        const res = await fetch(config.completeUrl, {
          method: 'POST', credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || !data.success) throw new Error(data.error || 'not saved');
        return data;
      } catch (err) {
        this.toast("Couldn't save this session, but well done anyway!");
        return null;
      }
    },

    shuffle(list) {
      for (let i = list.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [list[i], list[j]] = [list[j], list[i]];
      }
      return list;
    }
  };
  window.Game = Game;
})();
