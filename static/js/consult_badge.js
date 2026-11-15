// Keeps the doctor's sidebar "Consultations" badge current (pending count; red + pulsing when an
// emergency request is pending). Polls Flask once a minute and when the tab becomes visible again.
(function () {
  const badges = document.querySelectorAll('[data-consult-badge]');
  const url = document.currentScript && document.currentScript.dataset.url;
  if (!badges.length || !url) return;

  function render(data) {
    badges.forEach((b) => {
      b.textContent = data.pending > 99 ? '99+' : String(data.pending);
      b.hidden = !data.pending;
      b.classList.toggle('is-emergency', data.emergency > 0);
      b.title = data.emergency ? `${data.emergency} emergency request${data.emergency === 1 ? '' : 's'} waiting` : 'Pending consultation requests';
    });
  }

  async function refresh() {
    if (document.visibilityState !== 'visible') return;
    try {
      const res = await fetch(url, { credentials: 'same-origin' });
      if (res.ok) render(await res.json());
    } catch (e) { /* offline: keep the last value */ }
  }

  window.refreshConsultBadge = refresh;
  setInterval(refresh, 60000);
  document.addEventListener('visibilitychange', refresh);
})();
