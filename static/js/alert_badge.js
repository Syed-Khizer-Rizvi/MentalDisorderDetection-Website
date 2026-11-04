// Keeps the doctor's sidebar "Risk Alerts" badge current. Alerts are derived on the server from
// assessments, so this polls Flask (once a minute, and when the tab becomes visible again).
(function () {
  const badges = document.querySelectorAll('[data-alert-badge]');
  const url = document.currentScript && document.currentScript.dataset.url;
  if (!badges.length || !url) return;

  function render(n) {
    badges.forEach(b => { b.textContent = n > 99 ? '99+' : String(n); b.hidden = !n; });
    document.dispatchEvent(new CustomEvent('alert-unread', { detail: n }));
  }

  async function refresh() {
    if (document.visibilityState !== 'visible') return;
    try {
      const res = await fetch(url, { credentials: 'same-origin' });
      if (res.ok) render((await res.json()).unread_total || 0);
    } catch (e) { /* offline: keep the last value */ }
  }

  window.refreshAlertBadge = refresh;
  setInterval(refresh, 60000);
  document.addEventListener('visibilitychange', refresh);
})();
