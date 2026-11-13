// Activity Logs page (templates/admin_activity_logs.html).
// Live Mode is polling: every 10 seconds it re-requests this page with the same filters and swaps in
// the results, keeping expanded detail rows open and highlighting entries that weren't there before.
(function () {
  const form = document.getElementById('logFilters');
  // Dropdowns (including the date range in the header, tied to the form by form="logFilters") apply at once
  document.querySelectorAll('#logFilters select, select[form="logFilters"]').forEach((select) => {
    select.addEventListener('change', () => form.submit());
  });

  const toggle = document.getElementById('liveToggle');
  const KEY = 'adminLogsLive';
  const INTERVAL = 10000;
  let timer = null;

  function stamp(text) {
    const el = document.getElementById('liveUpdated');
    if (el) el.textContent = text;
  }

  async function refresh() {
    if (document.hidden) return;   // don't poll a background tab
    const results = document.getElementById('logResults');
    const before = new Set(Array.from(results.querySelectorAll('tr[data-id]')).map((r) => r.dataset.id));
    const open = new Set(Array.from(results.querySelectorAll('tr[data-id] details[open]')).map((d) => d.closest('tr').dataset.id));
    try {
      const res = await fetch(location.href, { credentials: 'same-origin', headers: { 'X-Requested-With': 'fetch' } });
      if (!res.ok || res.redirected) throw new Error('signed out');
      const doc = new DOMParser().parseFromString(await res.text(), 'text/html');
      const fresh = doc.getElementById('logResults');
      if (!fresh) throw new Error('unexpected response');
      results.innerHTML = fresh.innerHTML;
      results.querySelectorAll('tr[data-id]').forEach((row) => {
        if (open.has(row.dataset.id)) row.querySelector('details')?.setAttribute('open', '');
        if (before.size && !before.has(row.dataset.id)) row.classList.add('is-new');
      });
      stamp(` · updated ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`);
    } catch (err) {
      setLive(false);
      stamp(' · live mode stopped: refresh the page (you may have been signed out)');
    }
  }

  function setLive(on) {
    toggle.setAttribute('aria-checked', String(on));
    toggle.classList.toggle('is-on', on);
    toggle.querySelector('.lg-live-text').textContent = on ? 'Live' : 'Live Mode';
    clearInterval(timer);
    timer = on ? setInterval(refresh, INTERVAL) : null;
    try { localStorage.setItem(KEY, on ? '1' : '0'); } catch (e) { /* storage blocked: just not remembered */ }
  }

  toggle.addEventListener('click', () => {
    const on = toggle.getAttribute('aria-checked') !== 'true';
    setLive(on);
    if (on) refresh();
  });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && toggle.classList.contains('is-on')) refresh();   // catch up when the tab returns
  });

  let remembered = false;
  try { remembered = localStorage.getItem(KEY) === '1'; } catch (e) { /* ignore */ }
  if (remembered) setLive(true);
})();
