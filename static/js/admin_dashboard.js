// Admin dashboard (templates/admin_dashboard.html): critical banner dismiss, Pending Actions tabs,
// and the risk distribution donut (Chart.js, loaded only when there is data).
(function () {
  // ----- Critical alerts banner -----
  // The server remembers the dismissal in the admin's session (for the current count, so a new
  // critical patient shows it again). Without JS the form posts and the page reloads.
  const dismissForm = document.getElementById('criticalAlertForm');
  if (dismissForm) {
    dismissForm.addEventListener('submit', (e) => {
      e.preventDefault();
      document.getElementById('criticalAlert').hidden = true;
      fetch(dismissForm.action, { method: 'POST', body: new FormData(dismissForm), credentials: 'same-origin' })
        .catch(() => { /* banner simply returns on the next load */ });
    });
  }

  // ----- Pending Actions tabs (arrow keys move between tabs, per the ARIA tabs pattern) -----
  const tabs = Array.from(document.querySelectorAll('.ad-tabs [role="tab"]'));
  function select(tab, focus) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute('aria-selected', on);
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    });
    if (focus) tab.focus();
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', (e) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      select(tabs[(i + step + tabs.length) % tabs.length], true);
    });
  });

  // ----- Risk distribution donut -----
  const canvas = document.getElementById('riskDonut');
  const dataEl = document.getElementById('riskData');
  if (!canvas || !dataEl || typeof Chart === 'undefined') return;   // no data, or CDN unreachable: legend still shows the counts
  const rows = JSON.parse(dataEl.textContent);
  new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: rows.map((r) => `${r.label} (${r.range})`),
      datasets: [{
        data: rows.map((r) => r.count),
        backgroundColor: rows.map((r) => r.color),
        borderColor: '#FFFFFF',
        borderWidth: 3,
        hoverOffset: 6
      }]
    },
    options: {
      cutout: '68%',
      maintainAspectRatio: false,
      animation: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? false : { duration: 600 },
      plugins: {
        legend: { display: false },   // the HTML legend below the chart lists every band with its count
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
              const n = ctx.parsed;
              return ` ${n} patient${n === 1 ? '' : 's'} (${Math.round((n / total) * 100)}%)`;
            }
          }
        }
      }
    }
  });
})();
