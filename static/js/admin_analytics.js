// Admin analytics (templates/admin_analytics.html): custom range toggle and the four Chart.js charts.
// Chart data is computed server-side (admin_analytics_data in app.py) and embedded as JSON.
(function () {
  // ----- Custom range: toggle the date picker in place instead of reloading -----
  const toggle = document.getElementById('customToggle');
  const form = document.getElementById('customRange');
  if (toggle && form) {
    toggle.addEventListener('click', (e) => {
      e.preventDefault();
      form.hidden = !form.hidden;
      toggle.setAttribute('aria-expanded', String(!form.hidden));
      if (!form.hidden) form.querySelector('input[type="date"]').focus();
    });
  }

  const dataEl = document.getElementById('analyticsData');
  if (!dataEl || typeof Chart === 'undefined') return;   // no chart data, or the CDN is unreachable
  const data = JSON.parse(dataEl.textContent);
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  Chart.defaults.font.family = "'Inter', 'Plus Jakarta Sans', system-ui, sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.color = '#64748B';

  const grid = { color: '#EEF2F6', drawTicks: false };
  const axis = (title, extra) => Object.assign({
    beginAtZero: true,
    grid: grid,
    border: { display: false },
    ticks: { padding: 8, precision: 0 },
    title: { display: Boolean(title), text: title, color: '#94A3B8', font: { size: 11, weight: '600' } }
  }, extra || {});
  const xAxis = { grid: { display: false }, border: { color: '#E2E8F0' }, ticks: { padding: 6, maxRotation: 0, autoSkipPadding: 12 } };
  const tooltip = {
    backgroundColor: '#0F172A', padding: 10, cornerRadius: 8, boxPadding: 4,
    titleFont: { weight: '700' }, bodyFont: { size: 12 }
  };
  const base = (extra) => Object.assign({
    responsive: true,
    maintainAspectRatio: false,
    animation: reduceMotion ? false : { duration: 600 },
    interaction: { mode: 'index', intersect: false }
  }, extra);
  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

  function draw(id, config) {
    const canvas = document.getElementById(id);
    if (canvas) new Chart(canvas, config);
  }

  // 1. Cumulative registrations: smooth filled green area
  draw('chartRegistrations', {
    type: 'line',
    data: {
      labels: data.registrations.labels,
      datasets: [{
        label: 'Patients',
        data: data.registrations.values,
        borderColor: '#0F7A5F',
        borderWidth: 2.5,
        tension: 0.35,
        fill: true,
        backgroundColor: (ctx) => {
          const { chart } = ctx;
          if (!chart.chartArea) return 'rgba(15, 122, 95, 0.12)';
          const g = chart.ctx.createLinearGradient(0, chart.chartArea.top, 0, chart.chartArea.bottom);
          g.addColorStop(0, 'rgba(15, 122, 95, 0.28)');
          g.addColorStop(1, 'rgba(15, 122, 95, 0.02)');
          return g;
        },
        pointRadius: 3,
        pointHoverRadius: 6,
        pointBackgroundColor: '#FFFFFF',
        pointBorderColor: '#0F7A5F',
        pointBorderWidth: 2
      }]
    },
    options: base({
      plugins: {
        legend: { display: false },
        tooltip: Object.assign({}, tooltip, { callbacks: { label: (c) => ` ${plural(c.parsed.y, 'patient')} registered` } })
      },
      scales: { x: xAxis, y: axis('Patients') }
    })
  });

  // 2. Risk distribution per month, stacked (Critical on top)
  draw('chartRisk', {
    type: 'bar',
    data: {
      labels: data.risk.labels,
      // Low first so it sits at the bottom of each stack
      datasets: data.risk.series.slice().reverse().map((s) => ({
        label: s.label,
        data: s.values,
        backgroundColor: s.color,
        borderRadius: 3,
        borderSkipped: false,
        maxBarThickness: 28
      }))
    },
    options: base({
      plugins: {
        legend: { position: 'bottom', reverse: true, labels: { usePointStyle: true, pointStyle: 'rectRounded', boxWidth: 10, padding: 14 } },
        tooltip: Object.assign({}, tooltip, {
          itemSort: (a, b) => b.datasetIndex - a.datasetIndex,
          callbacks: {
            label: (c) => ` ${c.dataset.label}: ${plural(c.parsed.y, 'assessment')}`,
            footer: (items) => `Total: ${items.reduce((n, i) => n + i.parsed.y, 0)}`
          }
        })
      },
      scales: { x: Object.assign({ stacked: true }, xAxis), y: axis('Assessments', { stacked: true }) }
    })
  });

  // 3. Daily completions
  draw('chartDaily', {
    type: 'bar',
    data: {
      labels: data.daily.labels,
      datasets: [{
        label: 'Assessments',
        data: data.daily.values,
        backgroundColor: '#0F7A5F',
        hoverBackgroundColor: '#0A5A47',
        borderRadius: 6,
        maxBarThickness: 26
      }]
    },
    options: base({
      plugins: {
        legend: { display: false },
        tooltip: Object.assign({}, tooltip, { callbacks: { label: (c) => ` ${plural(c.parsed.y, 'assessment')} completed` } })
      },
      scales: { x: xAxis, y: axis('Assessments') }
    })
  });

  // 4. Alert response times with the target line
  const fmtMin = (m) => (m < 60 ? `${m.toFixed(1)} min` : m < 1440 ? `${(m / 60).toFixed(1)} h` : `${(m / 1440).toFixed(1)} days`);
  draw('chartResponse', {
    type: 'line',
    data: {
      labels: data.response.labels,
      datasets: [{
        label: 'Avg response',
        data: data.response.values,
        borderColor: '#2563EB',
        backgroundColor: '#2563EB',
        borderWidth: 2.5,
        tension: 0.3,
        spanGaps: true,   // days without acknowledged alerts have no average
        pointRadius: 4,
        pointHoverRadius: 6
      }, {
        label: `Target (${data.response.target} min)`,
        data: data.response.labels.map(() => data.response.target),
        borderColor: '#DC2626',
        borderWidth: 1.5,
        borderDash: [6, 5],
        pointRadius: 0,
        pointHoverRadius: 0
      }]
    },
    options: base({
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true, pointStyle: 'line', padding: 14 } },
        tooltip: Object.assign({}, tooltip, {
          filter: (i) => i.datasetIndex === 0 && i.parsed.y !== null,
          callbacks: { label: (c) => ` Avg response: ${fmtMin(c.parsed.y)}` }
        })
      },
      scales: { x: xAxis, y: axis('Minutes', { ticks: { padding: 8 } }) }
    })
  });
})();
