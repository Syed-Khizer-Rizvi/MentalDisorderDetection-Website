// Client-side filter, sort and CSV export for the Assessment Reports page (templates/doctor_reports.html).
// Each table row carries its values as data attributes, so nothing needs a page reload.
(function () {
  const table = document.getElementById('reportTable');
  if (!table) return;
  const tbody = table.tBodies[0];
  const rows = [...tbody.rows];
  const patient = document.getElementById('filterPatient');
  const risk = document.getElementById('filterRisk');
  const from = document.getElementById('filterFrom');
  const sortSelect = document.getElementById('sortSelect');
  const count = document.getElementById('reportCount');
  const wrap = document.getElementById('reportTableWrap');
  const noMatch = document.getElementById('reportNoMatch');
  const exportBtn = document.getElementById('exportCsv');

  const RISK_RANK = { low: 0, moderate: 1, high: 2, critical: 3 };
  const number = (v) => (v === '' ? -1 : Number(v));   // a missing score sorts below a real 0
  const SORT_VALUE = {
    name: (r) => r.dataset.name.toLowerCase(),
    created: (r) => Number(r.dataset.created),
    phq: (r) => number(r.dataset.phq),
    gad: (r) => number(r.dataset.gad),
    pss: (r) => number(r.dataset.pss),
    composite: (r) => Number(r.dataset.composite),
    risk: (r) => RISK_RANK[r.dataset.risk],
    mood: (r) => r.dataset.mood.toLowerCase(),
  };
  const TEXT_KEYS = ['name', 'mood'];   // these start A-Z on first click; numbers start highest first
  let sort = { key: 'created', dir: 'desc' };

  function visibleRows() {
    return [...tbody.rows].filter((r) => !r.hidden);
  }

  function applyFilters() {
    let shown = 0;
    rows.forEach((r) => {
      const match = (!patient.value || r.dataset.patient === patient.value)
        && (!risk.value || r.dataset.risk === risk.value)
        && (!from.value || r.dataset.date >= from.value);   // both are YYYY-MM-DD
      r.hidden = !match;
      if (match) shown += 1;
    });
    count.textContent = `Showing ${shown} of ${rows.length}`;
    wrap.hidden = shown === 0;
    noMatch.hidden = shown !== 0;
    exportBtn.disabled = shown === 0;
  }

  function applySort() {
    const value = SORT_VALUE[sort.key];
    const direction = sort.dir === 'asc' ? 1 : -1;
    const created = SORT_VALUE.created;
    rows.slice().sort((a, b) => {
      const x = value(a), y = value(b);
      const order = typeof x === 'string' ? x.localeCompare(y) : x - y;
      return direction * order || created(b) - created(a);   // ties: newest first
    }).forEach((r) => tbody.appendChild(r));

    table.querySelectorAll('th').forEach((th) => {
      const btn = th.querySelector('[data-sort]');
      if (btn && btn.dataset.sort === sort.key) {
        th.setAttribute('aria-sort', sort.dir === 'asc' ? 'ascending' : 'descending');
      } else {
        th.removeAttribute('aria-sort');
      }
    });
    sortSelect.value = `${sort.key}:${sort.dir}`;
  }

  table.querySelectorAll('[data-sort]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.sort;
      if (sort.key === key) {
        sort.dir = sort.dir === 'asc' ? 'desc' : 'asc';
      } else {
        sort = { key, dir: TEXT_KEYS.includes(key) ? 'asc' : 'desc' };
      }
      applySort();
    });
  });

  // The table header is hidden on narrow screens, where this dropdown takes over
  sortSelect.addEventListener('change', () => {
    const [key, dir] = sortSelect.value.split(':');
    sort = { key, dir };
    applySort();
  });

  [patient, risk, from].forEach((el) => el.addEventListener('input', applyFilters));
  document.getElementById('reportFilters').addEventListener('submit', (e) => e.preventDefault());
  document.getElementById('filterReset').addEventListener('click', () => {
    patient.value = risk.value = from.value = '';
    applyFilters();
  });

  function csvCell(value) {
    let text = String(value);
    // Patient names are user-entered: stop a spreadsheet running one as a formula
    if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  }

  // Exports the rows on screen, in the order on screen
  exportBtn.addEventListener('click', () => {
    const header = ['Patient', 'Date', 'PHQ-9', 'GAD-7', 'PSS', 'Composite', 'Risk', 'Facial Mood'];
    const lines = visibleRows().map((r) => {
      const d = r.dataset;
      return [d.name, d.date, d.phq, d.gad, d.pss, d.composite, d.riskLabel, d.mood].map(csvCell).join(',');
    });
    const csv = '﻿' + [header.map(csvCell).join(','), ...lines].join('\r\n');   // BOM: Excel reads it as UTF-8
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');   // local date: toISOString() would give the UTC day
    link.download = `assessment_reports_${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  });

  applySort();
  applyFilters();   // the browser may restore filter values on Back
})();
