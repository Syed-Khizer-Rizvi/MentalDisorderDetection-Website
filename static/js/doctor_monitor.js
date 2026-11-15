// Doctor live monitoring (templates/doctor_monitor.html): polls /api/doctor/monitor/<uid> every 2 s and
// redraws the vital cards, sparklines and the two trend charts (plain SVG, no chart library).
(function () {
  const POLL_MS = 2000;
  const config = document.querySelector('.mon-config');
  const $ = (id) => document.getElementById(id);
  const SVG = 'http://www.w3.org/2000/svg';

  let data = JSON.parse($('monInitial').textContent);
  let paused = false, timer = null, inFlight = false, failures = 0;
  let updates = 0;                 // new readings received since the page opened (or Reset)
  let since = 0;                   // Reset: only readings newer than this are charted
  let lastSeen = latestTime(data);

  function latestTime(d) {
    const r = d && d.readings;
    return r && r.length ? r[r.length - 1].t : 0;
  }
  function shown() { return (data.readings || []).filter((r) => r.t > since); }
  const fmtTime = (t) => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const fmt = (v, digits) => (v === null || v === undefined ? '—' : Number(v).toFixed(digits));

  // ---------- drawing ----------
  function linePath(values, w, h, pad, lo, hi) {
    const n = values.length;
    return values.map((v, i) => {
      const x = n === 1 ? w / 2 : (i / (n - 1)) * w;
      const y = pad + (1 - (v - lo) / (hi - lo || 1)) * (h - 2 * pad);
      return `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
  }

  function spark(svg, values) {
    const line = svg.querySelector('.mon-spark-line'), area = svg.querySelector('.mon-spark-area');
    if (values.length < 2) { line.setAttribute('d', ''); area.setAttribute('d', ''); return; }
    const lo = Math.min(...values), hi = Math.max(...values);
    const d = linePath(values, 120, 36, 4, lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.1);
    line.setAttribute('d', d);
    area.setAttribute('d', `${d} L120,36 L0,36 Z`);
  }

  function el(name, attrs, text) {
    const node = document.createElementNS(SVG, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // Full-width area chart with a y axis (fixed range when given, e.g. 0-100) and first/last times
  function chart(svg, points, key, fixed) {
    const box = svg.getBoundingClientRect();
    const W = Math.max(Math.round(box.width), 280), H = 220, L = 40, R = 12, T = 12, B = 26;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.replaceChildren();
    const values = points.map((p) => p[key]).filter((v) => typeof v === 'number');
    const usable = points.filter((p) => typeof p[key] === 'number');
    if (usable.length < 2) {
      svg.append(el('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle', class: 'mon-chart-empty' },
                    usable.length ? 'Waiting for more readings…' : 'No readings yet'));
      return;
    }
    let lo = fixed ? fixed[0] : Math.min(...values), hi = fixed ? fixed[1] : Math.max(...values);
    if (!fixed) { const pad = Math.max((hi - lo) * 0.15, 2); lo = Math.floor(lo - pad); hi = Math.ceil(hi + pad); }
    const pw = W - L - R, ph = H - T - B;
    const x = (i) => L + (i / (usable.length - 1)) * pw;
    const y = (v) => T + (1 - (v - lo) / (hi - lo || 1)) * ph;
    for (let g = 0; g <= 4; g++) {   // grid + y labels
      const v = lo + ((hi - lo) * g) / 4, gy = y(v);
      svg.append(el('line', { x1: L, x2: W - R, y1: gy, y2: gy, class: 'mon-grid' }));
      svg.append(el('text', { x: L - 8, y: gy + 4, 'text-anchor': 'end', class: 'mon-axis' }, Math.round(v)));
    }
    const d = usable.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`).join(' ');
    const gradId = svg.id + 'Grad';
    const defs = el('defs', {});
    const grad = el('linearGradient', { id: gradId, x1: 0, x2: 0, y1: 0, y2: 1 });
    grad.append(el('stop', { offset: '0%', class: 'mon-grad-top' }), el('stop', { offset: '100%', class: 'mon-grad-bottom' }));
    defs.append(grad);
    svg.append(defs,
      el('path', { d: `${d} L${x(usable.length - 1).toFixed(1)},${T + ph} L${L},${T + ph} Z`, fill: `url(#${gradId})` }),
      el('path', { d, class: 'mon-chart-line' }));
    const last = usable[usable.length - 1];
    svg.append(el('circle', { cx: x(usable.length - 1), cy: y(last[key]), r: 4.5, class: 'mon-chart-dot' }));
    svg.append(el('text', { x: L, y: H - 6, class: 'mon-axis' }, fmtTime(usable[0].t)),
               el('text', { x: W - R, y: H - 6, 'text-anchor': 'end', class: 'mon-axis' }, fmtTime(last.t)));
    const range = document.querySelector(`[data-range="${key}"]`);
    if (range) range.textContent = `${Math.min(...values).toFixed(0)}–${Math.max(...values).toFixed(0)} · ${usable.length} readings`;
  }

  function card(metric, value, digits, sub, series) {
    const c = document.querySelector(`.mon-card[data-metric="${metric}"]`);
    c.querySelector('[data-value]').textContent = fmt(value, digits);
    if (sub !== null) c.querySelector('[data-sub]').textContent = sub;
    spark(c.querySelector('.mon-spark'), series.filter((v) => typeof v === 'number'));
  }

  function render() {
    const connected = data.connected;
    $('monDisconnected').hidden = connected;
    $('monLive').hidden = !connected;
    if (data.status === 'none') $('monDisconnectedText').textContent =
      'This patient has not paired a PsyMonitor Band v2 yet. This page keeps checking and will update as soon as readings arrive.';
    const status = $('monStatus');
    status.dataset.status = paused ? 'paused' : data.status;
    $('monStatusText').textContent = paused ? 'Paused' : data.status_label;
    $('monUpdates').textContent = updates;
    $('monSync').textContent = data.last_sync ? `Last reading ${fmtTime(data.last_sync)} (${data.last_sync_ago})` : '';
    if (!connected) return;

    const pts = shown();
    const latest = since ? (pts[pts.length - 1] || null) : (data.latest || pts[pts.length - 1] || null);
    const v = (k) => (latest ? latest[k] : null);
    const series = (k) => pts.map((p) => p[k]);
    // Respiration only if the band reports it; PsyMonitor Band v2 measures skin temperature instead
    const hasResp = pts.some((p) => typeof p.resp === 'number');
    const second = document.querySelector('.mon-card[data-metric="second"]');
    second.querySelector('[data-title]').textContent = hasResp ? 'Respiration' : 'Skin Temperature';
    second.querySelector('[data-unit]').textContent = hasResp ? 'br/min' : '°C';
    second.querySelector('[data-icon]').textContent = hasResp ? '🫁' : '🌡';

    card('hr', v('hr'), 0, latest ? `Updated ${fmtTime(latest.t)}` : 'Waiting for the next reading…', series('hr'));
    card('second', hasResp ? v('resp') : v('temp'), hasResp ? 0 : 1,
         hasResp ? null : 'Respiration is not measured by this band', series(hasResp ? 'resp' : 'temp'));
    card('gsr', v('gsr'), 0, null, series('gsr'));
    card('risk', v('risk'), 0, latest && latest.risk_label ? `${latest.risk_label} · stress indicator from HR + GSR` : null, series('risk'));
    const riskValue = document.querySelector('.mon-card[data-metric="risk"] [data-value]');
    riskValue.style.color = latest && latest.risk_color ? latest.risk_color : '';
    chart($('chartHr'), pts, 'hr', null);
    chart($('chartRisk'), pts, 'risk', [0, 100]);
  }

  // ---------- polling ----------
  async function poll() {
    if (paused || inFlight || document.hidden) return;
    inFlight = true;
    try {
      const res = await fetch(config.dataset.url, { credentials: 'same-origin', headers: { Accept: 'application/json' } });
      if (res.status === 429) throw new Error('slow down');
      const next = await res.json();
      if (!res.ok || !next.success) throw new Error(next.error || 'failed');
      const newest = latestTime(next);
      const fresh = (next.readings || []).filter((r) => r.t > lastSeen).length;
      if (lastSeen && fresh) updates += fresh;
      if (newest) lastSeen = Math.max(lastSeen, newest);
      data = next;
      failures = 0;
      render();
    } catch (e) {
      failures += 1;
      if (failures >= 3) $('monSync').textContent = 'Connection lost. Retrying…';
    } finally {
      inFlight = false;
    }
  }

  function start() { stop(); timer = setInterval(poll, POLL_MS); }
  function stop() { if (timer) clearInterval(timer); timer = null; }

  const pauseBtn = $('monPause');
  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.setAttribute('aria-pressed', String(paused));
    pauseBtn.classList.toggle('is-paused', paused);
    pauseBtn.querySelector('span').textContent = paused ? 'Resume' : 'Pause';
    if (paused) stop(); else { start(); poll(); }
    render();
  });
  $('monReset').addEventListener('click', () => {
    // Start a fresh session: counter back to 0 and the charts only show readings that arrive from now on
    updates = 0;
    since = lastSeen;
    render();
  });
  $('monPatient').addEventListener('change', (e) => {
    window.location.href = `${config.dataset.page}?patient=${encodeURIComponent(e.target.value)}`;
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
  let resizeTimer;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 150); });

  render();
  start();
})();
