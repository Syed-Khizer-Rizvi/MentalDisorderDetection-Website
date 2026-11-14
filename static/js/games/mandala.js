// Mandala Draw: every stroke is repeated N times around the centre (and mirrored, if on),
// so any doodle becomes a symmetric pattern. Guides are on a separate overlay, never in the download.
(function () {
  const canvas = document.getElementById('mdCanvas');
  const guides = document.getElementById('mdGuideCanvas');
  const ctx = canvas.getContext('2d'), gctx = guides.getContext('2d');
  const S = canvas.width, C = S / 2, BG = '#0B1736';
  let axes = 8, mirror = true, showGuides = true, last = null, drawing = false, strokes = 0;
  const history = [];   // snapshots for Undo (one per stroke, capped)

  function background() { ctx.fillStyle = BG; ctx.fillRect(0, 0, S, S); }

  function drawGuides() {
    gctx.clearRect(0, 0, S, S);
    if (!showGuides) return;
    gctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
    gctx.lineWidth = 1.5;
    for (let k = 0; k < axes; k++) {
      const a = (Math.PI * 2 * k) / axes;
      gctx.beginPath(); gctx.moveTo(C, C); gctx.lineTo(C + Math.cos(a) * C, C + Math.sin(a) * C); gctx.stroke();
    }
    [0.25, 0.5, 0.75].forEach((f) => { gctx.beginPath(); gctx.arc(C, C, C * f, 0, Math.PI * 2); gctx.stroke(); });
    gctx.fillStyle = 'rgba(226, 232, 240, 0.6)';
    gctx.beginPath(); gctx.arc(C, C, 5, 0, Math.PI * 2); gctx.fill();
  }

  function segment(a, b) {
    const color = document.querySelector('input[name="color"]:checked').value;
    ctx.save();
    ctx.translate(C, C);
    ctx.strokeStyle = color;
    ctx.lineWidth = Number(document.getElementById('mdSize').value);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    for (let k = 0; k < axes; k++) {
      ctx.save();
      ctx.rotate((Math.PI * 2 * k) / axes);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      if (mirror) { ctx.scale(1, -1); ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
      ctx.restore();
    }
    ctx.restore();
  }

  function point(e) {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * (S / r.width) - C, y: (e.clientY - r.top) * (S / r.height) - C };   // relative to the centre
  }

  canvas.addEventListener('pointerdown', (e) => {
    try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* not supported for this pointer */ }
    history.push(ctx.getImageData(0, 0, S, S));
    if (history.length > 20) history.shift();
    drawing = true;
    last = point(e);
    segment(last, { x: last.x + 0.01, y: last.y });   // a tap leaves a dot
    strokes += 1;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    const p = point(e);
    if (Math.hypot(p.x - last.x, p.y - last.y) < 1.5) return;
    segment(last, p);
    last = p;
  });
  const end = () => { drawing = false; };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);

  document.querySelectorAll('input[name="axes"]').forEach((r) => r.addEventListener('change', () => { axes = Number(r.value); drawGuides(); }));
  const size = document.getElementById('mdSize');
  size.addEventListener('input', () => { document.getElementById('mdSizeOut').textContent = size.value; });
  function toggle(btn, fn) {
    btn.addEventListener('click', () => {
      const on = btn.getAttribute('aria-pressed') !== 'true';
      btn.setAttribute('aria-pressed', String(on));
      fn(on);
    });
  }
  toggle(document.getElementById('mdMirror'), (on) => { mirror = on; });
  toggle(document.getElementById('mdGuides'), (on) => { showGuides = on; drawGuides(); });
  document.getElementById('mdUndo').addEventListener('click', () => { const snap = history.pop(); if (snap) ctx.putImageData(snap, 0, 0); });
  document.getElementById('mdClear').addEventListener('click', () => {
    if (strokes && !confirm('Clear your mandala?')) return;
    history.push(ctx.getImageData(0, 0, S, S));
    background();
  });
  document.getElementById('mdDownload').addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = `mandala-${new Date().toISOString().slice(0, 10)}.png`;
    a.click();
  });
  document.getElementById('mdDone').addEventListener('click', async () => {
    const res = await Game.complete({});
    if (res) Game.toast(strokes ? 'Beautiful. Session saved. Download it to keep it.' : 'Session saved.');
  });

  Game.start();
  background();
  drawGuides();
})();
