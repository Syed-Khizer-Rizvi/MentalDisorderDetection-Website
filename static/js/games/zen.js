// Zen Garden: sand texture + a layer of rake grooves (5 tines following the pointer) + stones on top.
// Stones press a few raked rings into the sand around them. No score.
(function () {
  const canvas = document.getElementById('zgCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const sand = document.createElement('canvas'); sand.width = W; sand.height = H;
  const rake = document.createElement('canvas'); rake.width = W; rake.height = H;
  const rctx = rake.getContext('2d');
  const TINES = 5, GAP = 9;
  let stones = [], mode = 'rake', last = null, drawing = false, interacted = false;

  // Sand texture: warm beige with fine grain
  (function paintSand() {
    const s = sand.getContext('2d');
    const img = s.createImageData(W, H);
    for (let i = 0; i < img.data.length; i += 4) {
      const n = (Math.random() - 0.5) * 26;
      img.data[i] = 226 + n; img.data[i + 1] = 210 + n; img.data[i + 2] = 172 + n * 0.8; img.data[i + 3] = 255;
    }
    s.putImageData(img, 0, 0);
    const glow = s.createRadialGradient(W / 2, H / 2, 50, W / 2, H / 2, W * 0.7);
    glow.addColorStop(0, 'rgba(255,248,230,0.25)');
    glow.addColorStop(1, 'rgba(120,90,50,0.18)');
    s.fillStyle = glow;
    s.fillRect(0, 0, W, H);
  })();

  function groove(ctx2, x1, y1, x2, y2) {
    // A groove is a dark line with a light edge just beside it
    ctx2.lineCap = 'round';
    ctx2.strokeStyle = 'rgba(120, 88, 45, 0.38)'; ctx2.lineWidth = 3.4;
    ctx2.beginPath(); ctx2.moveTo(x1, y1); ctx2.lineTo(x2, y2); ctx2.stroke();
    ctx2.strokeStyle = 'rgba(255, 250, 235, 0.55)'; ctx2.lineWidth = 1.4;
    ctx2.beginPath(); ctx2.moveTo(x1 + 1.6, y1 + 1.6); ctx2.lineTo(x2 + 1.6, y2 + 1.6); ctx2.stroke();
  }

  function rakeSegment(a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    if (len < 0.5) return;
    const nx = -dy / len, ny = dx / len;   // perpendicular: tines are spread across the direction of travel
    for (let t = 0; t < TINES; t++) {
      const off = (t - (TINES - 1) / 2) * GAP;
      groove(rctx, a.x + nx * off, a.y + ny * off, b.x + nx * off, b.y + ny * off);
    }
  }

  function ringsAround(stone) {
    for (let k = 1; k <= 3; k++) {
      const r = stone.r + 10 + k * GAP * 1.2;
      const steps = Math.ceil(r * 0.6);
      let prev = null;
      for (let i = 0; i <= steps; i++) {
        const ang = (i / steps) * Math.PI * 2;
        const p = { x: stone.x + Math.cos(ang) * r, y: stone.y + Math.sin(ang) * r * 0.92 };
        if (prev) groove(rctx, prev.x, prev.y, p.x, p.y);
        prev = p;
      }
    }
  }

  function drawStone(s) {
    ctx.save();
    ctx.shadowColor = 'rgba(60, 40, 20, 0.35)'; ctx.shadowBlur = 14; ctx.shadowOffsetX = 4; ctx.shadowOffsetY = 6;
    const g = ctx.createRadialGradient(s.x - s.r * 0.35, s.y - s.r * 0.4, s.r * 0.15, s.x, s.y, s.r * 1.05);
    g.addColorStop(0, s.light); g.addColorStop(0.65, s.base); g.addColorStop(1, s.dark);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.ellipse(s.x, s.y, s.r * s.sx, s.r, s.rot, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function render() {
    ctx.drawImage(sand, 0, 0);
    ctx.drawImage(rake, 0, 0);
    stones.forEach(drawStone);
  }

  function point(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: (e.clientX - rect.left) * (W / rect.width), y: (e.clientY - rect.top) * (H / rect.height) };
  }

  const STONE_TONES = [
    ['#B8B5AE', '#8A867E', '#5E5A53'], ['#A9A39A', '#77716A', '#4D4943'], ['#C9C2B5', '#9C9384', '#6B6357'], ['#9FA4A6', '#6E7477', '#474C4F'],
  ];
  canvas.addEventListener('pointerdown', (e) => {
    try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* not supported for this pointer */ }
    const p = point(e);
    interacted = true;
    if (mode === 'stone') {
      const tone = STONE_TONES[Math.floor(Math.random() * STONE_TONES.length)];
      const stone = { x: p.x, y: p.y, r: 22 + Math.random() * 26, sx: 1 + Math.random() * 0.35, rot: Math.random() * Math.PI,
                      light: tone[0], base: tone[1], dark: tone[2] };
      ringsAround(stone);
      stones.push(stone);
      render();
      return;
    }
    drawing = true;
    last = p;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    const p = point(e);
    if (Math.hypot(p.x - last.x, p.y - last.y) < 3) return;
    rakeSegment(last, p);
    last = p;
    render();
  });
  const stop = () => { drawing = false; last = null; };
  canvas.addEventListener('pointerup', stop);
  canvas.addEventListener('pointercancel', stop);

  const rakeBtn = document.getElementById('zgRake'), stoneBtn = document.getElementById('zgStone');
  function setMode(m) {
    mode = m;
    rakeBtn.setAttribute('aria-pressed', String(m === 'rake'));
    stoneBtn.setAttribute('aria-pressed', String(m === 'stone'));
    document.getElementById('zgWrap').classList.toggle('is-stone', m === 'stone');
  }
  rakeBtn.addEventListener('click', () => setMode('rake'));
  stoneBtn.addEventListener('click', () => setMode(mode === 'stone' ? 'rake' : 'stone'));
  document.getElementById('zgSmooth').addEventListener('click', () => { rctx.clearRect(0, 0, W, H); render(); });
  document.getElementById('zgClear').addEventListener('click', () => { rctx.clearRect(0, 0, W, H); stones = []; render(); });
  document.getElementById('zgDone').addEventListener('click', async () => {
    const res = await Game.complete({});
    if (res) Game.toast(interacted ? 'Lovely garden. Session saved.' : 'Session saved. Come back any time.');
  });

  Game.start();
  render();
})();
