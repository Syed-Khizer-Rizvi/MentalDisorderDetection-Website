// Bubble Pop: one pastel bubble per second drifts up; tap (or Enter/Space) to pop it.
// 60-second mode records a score; endless mode just records the session.
// Reduced motion: bubbles appear in place and fade away instead of floating.
(function () {
  const TINTS = ['rgba(167,243,208,0.75)', 'rgba(191,219,254,0.8)', 'rgba(221,214,254,0.8)', 'rgba(251,207,232,0.8)',
                 'rgba(254,240,138,0.75)', 'rgba(186,230,253,0.8)', 'rgba(254,215,170,0.75)'];
  const $ = (id) => document.getElementById(id);
  const stage = $('bbStage'), countEl = $('bbCount'), clockEl = $('bbClock');
  let popped = 0, spawnTimer = null, clock = null, running = false, mode = 'timed', endsAt = 0;

  function spawn() {
    if (document.hidden) return;   // don't pile bubbles up in a background tab
    const size = 30 + Math.random() * 50;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'bb-bubble';
    b.setAttribute('aria-label', 'Bubble');
    b.style.setProperty('--size', `${size}px`);
    b.style.setProperty('--x', `${Math.random() * (stage.clientWidth - size)}px`);
    b.style.setProperty('--tint', TINTS[Math.floor(Math.random() * TINTS.length)]);
    b.style.setProperty('--dur', `${7 + Math.random() * 5}s`);
    if (Game.reducedMotion) {
      b.classList.add('bb-static');
      b.style.setProperty('--y', `${10 + Math.random() * 70}%`);
      setTimeout(() => { b.classList.add('is-fading'); setTimeout(() => b.remove(), 700); }, 6000);
    } else {
      b.addEventListener('animationend', (e) => { if (e.animationName === 'bb-rise') b.remove(); });
    }
    b.addEventListener('click', () => pop(b));
    stage.appendChild(b);
  }

  function pop(b) {
    if (!running || b.classList.contains('is-popping')) return;
    b.classList.add('is-popping');
    popped += 1;
    countEl.textContent = popped;
    setTimeout(() => b.remove(), Game.reducedMotion ? 600 : 320);
  }

  function start() {
    mode = document.querySelector('input[name="mode"]:checked').value;
    popped = 0; running = true;
    countEl.textContent = '0';
    stage.querySelectorAll('.bb-bubble').forEach((b) => b.remove());
    $('bbHint').hidden = true; $('bbDone').hidden = true; $('bbBest').hidden = true;
    $('bbStart').hidden = true; $('bbFinish').hidden = false;
    document.querySelectorAll('input[name="mode"]').forEach((r) => { r.disabled = true; });
    $('bbClockLabel').textContent = mode === 'timed' ? 'Time left' : 'Time';
    Game.start();
    endsAt = performance.now() + 60_000;
    spawn();
    spawnTimer = setInterval(spawn, 1000);
    clock = setInterval(tick, 250);
    tick();
    $('bbAnnounce').textContent = 'Bubbles are rising. Tab to a bubble and press Enter to pop it.';
  }

  function tick() {
    if (mode === 'timed') {
      const left = Math.max(0, (endsAt - performance.now()) / 1000);
      clockEl.textContent = Game.formatTime(Math.ceil(left));
      if (left <= 0) finish();
    } else {
      clockEl.textContent = Game.formatTime(Game.elapsed());
    }
  }

  async function finish() {
    if (!running) return;
    running = false;
    clearInterval(spawnTimer); clearInterval(clock);
    stage.querySelectorAll('.bb-bubble').forEach((b) => b.remove());
    $('bbStart').hidden = false; $('bbFinish').hidden = true;
    document.querySelectorAll('input[name="mode"]').forEach((r) => { r.disabled = false; });
    const secs = Math.round(Game.elapsed());
    $('bbDoneText').textContent = `You popped ${popped} bubble${popped === 1 ? '' : 's'}` +
      (mode === 'timed' ? ' in 60 seconds.' : ` in ${Game.formatTime(secs)}.`);
    $('bbDone').hidden = false;
    $('bbHint').hidden = false;
    $('bbAgain').focus();
    // Only timed rounds are comparable, so only they set a best score
    const result = await Game.complete(mode === 'timed' ? { score: popped } : {});
    if (result && result.new_best) $('bbBest').hidden = false;
  }

  $('bbStart').addEventListener('click', start);
  $('bbFinish').addEventListener('click', finish);
  $('bbAgain').addEventListener('click', start);
})();
