// 4-7-8 breathing: the circle grows while breathing in (4 s), stays while holding (7 s) and shrinks
// while breathing out (8 s). With reduced motion the circle stays still; the text, count and bar guide instead.
(function () {
  const PHASES = [
    { name: 'Breathe In', secs: 4, scale: 1, cls: 'is-in' },
    { name: 'Hold', secs: 7, scale: 1, cls: 'is-hold' },
    { name: 'Breathe Out', secs: 8, scale: 0.55, cls: 'is-out' },
  ];
  const $ = (id) => document.getElementById(id);
  const circle = $('brCircle'), phaseText = $('brPhaseText'), count = $('brCount'), bar = $('brBar');
  const cycleText = $('brCycle'), announce = $('brAnnounce'), setup = $('brSetup'), stopBtn = $('brStop'), done = $('brDone');
  let timers = [], running = false, totalCycles = 5;

  function clearTimers() { timers.forEach(clearTimeout); timers.forEach(clearInterval); timers = []; }
  function later(fn, ms) { const t = setTimeout(fn, ms); timers.push(t); return t; }

  function reset() {
    clearTimers();
    running = false;
    circle.className = 'br-circle';
    circle.style.transitionDuration = '0.8s';
    circle.style.setProperty('--scale', 0.55);
    phaseText.textContent = 'Ready';
    count.textContent = '';
    bar.style.transition = 'none';
    bar.style.width = '0';
  }

  function runPhase(cycle, index) {
    if (!running) return;
    const phase = PHASES[index];
    circle.className = `br-circle ${phase.cls}`;
    circle.style.transitionDuration = `${phase.secs}s`;
    circle.style.setProperty('--scale', phase.scale);
    phaseText.textContent = phase.name;
    announce.textContent = `${phase.name}, ${phase.secs} seconds`;
    cycleText.textContent = `Cycle ${cycle} of ${totalCycles}`;
    // Progress through the whole session
    const doneSecs = (cycle - 1) * 19 + PHASES.slice(0, index).reduce((s, p) => s + p.secs, 0);
    bar.style.transition = 'none';
    bar.style.width = `${(doneSecs / (totalCycles * 19)) * 100}%`;
    requestAnimationFrame(() => {
      bar.style.transition = `width ${phase.secs}s linear`;
      bar.style.width = `${((doneSecs + phase.secs) / (totalCycles * 19)) * 100}%`;
    });
    let left = phase.secs;
    count.textContent = left;
    const tick = setInterval(() => { left -= 1; if (left > 0) count.textContent = left; }, 1000);
    timers.push(tick);
    later(() => {
      clearInterval(tick);
      if (index < PHASES.length - 1) runPhase(cycle, index + 1);
      else if (cycle < totalCycles) runPhase(cycle + 1, 0);
      else finish();
    }, phase.secs * 1000);
  }

  async function finish() {
    running = false;
    clearTimers();
    circle.className = 'br-circle';
    circle.style.transitionDuration = '1.2s';
    circle.style.setProperty('--scale', 0.75);
    phaseText.textContent = 'Well done';
    count.textContent = '';
    stopBtn.hidden = true;
    cycleText.textContent = '';
    $('brDoneText').textContent = `You completed ${totalCycles} breathing cycles in ${Game.formatTime(Game.elapsed())}. Notice how your body feels now.`;
    done.hidden = false;
    announce.textContent = 'Session complete';
    await Game.complete({ score: totalCycles });
  }

  $('brStart').addEventListener('click', () => {
    totalCycles = Number(document.querySelector('input[name="cycles"]:checked').value);
    reset();
    running = true;
    setup.hidden = true;
    done.hidden = true;
    stopBtn.hidden = false;
    Game.start();
    runPhase(1, 0);
    stopBtn.focus();
  });
  stopBtn.addEventListener('click', () => {
    reset();
    stopBtn.hidden = true;
    setup.hidden = false;
    cycleText.textContent = 'Stopped. Start again whenever you like.';
    $('brStart').focus();
  });
  $('brAgain').addEventListener('click', () => {
    done.hidden = true;
    setup.hidden = false;
    reset();
    cycleText.textContent = '';
    $('brStart').focus();
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && running) stopBtn.click(); });
  reset();
})();
