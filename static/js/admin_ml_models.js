// ML Models admin page (templates/admin_ml_models.html): fusion weight sliders, the three-handle
// risk threshold bar and the retrain instructions dialog. The server re-validates everything on save.
(function () {
  // ----- Fusion weights: live total, Save only when they add up to 1.00 -----
  const fusion = document.getElementById('fusion');
  if (fusion) {
    const sliders = Array.from(fusion.querySelectorAll('.ml-range'));
    const totalBox = document.getElementById('weightTotal');
    const totalValue = document.getElementById('weightTotalValue');
    const mark = document.getElementById('weightTotalMark');
    const save = document.getElementById('saveWeights');
    const okIcon = mark.innerHTML;
    const badIcon = '<svg class="ad-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';

    function paint(slider) {
      // Green fill up to the thumb
      slider.style.setProperty('--fill', `${slider.value * 100}%`);
      document.getElementById(`${slider.id}-out`).textContent = Number(slider.value).toFixed(2);
    }
    function update() {
      // Work in hundredths so 0.05 steps add up exactly
      const hundredths = sliders.reduce((sum, s) => sum + Math.round(s.value * 100), 0);
      const ok = hundredths === 100;
      totalValue.textContent = (hundredths / 100).toFixed(2);
      totalBox.classList.toggle('is-ok', ok);
      totalBox.classList.toggle('is-bad', !ok);
      mark.innerHTML = ok ? okIcon : badIcon;
      mark.setAttribute('aria-label', ok ? 'adds up to 1.00' : 'must add up to 1.00');
      save.disabled = !ok;
      save.title = ok ? '' : 'The weights must add up to 1.00';
    }
    sliders.forEach((s) => {
      paint(s);
      s.addEventListener('input', () => { paint(s); update(); });
    });
    fusion.querySelector('.ml-reset').addEventListener('click', (e) => {
      const defaults = JSON.parse(e.currentTarget.dataset.reset);
      sliders.forEach((s) => { s.value = defaults[s.name]; paint(s); });
      update();
    });
    update();
  }

  // ----- Risk thresholds: three handles on one 0-100 bar, kept in order with 5-point gaps -----
  const thresholds = document.getElementById('thresholds');
  if (thresholds) {
    const [low, moderate, high] = ['low', 'moderate', 'high'].map((n) => thresholds.querySelector(`[name="${n}"]`));
    const GAP = 5;
    const bounds = {
      low: () => [10, 40],
      moderate: () => [Number(low.value) + GAP, 70],
      high: () => [Number(moderate.value) + GAP, 90]
    };
    const clamp = (input) => {
      const [min, max] = bounds[input.name]();
      input.value = Math.min(Math.max(Number(input.value), min), max);
    };
    function render() {
      const t = { low: +low.value, moderate: +moderate.value, high: +high.value };
      const widths = { low: t.low, moderate: t.moderate - t.low, high: t.high - t.moderate, critical: 100 - t.high };
      Object.entries(widths).forEach(([band, w]) => {
        thresholds.querySelector(`[data-band="${band}"]`).style.width = `${w}%`;
      });
      document.getElementById('range-low').textContent = `0–${t.low}`;
      document.getElementById('range-moderate').textContent = `${t.low + 1}–${t.moderate}`;
      document.getElementById('range-high').textContent = `${t.moderate + 1}–${t.high}`;
      document.getElementById('range-critical').textContent = `${t.high + 1}–100`;
      [low, moderate, high].forEach((input) => {
        input.setAttribute('aria-valuetext', `${input.value}`);
      });
    }
    // Re-clamp all three in order, so pushing Low up also pushes Moderate and High along when needed
    function settle(changed) {
      clamp(changed);
      [low, moderate, high].forEach(clamp);
      render();
    }
    [low, moderate, high].forEach((input) => input.addEventListener('input', () => settle(input)));
    // Overlapping handles: lift the one being used so it stays grabbable
    [low, moderate, high].forEach((input) => {
      input.addEventListener('pointerdown', () => {
        [low, moderate, high].forEach((i) => i.classList.toggle('is-active', i === input));
      });
    });
    thresholds.querySelector('.ml-reset').addEventListener('click', (e) => {
      const d = JSON.parse(e.currentTarget.dataset.reset);
      low.value = d.low; moderate.value = d.moderate; high.value = d.high;
      render();
    });
    render();
  }

  // ----- Retrain instructions dialog -----
  const dialog = document.getElementById('retrainDialog');
  const open = document.getElementById('retrainOpen');
  if (dialog && open && typeof dialog.showModal === 'function') {
    open.addEventListener('click', () => dialog.showModal());
    document.getElementById('retrainClose').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });   // backdrop click
    dialog.addEventListener('close', () => open.focus());
  }
})();
