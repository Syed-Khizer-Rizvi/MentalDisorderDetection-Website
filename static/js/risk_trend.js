// Tooltip and crosshair for the Risk Trend chart (templates/_risk_trend.html).
// Mouse: hover previews a point; the tooltip stays while the pointer moves onto it, so its
// "View details" link can be clicked. Touch/click: tapping a point pins the tooltip until
// tapping elsewhere or pressing Escape. Keyboard: focusing a point shows it too.
const trendChart = document.getElementById('trendChart');
if (trendChart) {
  const tooltip = document.getElementById('trendTooltip');
  const crosshair = document.getElementById('trendCrosshair');
  const plot = trendChart.querySelector('.trend-plot');
  let pinned = null;
  let hideTimer;

  const escapeHtml = (s) => s.replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

  function show(pt) {
    clearTimeout(hideTimer);
    const d = pt.dataset;
    crosshair.style.left = pt.style.left;
    crosshair.hidden = false;
    tooltip.innerHTML =
      `<span class="trend-tip-date">${escapeHtml(d.date)}</span>` +
      `<span class="trend-tip-score"><strong>${escapeHtml(d.score)}</strong>/100 ` +
      `<span class="trend-badge trend-badge-${escapeHtml(d.level)}">${escapeHtml(d.risk)}</span></span>` +
      (d.url ? `<a class="trend-tip-link" href="${escapeHtml(d.url)}">View details →</a>` : '');
    tooltip.hidden = false;
    // Keep the card inside the plot horizontally; flip below the point when there's no room above
    const half = tooltip.offsetWidth / 2;
    tooltip.style.left = Math.min(Math.max(pt.offsetLeft, half), plot.offsetWidth - half) + 'px';
    const below = pt.offsetTop - tooltip.offsetHeight - 16 < -40;
    tooltip.classList.toggle('trend-tooltip-below', below);
    tooltip.style.top = (below ? pt.offsetTop + 16 : pt.offsetTop - 14) + 'px';
    trendChart.querySelectorAll('.trend-point.is-active').forEach(p => p.classList.remove('is-active'));
    pt.classList.add('is-active');
  }

  function hide() {
    tooltip.hidden = true;
    crosshair.hidden = true;
    pinned = null;
    trendChart.querySelectorAll('.trend-point.is-active').forEach(p => p.classList.remove('is-active'));
  }

  const hideSoon = () => { if (!pinned) hideTimer = setTimeout(hide, 180); };

  trendChart.querySelectorAll('.trend-point').forEach(pt => {
    pt.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse' && !pinned) show(pt); });
    pt.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') hideSoon(); });
    pt.addEventListener('focus', () => { if (!pinned) show(pt); });
    pt.addEventListener('blur', () => { if (!pinned) hideSoon(); });
    pt.addEventListener('click', (e) => {
      e.stopPropagation();
      if (pinned === pt) { hide(); return; }
      show(pt);
      pinned = pt;
    });
  });

  tooltip.addEventListener('pointerenter', () => clearTimeout(hideTimer));
  tooltip.addEventListener('pointerleave', hideSoon);
  tooltip.addEventListener('click', (e) => e.stopPropagation());
  document.addEventListener('click', () => { if (!tooltip.hidden) hide(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !tooltip.hidden) hide(); });
}
