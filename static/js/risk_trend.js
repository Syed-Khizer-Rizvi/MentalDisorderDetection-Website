// Hover/focus tooltip and crosshair for the Risk Trend chart (templates/_risk_trend.html).
const trendChart = document.getElementById('trendChart');
if (trendChart) {
  const tooltip = document.getElementById('trendTooltip');
  const crosshair = document.getElementById('trendCrosshair');
  const plot = trendChart.querySelector('.trend-plot');
  const hide = () => { tooltip.hidden = true; crosshair.hidden = true; };
  trendChart.querySelectorAll('.trend-point').forEach(pt => {
    const show = () => {
      crosshair.style.left = pt.style.left;
      crosshair.hidden = false;
      tooltip.innerHTML = `<strong>${pt.dataset.score}/100 · ${pt.dataset.risk}</strong><span>${pt.dataset.date}</span>`;
      tooltip.hidden = false;
      const half = tooltip.offsetWidth / 2;
      const x = pt.offsetLeft;
      tooltip.style.left = Math.min(Math.max(x, half), plot.offsetWidth - half) + 'px';
      tooltip.style.top = (pt.offsetTop - 14) + 'px';
    };
    pt.addEventListener('mouseenter', show);
    pt.addEventListener('focus', show);
    pt.addEventListener('mouseleave', hide);
    pt.addEventListener('blur', hide);
  });
}
