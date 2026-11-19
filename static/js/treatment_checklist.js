// "My Treatment Plan" checklist on the patient dashboard. Ticks are stored with the time they were made;
// only ticks since the patient's own local midnight count, so the list clears at their midnight.
(function () {
  const dataEl = document.getElementById('tpData');
  if (!dataEl) return;
  const data = JSON.parse(dataEl.textContent);
  const boxes = [...document.querySelectorAll('.tp-checklist input[type="checkbox"]')];
  const error = document.getElementById('tpTickError');
  let completed = data.completed || {};   // {activity: ms}

  const midnight = () => { const d = new Date(); d.setHours(0, 0, 0, 0); return d.getTime(); };
  const isToday = (ms) => typeof ms === 'number' && ms >= midnight();

  function render() {
    const done = boxes.filter((b) => isToday(completed[b.value]));
    boxes.forEach((b) => {
      b.checked = isToday(completed[b.value]);
      b.closest('li').classList.toggle('is-done', b.checked);
    });
    const count = done.length, total = boxes.length;
    document.getElementById('tpDone').textContent = count;
    const bar = document.getElementById('tpProgress');
    bar.setAttribute('aria-valuenow', count);
    bar.firstElementChild.style.width = `${total ? Math.round((count / total) * 100) : 0}%`;
    const badge = document.getElementById('tpBadge');
    if (badge) {
      document.getElementById('tpBadgeDone').textContent = count;
      badge.classList.toggle('is-complete', total > 0 && count === total);
    }
  }

  boxes.forEach((box) => box.addEventListener('change', async () => {
    const done = box.checked;
    error.hidden = true;
    box.disabled = true;
    try {
      const res = await fetch(data.url, {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ activity: box.value, done, tz_offset: new Date().getTimezoneOffset() }),
      });
      const reply = await res.json().catch(() => ({}));
      if (!res.ok || !reply.success) throw new Error(reply.error || 'Could not save. Please try again.');
      if (done) completed[box.value] = Date.now(); else delete completed[box.value];
    } catch (e) {
      box.checked = !done;
      error.textContent = e.message;
      error.hidden = false;
    } finally {
      box.disabled = false;
      render();
    }
  }));

  render();
  // A dashboard left open overnight clears itself at midnight
  setTimeout(render, midnight() + 86_400_000 - Date.now() + 1000);
})();
