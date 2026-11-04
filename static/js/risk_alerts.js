// Acknowledge / undo for risk alert cards (templates/_alert_card.html), on the Risk Alerts page
// and the patient detail page. After a change the page reloads so the counts, tabs and stats agree.
(function () {
  const dialog = document.getElementById('ackDialog');
  if (!dialog) return;
  const note = document.getElementById('ackNote');
  const error = document.getElementById('ackError');
  const confirmBtn = document.getElementById('ackConfirm');
  let current = null;

  async function post(alertId, action, body) {
    const res = await fetch(`/api/alerts/${encodeURIComponent(alertId)}/${action}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'same-origin',
      body: JSON.stringify(body || {})
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.success) throw new Error(data.error || 'Something went wrong');
    return data;
  }

  function showError(msg) { error.textContent = msg; error.hidden = !msg; }

  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-alert-action]');
    if (!btn) return;
    const card = btn.closest('.alert-card');
    if (btn.dataset.alertAction === 'acknowledge') {
      current = card;
      const who = card.querySelector('.alert-patient');
      const score = card.querySelector('.alert-score strong').textContent;
      const sev = card.querySelector('.alert-sev').textContent;
      document.getElementById('ackSummary').textContent =
        `${sev} alert${who ? ' for ' + who.textContent : ''}: composite ${score}/100.`;
      note.value = '';
      showError('');
      dialog.showModal();
      note.focus();
    } else {
      btn.disabled = true;
      try {
        await post(card.dataset.alertId, 'unacknowledge');
        location.reload();
      } catch (err) {
        btn.disabled = false;
        alert(err.message);
      }
    }
  });

  document.getElementById('ackForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!current) return;
    confirmBtn.disabled = true;
    confirmBtn.textContent = 'Saving…';
    try {
      await post(current.dataset.alertId, 'acknowledge', { note: note.value.trim() });
      location.reload();
    } catch (err) {
      showError(err.message);
      confirmBtn.disabled = false;
      confirmBtn.textContent = 'Acknowledge';
    }
  });
  document.getElementById('ackCancel').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });
})();
