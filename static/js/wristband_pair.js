// Pair / Revoke & Regenerate on the Wristband page (templates/wristband.html).
// The API key exists only in the response, so it's shown once and the page reloads when the
// patient confirms they've saved it.
(function () {
  const btn = document.getElementById('pairBtn');
  if (!btn) return;
  const result = document.getElementById('pairResult');
  const error = document.getElementById('pairError');

  btn.addEventListener('click', async () => {
    if (btn.hasAttribute('data-rotate') &&
        !confirm('Revoke the current credentials? The wristband stops syncing until you enter the new ones on it.')) {
      return;
    }
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Generating…';
    error.hidden = true;
    try {
      const res = await fetch(btn.dataset.url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'same-origin', body: '{}'
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.error || 'Could not generate credentials. Please try again.');
      document.getElementById('pairDevice').value = data.device_id;
      document.getElementById('pairKey').value = data.api_key;
      document.getElementById('pairEndpoint').value = data.endpoint;
      btn.hidden = true;
      result.hidden = false;
      result.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    } catch (err) {
      error.textContent = err.message;
      error.hidden = false;
      btn.disabled = false;
      btn.textContent = label;
    }
  });

  document.querySelectorAll('[data-copy]').forEach((copyBtn) => {
    copyBtn.addEventListener('click', async () => {
      const input = document.getElementById(copyBtn.dataset.copy);
      try {
        await navigator.clipboard.writeText(input.value);
      } catch {
        // The Clipboard API needs HTTPS or localhost; fall back to selecting the text
        input.select();
        document.execCommand('copy');
      }
      copyBtn.textContent = 'Copied ✓';
      setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
    });
  });

  document.getElementById('pairDone').addEventListener('click', () => location.reload());
})();
