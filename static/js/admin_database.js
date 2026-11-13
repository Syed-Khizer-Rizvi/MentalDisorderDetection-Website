// Database Browser (templates/admin_database.html): copy buttons and the on-demand size estimate.
// Read-only page: nothing here sends anything but GET requests.
(function () {
  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {
      // Clipboard API needs HTTPS or localhost; fall back to a hidden textarea
      const area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.appendChild(area);
      area.select();
      document.execCommand('copy');
      area.remove();
    }
  }

  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('.db-copy');
    if (!btn) return;
    e.preventDefault();        // inside a <summary>: don't toggle the branch
    e.stopPropagation();
    await copyText(btn.dataset.copy);
    btn.classList.add('is-copied');
    btn.setAttribute('title', 'Copied');
    setTimeout(() => { btn.classList.remove('is-copied'); btn.setAttribute('title', 'Copy'); }, 1200);
  });

  // "open" links inside summaries should navigate, not toggle
  document.querySelectorAll('summary .db-open').forEach((a) => {
    a.addEventListener('click', (e) => e.stopPropagation());
  });

  const sizeBtn = document.getElementById('dbSizeBtn');
  if (sizeBtn) {
    sizeBtn.addEventListener('click', async () => {
      const out = document.getElementById('dbSize');
      sizeBtn.disabled = true;
      out.textContent = '…';
      try {
        const res = await fetch(sizeBtn.dataset.url, { credentials: 'same-origin' });
        const data = await res.json();
        if (!data.success) throw new Error(data.error);
        const b = data.bytes;
        out.textContent = b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(2)} MB`;
        sizeBtn.textContent = 'Recalculate';
        out.title = `${b.toLocaleString()} bytes of JSON (approximate; Firebase's own storage figure differs)`;
      } catch (err) {
        out.textContent = '—';
        sizeBtn.textContent = 'Retry';
      }
      sizeBtn.disabled = false;
    });
  }
})();
