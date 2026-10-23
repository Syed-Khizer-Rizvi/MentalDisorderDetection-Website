// Sidebar toggle shared by the dashboard pages.
// Desktop: collapses the sidebar to an icon rail. Phone (<=720px): slides it in as an overlay.
(function () {
  const toggle = document.getElementById('menuToggle');
  const sidebar = document.getElementById('sidebar');
  if (!toggle || !sidebar) return;

  const mobile = window.matchMedia('(max-width: 720px)');
  const backdrop = document.createElement('div');
  backdrop.className = 'sidebar-backdrop';
  backdrop.hidden = true;
  document.body.appendChild(backdrop);

  toggle.setAttribute('aria-label', 'Toggle navigation');
  toggle.setAttribute('aria-controls', 'sidebar');
  toggle.setAttribute('aria-expanded', 'false');

  function setOpen(open) {
    sidebar.classList.toggle('open', open);
    backdrop.hidden = !open;
    document.body.classList.toggle('sidebar-open', open);
    toggle.setAttribute('aria-expanded', String(open));
  }

  toggle.addEventListener('click', () => {
    if (mobile.matches) setOpen(!sidebar.classList.contains('open'));
    else sidebar.classList.toggle('collapsed');
  });
  backdrop.addEventListener('click', () => setOpen(false));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && sidebar.classList.contains('open')) {
      setOpen(false);
      toggle.focus();
    }
  });
  mobile.addEventListener('change', (e) => { if (!e.matches) setOpen(false); });
})();
