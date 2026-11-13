// System Settings (templates/admin_settings.html). The server validates every field on save;
// this script only tracks unsaved changes and fills in the defaults for "Reset to Defaults".
(function () {
  const form = document.getElementById('settingsForm');
  if (!form) return;
  const fields = Array.from(form.querySelectorAll('input[name*="."], select[name*="."]'));
  const note = document.getElementById('dirtyNote');
  const read = (el) => (el.type === 'checkbox' ? (el.checked ? '1' : '0') : String(el.value));
  const saved = new Map(fields.map((el) => [el, read(el)]));
  let submitting = false;

  function refresh() {
    const dirty = fields.some((el) => read(el) !== saved.get(el));
    note.hidden = !dirty;
    return dirty;
  }
  fields.forEach((el) => {
    el.addEventListener('input', refresh);
    el.addEventListener('change', refresh);
  });

  document.getElementById('resetDefaults').addEventListener('click', () => {
    if (!confirm('Fill every field with its default value? Nothing is saved until you click "Save All Changes".')) return;
    fields.forEach((el) => {
      if (el.type === 'checkbox') el.checked = el.dataset.default === '1';
      else el.value = el.dataset.default;
    });
    refresh();
  });

  // Warn before leaving with unsaved edits (not when submitting the form itself)
  form.addEventListener('submit', () => { submitting = true; });
  window.addEventListener('beforeunload', (e) => {
    if (!submitting && refresh()) {
      e.preventDefault();
      e.returnValue = '';
    }
  });
})();
