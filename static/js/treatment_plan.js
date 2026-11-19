// Doctor treatment plan form (templates/doctor_treatment_plan.html): validates, saves as JSON, shows field errors.
(function () {
  const form = document.getElementById('tpForm');
  const saveBtn = document.getElementById('tpSave');
  const formError = document.getElementById('tpFormError');
  const toast = document.getElementById('tpToast');
  const instructions = document.getElementById('tpInstructions');
  const counter = document.querySelector('[data-count="tpInstructions"]');

  const updateCount = () => { counter.textContent = instructions.value.length; };
  instructions.addEventListener('input', updateCount);
  updateCount();

  function showErrors(errors) {
    form.querySelectorAll('[data-error]').forEach((el) => {
      const message = errors[el.dataset.error];
      el.textContent = message || '';
      el.hidden = !message;
      const field = form.querySelector(`[name="${el.dataset.error}"]`);
      if (field && field.type !== 'checkbox') field.setAttribute('aria-invalid', message ? 'true' : 'false');
    });
    const first = Object.keys(errors)[0];
    const target = first && form.querySelector(`[name="${first}"]`);
    if (target) target.focus();
  }

  function collect() {
    const checked = (name) => [...form.querySelectorAll(`input[name="${name}"]:checked`)].map((i) => i.value);
    return {
      observations: document.getElementById('tpObservations').value.trim(),
      instructions: instructions.value.trim(),
      recommended_activities: checked('recommended_activities'),
      recommended_games: checked('recommended_games'),
      follow_up_date: document.getElementById('tpFollowUp').value,
    };
  }

  form.addEventListener('input', (e) => {
    e.target.removeAttribute('aria-invalid');
    const err = form.querySelector(`[data-error="${e.target.name}"]`);
    if (err) err.hidden = true;
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    formError.hidden = true;
    const body = collect();
    const errors = {};
    if (!body.instructions) errors.instructions = 'Write the instructions your patient will see.';
    if (!body.recommended_activities.length && !body.recommended_games.length)
      errors.recommended_activities = 'Recommend at least one activity or game.';
    if (Object.keys(errors).length) { showErrors(errors); return; }

    saveBtn.disabled = true;
    const label = saveBtn.textContent;
    saveBtn.textContent = 'Saving...';
    try {
      const res = await fetch(form.dataset.url, {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) {
        if (data.errors) showErrors(data.errors);
        throw new Error(data.error || 'Could not save the plan. Please try again.');
      }
      toast.textContent = data.created ? '✓ Treatment plan saved. Your patient can see it now.' : '✓ Treatment plan updated.';
      toast.hidden = false;
      setTimeout(() => window.location.reload(), 900);
    } catch (err) {
      formError.textContent = err.message;
      formError.hidden = false;
      saveBtn.disabled = false;
      saveBtn.textContent = label;
    }
  });
})();
