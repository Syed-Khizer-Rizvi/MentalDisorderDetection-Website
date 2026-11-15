// Patient consultation request dialog (templates/_consultation_modal.html).
// Buttons with data-consult-open="consultation" open a normal request (routine/urgent);
// data-consult-open="emergency" opens an emergency appointment request (urgency fixed to emergency).
(function () {
  const dialog = document.getElementById('consultDialog');
  if (!dialog) return;
  const form = document.getElementById('consultForm');
  const $ = (id) => document.getElementById(id);
  const urgency = $('csUrgency'), preferred = $('csPreferredField'), submit = $('consultSubmit');
  let opener = null;

  function setMode(mode) {
    const emergency = mode === 'emergency';
    form.elements.type.value = emergency ? 'emergency_appointment' : 'consultation';
    $('consultTitle').textContent = emergency ? 'Book an emergency appointment' : 'Request a consultation';
    $('consultHead').classList.toggle('is-emergency', emergency);
    $('consultEmergencyNote').hidden = !emergency;
    urgency.querySelector('option[value="emergency"]').hidden = !emergency;
    urgency.querySelector('option[value="routine"]').hidden = emergency;
    urgency.querySelector('option[value="urgent"]').hidden = emergency;
    urgency.value = emergency ? 'emergency' : 'routine';
    urgency.disabled = emergency;
    submit.textContent = emergency ? 'Request Emergency Appointment' : 'Send Request';
    submit.classList.toggle('cs-btn-emergency', emergency);
    submit.classList.toggle('cs-btn-primary', !emergency);
    syncPreferred();
  }
  function syncPreferred() {
    // Preferred times only make sense for routine requests
    preferred.hidden = urgency.value !== 'routine';
    const now = new Date(Date.now() - new Date().getTimezoneOffset() * 60000);
    $('csPreferred').min = now.toISOString().slice(0, 16);
  }
  function clearErrors() {
    ['csSubjectErr', 'csMessageErr', 'csPreferredErr', 'consultError'].forEach((id) => { $(id).hidden = true; });
    form.querySelectorAll('[aria-invalid]').forEach((el) => el.removeAttribute('aria-invalid'));
  }
  function fieldError(inputId, errId, text) {
    $(inputId).setAttribute('aria-invalid', 'true');
    $(errId).textContent = text;
    $(errId).hidden = false;
  }
  function toast(text, isError) {
    const t = $('consultToast');
    t.textContent = text;
    t.classList.toggle('is-error', !!isError);
    t.hidden = false;
    clearTimeout(t._timer);
    t._timer = setTimeout(() => { t.hidden = true; }, 4500);
  }

  document.querySelectorAll('[data-consult-open]').forEach((btn) => btn.addEventListener('click', (e) => {
    e.preventDefault();
    opener = btn;
    form.reset();
    clearErrors();
    $('csCount').textContent = '0';
    setMode(btn.dataset.consultOpen);
    dialog.showModal();
    $('csSubject').focus();
  }));
  dialog.querySelectorAll('[data-consult-close]').forEach((b) => b.addEventListener('click', () => dialog.close()));
  dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => { if (opener) opener.focus(); });
  urgency.addEventListener('change', syncPreferred);
  $('csMessage').addEventListener('input', (e) => { $('csCount').textContent = e.target.value.length; });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearErrors();
    const subject = $('csSubject').value.trim(), message = $('csMessage').value.trim();
    let ok = true;
    if (!subject) { fieldError('csSubject', 'csSubjectErr', 'Add a short subject.'); ok = false; }
    if (message.length < 10) { fieldError('csMessage', 'csMessageErr', 'Please write at least 10 characters.'); ok = false; }
    if (!ok) return;
    submit.disabled = true;
    const label = submit.textContent;
    submit.textContent = 'Sending…';
    try {
      const res = await fetch(dialog.dataset.url, {
        method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: form.elements.type.value, urgency: urgency.value, subject, message,
                               preferred_time: urgency.value === 'routine' ? $('csPreferred').value : '' })
      });
      const data = await res.json().catch(() => ({}));
      if (res.status === 429) throw new Error('You can send up to 3 requests an hour. Please try again later, or use chat.');
      if (!res.ok || !data.success) {
        const errs = data.errors || {};
        if (errs.subject) fieldError('csSubject', 'csSubjectErr', errs.subject);
        if (errs.message) fieldError('csMessage', 'csMessageErr', errs.message);
        if (errs.preferred_time) fieldError('csPreferred', 'csPreferredErr', errs.preferred_time);
        throw new Error(Object.keys(errs).length ? 'Please fix the highlighted fields.' : (data.error || 'Could not send the request.'));
      }
      dialog.close();
      toast(data.message);
      document.dispatchEvent(new CustomEvent('consultation-sent', { detail: data }));
    } catch (err) {
      $('consultError').textContent = err.message;
      $('consultError').hidden = false;
    } finally {
      submit.disabled = false;
      submit.textContent = label;
    }
  });
})();
