// Doctor consultation requests (templates/doctor_consultations.html): detail dialog + accept / decline /
// complete. Each row carries its own data as JSON, so opening a request needs no extra request.
(function () {
  const dialog = document.getElementById('csDetail');
  const $ = (id) => document.getElementById(id);
  let current = null, opener = null;

  function badge(cls, text) {
    const s = document.createElement('span');
    s.className = `cs-badge ${cls}`;
    s.textContent = text;
    return s;
  }
  function toast(text, isError) {
    const t = $('csToast');
    t.textContent = text;
    t.classList.toggle('is-error', !!isError);
    t.hidden = false;
    clearTimeout(t._timer);
    t._timer = setTimeout(() => { t.hidden = true; }, 4000);
  }

  function open(data, focus) {
    current = data;
    $('csdTitle').textContent = data.subject;
    $('csdSub').textContent = `${data.patient_name} · ${data.created_full}`;
    $('csdHead').classList.toggle('is-emergency', data.urgency === 'emergency');
    const badges = $('csdBadges');
    badges.innerHTML = '';
    badges.append(badge(data.type_label.startsWith('Emergency') ? 'cs-type-emergency' : 'cs-type', data.type_label),
                  badge(`cs-u-${data.urgency}`, data.urgency_label), badge(`cs-s-${data.status}`, data.status_label));
    const meta = $('csdMeta');
    meta.innerHTML = '';
    const add = (label, value, color) => {
      const span = document.createElement('span');
      span.append(`${label}: `);
      const b = document.createElement('b');
      b.textContent = value;
      if (color) b.style.color = color;
      span.append(b);
      meta.append(span);
    };
    if (data.score !== null && data.score !== undefined) add('Latest composite score', `${data.score}/100 (${data.score_label})`, data.score_color);
    else add('Latest composite score', 'no assessment yet');
    if (data.preferred) add('Preferred time', data.preferred);
    if (data.scheduled) add('Scheduled', data.scheduled);
    $('csdMessage').textContent = data.message;
    const resp = $('csdResponse');
    resp.hidden = !data.response;
    if (data.response) {
      resp.innerHTML = '<span class="cs-response-head"></span>';
      resp.firstChild.textContent = `Your response${data.responded ? ' · ' + data.responded : ''}`;
      resp.append(data.response);
    }
    const pending = data.status === 'pending';
    $('csdForm').hidden = !pending;
    $('csdDoneRow').hidden = pending;
    $('csdComplete').hidden = data.status !== 'accepted';
    $('csdChat').href = $('csdChat2').href = data.chat_url;
    $('csdReply').value = '';
    $('csdWhen').value = '';
    $('csdWhen').min = dialog.dataset.min;
    $('csdError').hidden = true;
    dialog.showModal();
    if (pending) (focus === 'accept' ? $('csdWhen') : $('csdReply')).focus();
  }

  document.querySelectorAll('[data-view]').forEach((btn) => btn.addEventListener('click', () => {
    opener = btn;
    const data = JSON.parse(btn.closest('.cs-row').querySelector('.cs-data').textContent);
    open(data, btn.dataset.focus);
  }));
  dialog.querySelector('[data-close]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => { if (opener) opener.focus(); });

  dialog.querySelectorAll('[data-action]').forEach((btn) => btn.addEventListener('click', async () => {
    const action = btn.dataset.action;
    const err = $('csdError');
    err.hidden = true;
    const response = $('csdReply').value.trim();
    const when = $('csdWhen').value;
    if (action !== 'complete' && !response) { err.textContent = 'Write a short response for the patient first.'; err.hidden = false; $('csdReply').focus(); return; }
    if (action === 'accept' && !when) { err.textContent = 'Choose the date and time of the consultation to accept it.'; err.hidden = false; $('csdWhen').focus(); return; }
    if (action === 'decline' && !confirm('Decline this request? Your response is sent to the patient.')) return;
    btn.disabled = true;
    try {
      const res = await fetch(current.respond_url, {
        method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, response, scheduled_at: when })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.error || 'Could not save your response.');
      dialog.close();
      toast(action === 'complete' ? 'Marked as completed.'
        : `${action === 'accept' ? 'Accepted' : 'Declined'}. ${data.chat_sent ? 'Your reply was also sent in chat.' : 'Saved (chat message could not be sent).'}`);
      if (window.refreshConsultBadge) window.refreshConsultBadge();
      setTimeout(() => location.reload(), 1200);
    } catch (e) {
      if (action === 'complete') toast(e.message, true);
      else { err.textContent = e.message; err.hidden = false; }
    } finally {
      btn.disabled = false;
    }
  }));
})();
