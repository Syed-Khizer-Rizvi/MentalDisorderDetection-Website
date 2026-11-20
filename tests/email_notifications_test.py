"""Email notifications. Run from the project root:  venv\\Scripts\\python tests\\email_notifications_test.py
SMTP is mocked and pointed at a fake local server, so no email is ever sent. Activity-log calls are captured
in memory (nothing is written to activity_logs). Throwaway zz_mail_* users, consultations and throttle records
are removed at the end, and system/settings/emails_enabled is restored."""
import os
import smtplib
import sys
import threading
import time
from unittest import mock

# Fake SMTP settings before the app loads .env (load_dotenv never overrides variables that are already set),
# so the real Gmail credentials are never used by this process
FAKE_SMTP = {'SMTP_SERVER': '127.0.0.1', 'SMTP_PORT': '2525', 'SMTP_USERNAME': 'sender@psymonitor-tests.dev',
             'SMTP_PASSWORD': 'abcdefghijklmnop', 'SMTP_FROM_NAME': 'PsyMonitor Test'}
os.environ.update(FAKE_SMTP)
sys.path.insert(0, os.getcwd())
import app as appmod  # noqa: E402
from app import app, db  # noqa: E402

results = []


def check(name, ok, extra=''):
    results.append(bool(ok))
    print(('PASS ' if ok else 'FAIL ') + name + (f' -> {extra}' if extra != '' else ''))


# ---------- mocks ----------
sent, connections = [], []
smtp_mode = {'fail': False, 'gate': None}


class FakeSMTP:
    def __init__(self, host, port, timeout=None, context=None):
        connections.append((host, port))
        self.tls, self.login_as = False, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo(self):
        pass

    def starttls(self, context=None):
        self.tls = True

    def login(self, user, password):
        self.login_as = (user, password)

    def send_message(self, message):
        if smtp_mode['gate'] is not None:
            smtp_mode['gate'].wait(10)
        if smtp_mode['fail']:
            raise smtplib.SMTPServerDisconnected('connection lost')
        html = message.get_body(('html',)).get_content()
        text = message.get_body(('plain',)).get_content()
        sent.append({'to': message['To'], 'subject': message['Subject'], 'from': message['From'], 'html': html,
                     'text': text, 'raw': message.as_string(), 'tls': self.tls, 'login': self.login_as})


appmod.smtplib.SMTP = FakeSMTP
appmod.smtplib.SMTP_SSL = FakeSMTP

logs = []
appmod.activity_log = lambda action, label, status='success', details=None, user=None: logs.append(
    {'action': action, 'label': label, 'status': status, 'details': dict(details or {}), 'user': user})

pending = []
_submit = appmod._WRITE_POOL.submit


def tracking_submit(fn, *args, **kwargs):
    future = _submit(fn, *args, **kwargs)
    pending.append(future)
    return future


appmod._WRITE_POOL.submit = tracking_submit


def drain():
    """Waits for every background job (including jobs queued by jobs)."""
    while True:
        waiting = [f for f in pending if not f.done()]
        if not waiting:
            return
        for f in waiting:
            f.result(timeout=60)


def reset():
    drain()
    sent.clear()
    logs.clear()
    connections.clear()


def email_logs():
    return [entry for entry in logs if entry['action'] == 'email_sent']


appmod.limiter.enabled = False
st = str(int(time.time()))
D, DP, P, PX = (f'zz_mail_{k}_{st}' for k in ('doc', 'pending', 'pat', 'patx'))
DOMAIN = 'psymonitor-tests.dev'   # never reached: SMTP is mocked and points at 127.0.0.1
ADDR = {D: f'{D}@{DOMAIN}', DP: f'{DP}@{DOMAIN}', P: f'{P}@{DOMAIN}', PX: f'{PX}@example.invalid'}
SECRET_OBS = f'SECRET-OBSERVATION-{st}'
INSTRUCTIONS = ('Breathe slowly for five minutes every morning before breakfast, then write three lines in your '
                'journal about how you slept. Keep screens away for an hour before bed. TAIL-' + st)
saved = {'flag': db.reference('system/settings/emails_enabled').get(), 'meta': db.reference('system/settings/_meta/emails').get()}
consultation_ids = []


def as_user(uid, role, status='active', name='Zz', **extra):
    client = app.test_client()
    with client.session_transaction() as s:
        s['user'] = {'uid': uid, 'name': name, 'role': role, 'status': status}
        s['_presence_at'] = 9e12
        s.update(extra)
    return client


def set_emails_enabled(value):
    db.reference('system/settings/emails_enabled').set(value)
    appmod.system_settings(refresh=True)


try:
    db.reference().update({
        f'users/{D}': {'role': 'doctor', 'status': 'approved', 'name': 'Dr. Zz Mail', 'email': ADDR[D], 'created_at': 1,
                       'profile': {'specialization': 'Psychiatry', 'affiliation': 'Zz Clinic'}},
        f'users/{DP}': {'role': 'doctor', 'status': 'pending', 'name': 'Zz Newdoc', 'email': ADDR[DP], 'created_at': 1,
                        'profile': {'specialization': 'Clinical Psychology', 'affiliation': 'Zz General Hospital',
                                    'license_number': '12345-P'}},
        f'users/{P}': {'role': 'patient', 'status': 'active', 'name': 'Ayesha Mailtest', 'email': ADDR[P], 'created_at': 1,
                       'assigned_doctor_uid': D},
        f'users/{PX}': {'role': 'patient', 'status': 'active', 'name': 'Zz Reserved', 'email': ADDR[PX], 'created_at': 1,
                        'assigned_doctor_uid': D},
    })
    set_emails_enabled(True)

    # ---------- send_email helper ----------
    status = appmod.send_email('someone@' + DOMAIN, 'Hello\nthere', '<p>Hi</p>', 'Hi')
    m = sent[-1] if sent else {}
    check('send_email: STARTTLS + login with the .env credentials', status == 'sent' and m.get('tls')
          and m.get('login') == (FAKE_SMTP['SMTP_USERNAME'], FAKE_SMTP['SMTP_PASSWORD'])
          and connections == [('127.0.0.1', 2525)], (status, connections))
    check('send_email: From uses SMTP_FROM_NAME, HTML + plain text parts, no line break in the subject',
          m.get('from') == 'PsyMonitor Test <sender@psymonitor-tests.dev>' and m.get('subject') == 'Hello there'
          and '<p>Hi</p>' in m.get('html', '') and m.get('text', '').strip() == 'Hi')
    reset()
    with mock.patch.dict(os.environ, {'SMTP_PASSWORD': ''}), mock.patch.object(app.logger, 'warning') as warn:
        status = appmod.send_email('someone@' + DOMAIN, 'x', '<p>x</p>', 'x')
    check('missing SMTP setting: warning logged, send skipped, no crash',
          status == 'not_configured' and not connections and warn.called and 'SMTP_PASSWORD' in str(warn.call_args), status)
    with mock.patch.dict(os.environ, {'SMTP_PORT': 'abc'}):
        check('non-numeric port counts as missing', appmod.send_email('a@' + DOMAIN, 'x', 'x', 'x') == 'not_configured')
    check('reserved test domains are never mailed', all(appmod.undeliverable_address(a) for a in
          ('a@example.invalid', 'b@x.test', 'c@example.com', 'd@example.org', 'no-at-sign', '@x.dev', 'e@'))
          and not appmod.undeliverable_address('f@gmail.com') and not appmod.undeliverable_address(ADDR[D]))

    # ---------- A) doctor approved ----------
    reset()
    admin = as_user('zz_mail_admin', 'admin', name='Zz Admin', _admin_csrf_token='tok-' + st)
    with mock.patch.object(appmod.auth, 'update_user', return_value=None):
        r = admin.post(f'/admin/doctors/{DP}/approve', data={'csrf_token': 'tok-' + st})
    drain()
    mail = [s for s in sent if s['to'] == ADDR[DP]]
    m = mail[0] if mail else {}
    check('A: approval redirects as before and sends one email to the doctor', r.status_code == 302 and len(mail) == 1, (r.status_code, len(sent)))
    check('A: subject', m.get('subject') == 'Your PsyMonitor doctor account has been approved', m.get('subject'))
    check('A: body has name, sign-in URL, specialty and hospital (HTML + text)',
          all(x in m.get('html', '') and x in m.get('text', '') for x in
              ('Dr. Zz Newdoc', 'http://localhost/doctor/login', 'Clinical Psychology', 'Zz General Hospital')))
    check('A: license number is not emailed', '12345-P' not in m.get('raw', ''))
    el = email_logs()
    check('A: logged as email_sent (event_type, to_uid, result) by System',
          len(el) == 1 and el[0]['details'] == {'event_type': 'doctor_approved', 'to_uid': DP, 'result': 'sent'}
          and el[0]['status'] == 'success' and el[0]['user'] == appmod.SYSTEM_ACTOR, el)

    # ---------- B) consultation request ----------
    reset()
    pat = as_user(P, 'patient', name='Ayesha Mailtest')
    body = {'type': 'consultation', 'urgency': 'routine', 'subject': 'Trouble sleeping',
            'message': 'PRIVATE-MSG I have not slept well for two weeks.'}
    smtp_mode['gate'] = threading.Event()   # holds the SMTP send open
    started = time.time()
    r = pat.post('/api/consultation/request', json=body)
    returned_first = r.status_code == 200 and not sent
    smtp_mode['gate'].set()
    smtp_mode['gate'] = None
    drain()
    check('background send: the request returns while SMTP is still busy', returned_first and time.time() - started < 10)
    if r.is_json and r.get_json().get('id'):
        consultation_ids.append(r.get_json()['id'])
    m = sent[0] if sent else {}
    check('B: one email to the assigned doctor', len(sent) == 1 and m.get('to') == ADDR[D], len(sent))
    check('B: subject', m.get('subject') == 'New consultation request from Ayesha Mailtest', m.get('subject'))
    check('B: body has patient name, subject, time and the consultations link',
          all(x in m.get('html', '') and x in m.get('text', '') for x in
              ('Ayesha Mailtest', 'Trouble sleeping', 'http://localhost/doctor/consultations', time.strftime('%d %b %Y'))))
    check("B: the patient's message stays in the app", 'PRIVATE-MSG' not in m.get('raw', ''))
    check('B: greets the doctor by name', 'Hi Dr. Zz Mail,' in m.get('text', ''))
    stamp = db.reference(f'email_throttle/{D}/consultation_received').get()
    check('B: send time recorded at email_throttle/{uid}/{event_type}', isinstance(stamp, int) and abs(stamp - time.time() * 1000) < 120_000, stamp)

    # rate limit
    reset()
    r = pat.post('/api/consultation/request', json={**body, 'subject': 'Second request'})
    if r.is_json and r.get_json().get('id'):
        consultation_ids.append(r.get_json()['id'])
    drain()
    check('rate limit: a second request within 10 minutes still succeeds but sends no email',
          r.status_code == 200 and not sent and not email_logs(), (r.status_code, len(sent)))
    db.reference(f'email_throttle/{D}/consultation_received').set(int(time.time() * 1000) - 11 * 60 * 1000)
    r = pat.post('/api/consultation/request', json={**body, 'subject': 'Third request'})
    if r.is_json and r.get_json().get('id'):
        consultation_ids.append(r.get_json()['id'])
    drain()
    check('rate limit: allowed again once 10 minutes have passed', len(sent) == 1 and sent[0]['subject'].startswith('New consultation'), len(sent))

    # ---------- C) emergency request ----------
    reset()
    emergency = {'type': 'emergency_appointment', 'subject': 'I need help now', 'message': 'EMERGENCY-MSG I feel unsafe tonight.'}
    ids = []
    for _ in range(2):
        r = pat.post('/api/consultation/request', json=emergency)
        if r.is_json and r.get_json().get('id'):
            ids.append(r.get_json()['id'])
    consultation_ids.extend(ids)
    drain()
    check('C: emergency bypasses the rate limit (two requests, two emails)', len(sent) == 2, len(sent))
    m = sent[0] if sent else {}
    check('C: subject', m.get('subject') == 'URGENT: Emergency request from Ayesha Mailtest', m.get('subject'))
    check('C: body has patient name, message, time', all(x in m.get('html', '') and x in m.get('text', '') for x in
          ('Ayesha Mailtest', 'EMERGENCY-MSG I feel unsafe tonight.', time.strftime('%d %b %Y'))))
    check('C: direct link opens the request', bool(ids) and all(f'http://localhost/doctor/consultations?open={i}' in ''.join(s['html'] for s in sent) for i in ids), ids)
    check('C: red urgent styling', 'background:#B91C1C' in m.get('html', '') and 'EMERGENCY APPOINTMENT REQUEST' in m.get('html', '')
          and 'background:#DC2626' in m.get('html', ''))
    check('C: no throttle record for emergencies', db.reference(f'email_throttle/{D}/emergency_request').get() is None)
    doc = as_user(D, 'doctor', 'approved', name='Dr. Zz Mail')
    h = doc.get('/doctor/consultations').get_data(as_text=True)
    check('C: the consultations page has the rows the link opens', bool(ids) and f'id="request-{ids[0]}"' in h)
    js = open(os.path.join('static', 'js', 'doctor_consultations.js'), encoding='utf-8').read()
    check('C: the page script opens ?open=<id>', "get('open')" in js and 'request-${wanted}' in js)

    # ---------- D) risk alert ----------
    def submit_assessment(scores):
        phq, gad, pss = scores
        client = as_user(P, 'patient', name='Ayesha Mailtest',
                         assessment={'phq_score': phq, 'gad_score': gad, 'pss_score': pss})
        return client.post('/api/submit-facial', json={'skipped': True})

    for label, scores in (('moderate', (9, 7, 18)), ('high', (18, 14, 28)), ('critical', (25, 20, 38))):
        level = appmod.risk_band(appmod.compute_composite({'phq_score': scores[0], 'gad_score': scores[1], 'pss_score': scores[2], 'facial_skipped': True}))[0]
        check(f'D: {label} sample really is {label}', level == label, level)
    reset()
    r = submit_assessment((9, 7, 18))
    drain()
    check('D: a Moderate result sends no risk email', r.status_code == 200 and not sent, len(sent))
    r = submit_assessment((18, 14, 28))
    drain()
    m = sent[0] if sent else {}
    high_score = appmod.compute_composite({'phq_score': 18, 'gad_score': 14, 'pss_score': 28, 'facial_skipped': True})
    check('D: a High result emails the assigned doctor', r.status_code == 200 and len(sent) == 1 and m.get('to') == ADDR[D], len(sent))
    check('D: subject', m.get('subject') == 'Risk alert for Ayesha Mailtest', m.get('subject'))
    check('D: body has patient name, band, score and the risk alerts link',
          all(x in m.get('html', '') and x in m.get('text', '') for x in
              ('Ayesha Mailtest', 'High', str(high_score), 'http://localhost/doctor/risk-alerts'))
          and f'{high_score}/100' in m.get('text', ''))
    db.reference(f'email_throttle/{D}/risk_alert').delete()
    reset()
    submit_assessment((25, 20, 38))
    drain()
    m = sent[0] if sent else {}
    check('D: Critical too, with critical styling', len(sent) == 1 and 'Critical' in m.get('text', '') and 'CRITICAL RISK' in m.get('html', ''))
    reset()
    submit_assessment((25, 20, 38))
    drain()
    check('D: rate limited like the other types', not sent)

    # ---------- E) treatment plan ----------
    db.reference(f'email_throttle/{P}').delete()
    reset()
    plan = {'observations': SECRET_OBS, 'instructions': INSTRUCTIONS, 'recommended_activities': ['breathing'],
            'recommended_games': [], 'follow_up_date': ''}
    r = doc.post(f'/doctor/patients/{P}/treatment-plan', json=plan)
    drain()
    m = sent[0] if sent else {}
    check('E: saving a plan emails the patient', r.status_code == 200 and len(sent) == 1 and m.get('to') == ADDR[P], (r.status_code, len(sent)))
    check('E: subject', m.get('subject') == 'Your treatment plan has been updated', m.get('subject'))
    preview = INSTRUCTIONS[:150].rstrip()
    check('E: doctor name + first 150 characters of the instructions + dashboard link',
          all(x in m.get('html', '') and x in m.get('text', '') for x in ('Dr. Zz Mail', preview + '…', 'http://localhost/patient/dashboard'))
          and 'TAIL-' + st not in m.get('raw', ''))
    check("E: the doctor's private observations never reach the patient", SECRET_OBS not in m.get('raw', '')
          and all(SECRET_OBS not in str(e['details']) for e in logs))
    check('E: greets the patient by name', 'Hi Ayesha Mailtest,' in m.get('text', ''))

    # reserved address: no send, no throttle record, no log
    reset()
    r = doc.post(f'/doctor/patients/{PX}/treatment-plan', json=plan)
    drain()
    check('reserved test address: plan saved, nothing sent or logged', r.status_code == 200 and not sent and not email_logs()
          and db.reference(f'email_throttle/{PX}').get() is None)

    # ---------- emails_enabled ----------
    reset()
    set_emails_enabled(False)
    r = pat.post('/api/consultation/request', json=emergency)
    if r.is_json and r.get_json().get('id'):
        consultation_ids.append(r.get_json()['id'])
    db.reference(f'email_throttle/{P}').delete()
    r2 = doc.post(f'/doctor/patients/{P}/treatment-plan', json=plan)
    drain()
    check('emails_enabled=false: no emails go out (not even emergencies), requests still work',
          r.status_code == 200 and r2.status_code == 200 and not sent and not connections and not email_logs())
    set_emails_enabled(True)

    # admin toggle
    h = admin.get('/admin/settings').get_data(as_text=True)
    check('Admin Settings shows the toggle', 'name="emails.enabled"' in h and 'Email Notifications' in h and 'id="emails"' in h)
    r = admin.post('/admin/settings/save', data={'csrf_token': 'tok-' + st, 'section': 'emails'})   # switch unchecked
    stored = db.reference('system/settings/emails_enabled').get()
    check('turning it off saves false at system/settings/emails_enabled', r.status_code == 302 and stored is False
          and db.reference('system/settings/emails').get() is None and appmod.emails_enabled() is False, stored)
    r = admin.post('/admin/settings/save', data={'csrf_token': 'tok-' + st, 'section': 'emails', 'emails.enabled': '1'})
    check('turning it back on saves true', r.status_code == 302 and db.reference('system/settings/emails_enabled').get() is True
          and appmod.emails_enabled() is True)
    check('default is on when the setting is missing', appmod.clean_settings_section('emails', {})['enabled'] is True)

    # ---------- failures never break the request ----------
    reset()
    smtp_mode['fail'] = True
    r = pat.post('/api/consultation/request', json=emergency)
    if r.is_json and r.get_json().get('id'):
        consultation_ids.append(r.get_json()['id'])
    drain()
    smtp_mode['fail'] = False
    el = email_logs()
    check('SMTP failure: the request still succeeds', r.status_code == 200 and r.get_json().get('success') is True, r.status_code)
    check('SMTP failure: logged as a failed email_sent', len(el) == 1 and el[0]['status'] == 'failed'
          and el[0]['details'] == {'event_type': 'emergency_request', 'to_uid': D, 'result': 'failed'}, el)
    reset()
    with mock.patch.object(appmod.smtplib, 'SMTP', side_effect=ConnectionRefusedError('refused')):
        db.reference(f'email_throttle/{P}').delete()
        r = doc.post(f'/doctor/patients/{P}/treatment-plan', json=plan)
        drain()
    el = email_logs()
    check('unreachable SMTP server: plan still saved, failure logged, window not used up', r.status_code == 200
          and len(el) == 1 and el[0]['details']['result'] == 'failed'
          and db.reference(f'email_throttle/{P}/treatment_plan_updated').get() is None, el)
    reset()
    with mock.patch.dict(os.environ, {'SMTP_SERVER': ''}):
        r = pat.post('/api/consultation/request', json=emergency)
        if r.is_json and r.get_json().get('id'):
            consultation_ids.append(r.get_json()['id'])
        drain()
    el = email_logs()
    check('SMTP not configured: request succeeds, logged as not_configured', r.status_code == 200 and not connections
          and len(el) == 1 and el[0]['details']['result'] == 'not_configured' and el[0]['status'] == 'failed', el)

    # ---------- the log never holds addresses or bodies ----------
    reset()
    r = pat.post('/api/consultation/request', json=emergency)
    if r.is_json and r.get_json().get('id'):
        consultation_ids.append(r.get_json()['id'])
    drain()
    el = email_logs()
    check('log entries: only event_type, to_uid and result; no address or message text',
          el and all(set(e['details']) == {'event_type', 'to_uid', 'result'} and '@' not in str(e['details']) + e['label']
                     and 'EMERGENCY-MSG' not in str(e) for e in el), el)
finally:
    drain()
    smtp_mode['gate'] = None
    updates = {f'system/settings/emails_enabled': saved['flag'], 'system/settings/_meta/emails': saved['meta']}
    for uid in (D, DP, P, PX):
        updates.update({f'users/{uid}': None, f'presence/{uid}': None, f'email_throttle/{uid}': None})
    for cid in consultation_ids:
        updates[f'consultations/{cid}'] = None
    db.reference().update(updates)
    appmod.system_settings(refresh=True)
    left = [cid for cid, c in (db.reference('consultations').get() or {}).items()
            if isinstance(c, dict) and c.get('patient_uid') in (P, PX)]
    check('cleanup: test consultations, users and throttle records removed', not left
          and all(db.reference(f'email_throttle/{u}').get() is None for u in (D, DP, P, PX)), left)

print(f'\n{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
