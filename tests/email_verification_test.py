"""Email verification. Run from the project root:  venv\\Scripts\\python tests\\email_verification_test.py
Firebase Auth is mocked (ID tokens, create/update user) and so is the call that asks Firebase to send the
verification email, so no email is ever sent. Activity-log calls are captured in memory. Throwaway zz_ev_* users,
their email_verification records and the zz_ev_migration tree are removed at the end."""
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
import app as appmod  # noqa: E402
from app import app, db  # noqa: E402

results = []


def check(name, ok, extra=''):
    results.append(bool(ok))
    print(('PASS ' if ok else 'FAIL ') + name + (f' -> {extra}' if extra != '' else ''))


# ---------- mocks ----------
appmod.limiter.enabled = False
logs = []
appmod.activity_log = lambda action, label, status='success', details=None, user=None: logs.append(
    {'action': action, 'label': label, 'status': status, 'details': dict(details or {}), 'user': user})
appmod.run_in_background = lambda fn, *a, **k: None   # login counters etc.: not written by tests
appmod.queue_email = lambda *a, **k: None             # the doctor-approved notification

TOKENS = {}   # token -> decoded claims


def token(uid, email, verified, provider='password'):
    t = f'tok-{uid}-{"v" if verified else "u"}-{provider}'
    TOKENS[t] = {'uid': uid, 'email': email, 'email_verified': verified, 'firebase': {'sign_in_provider': provider}}
    return t


def fake_verify_id_token(tok, **kwargs):
    if tok not in TOKENS:
        raise ValueError('invalid token')
    return TOKENS[tok]


verification_sends = []
send_result = {'value': 'sent'}


def fake_send_verification_email(id_token):
    verification_sends.append(id_token)
    return send_result['value']


auth_updates, auth_creates = [], []
appmod.auth.verify_id_token = fake_verify_id_token
appmod.send_verification_email = fake_send_verification_email
appmod.auth.update_user = lambda uid, **kw: auth_updates.append((uid, kw))
appmod.auth.delete_user = lambda uid: None


def fake_create_user(**kw):
    auth_creates.append(kw)
    return SimpleNamespace(uid=f'zz_ev_created_{STAMP}')


appmod.auth.create_user = fake_create_user
appmod.load_approved_doctors = lambda users=None: []   # registration doesn't require choosing a doctor

STAMP = str(int(time.time()))
P, D, LEG, G = f'zz_ev_pat_{STAMP}', f'zz_ev_doc_{STAMP}', f'zz_ev_legacy_{STAMP}', f'zz_ev_google_{STAMP}'
CREATED = f'zz_ev_created_{STAMP}'
ADMIN = 'zz_ev_admin'
MIG = f'zz_ev_migration_{STAMP}'
EMAIL_P, EMAIL_D = f'mohib.pat{STAMP}@example.invalid', f'doc{STAMP}@example.invalid'


def actions(name):
    return [entry for entry in logs if entry['action'] == name]


def register(client, tok, role, **extra):
    body = {'idToken': tok, 'role': role, 'name': 'Zz Verify Test', 'phone': '03001234567'}
    body.update({'age': 30, 'gender': 'Male'} if role == 'patient' else
                {'specialization': 'Psychiatry', 'license_number': 'PMC-12345', 'affiliation': 'Zz Clinic'})
    body.update(extra)
    return client.post('/api/register-profile', json=body)


def login(client, tok, portal):
    return client.post('/api/verify-token', json={'idToken': tok, 'portal': portal})


def admin_client():
    c = app.test_client()
    with c.session_transaction() as s:
        s['user'] = {'uid': ADMIN, 'name': 'Zz Admin', 'role': 'admin', 'status': 'active'}
        s['_admin_csrf_token'] = 'csrf-ok'
        s['_presence_at'] = 9e12
    return c


try:
    # ---------- helpers ----------
    check('mask_email keeps 2 letters + domain', appmod.mask_email('mohib.k@gmail.com') == 'mo*****@gmail.com',
          appmod.mask_email('mohib.k@gmail.com'))
    check('mask_email short local part', appmod.mask_email('ab@x.org') == 'a***@x.org', appmod.mask_email('ab@x.org'))
    check('mask_email rejects junk', appmod.mask_email('not-an-email') == '')

    # ---------- 1. patient registration triggers the verification email ----------
    pc = app.test_client()
    tok_p_unv = token(P, EMAIL_P, False)
    r = register(pc, tok_p_unv, 'patient')
    j = r.get_json()
    check('patient registration succeeds', r.status_code == 200 and j['success'], j)
    check('response asks to verify and points at /verify-email?sent=1',
          j.get('verify_email') is True and j.get('redirect') == '/verify-email?sent=1', j.get('redirect'))
    check('stored email_verified: false', db.reference(f'users/{P}/email_verified').get() is False)
    check('Firebase asked to send the verification email (once, with the new account token)',
          verification_sends == [tok_p_unv], verification_sends)
    sent_logs = actions('email_verification_sent')
    check('email_verification_sent logged with uid', len(sent_logs) == 1 and sent_logs[0]['details'].get('uid') == P
          and sent_logs[0]['details'].get('reason') == 'registration', sent_logs)
    check('log has no email content / address', all('@' not in str(v) for v in sent_logs[0]['details'].values()))
    check('resend slot recorded', isinstance(db.reference(f'email_verification/{P}/last_sent_at').get(), int))
    with pc.session_transaction() as s:
        check('no app session created at registration', 'user' not in s)

    page = pc.get('/verify-email?sent=1').get_data(as_text=True)
    check('/verify-email shows the masked address', 'mo*********' in page and f'@example.invalid' in page and EMAIL_P not in page)
    check('/verify-email has the check-inbox message',
          'We sent a verification link to your email' in page and 'Please click the link to activate your account.' in page)
    check('/verify-email has a resend button and spam tip',
          'Resend verification email' in page and 'Check your spam' in page)
    check('/verify-email starts the 60 s countdown after a fresh send', 'cooldown(INTERVAL_S)' in page)
    check('/verify-email without ?sent=1 has no countdown', 'cooldown(INTERVAL_S)' not in pc.get('/verify-email').get_data(as_text=True))
    check('register page keeps the Firebase sign-in for the verify page',
          'data.verify_email' in open('templates/register.html', encoding='utf-8').read())

    # ---------- 2. unverified login is blocked ----------
    lc = app.test_client()
    logs.clear()
    r = login(lc, tok_p_unv, 'patient')
    j = r.get_json()
    check('unverified patient login refused (403)', r.status_code == 403 and j['success'] is False, r.status_code)
    check('error text', j.get('error') == 'Please verify your email first. Check your inbox or click resend verification.', j.get('error'))
    check('reply flags unverified + verify_url', j.get('unverified') is True and j.get('verify_url') == '/verify-email')
    with lc.session_transaction() as s:
        check('no session user after blocked login', 'user' not in s)
        check('verify page context remembered', (s.get('_verify_pending') or {}).get('uid') == P)
    blocked = actions('login_blocked_unverified')
    check('login_blocked_unverified logged with uid', len(blocked) == 1 and blocked[0]['details'] == {'uid': P}, blocked)
    login_js = open('templates/login.html', encoding='utf-8').read()
    check('login page offers resend (link to verify page, stays signed in)',
          'data.unverified' in login_js and 'Resend verification email' in login_js)

    # ---------- 3. verified login works ----------
    logs.clear()
    tok_p_ver = token(P, EMAIL_P, True)
    r = login(lc, tok_p_ver, 'patient')
    j = r.get_json()
    check('patient login succeeds once Firebase reports verified', r.status_code == 200 and j['success'], j)
    check('flag flipped to true', db.reference(f'users/{P}/email_verified').get() is True)
    done = actions('email_verification_completed')
    check('email_verification_completed logged with uid', len(done) == 1 and done[0]['details'] == {'uid': P}, done)
    with lc.session_transaction() as s:
        check('session created, pending context cleared', s.get('user', {}).get('uid') == P and '_verify_pending' not in s)
    logs.clear()
    r = login(app.test_client(), tok_p_unv, 'patient')
    check('afterwards even an old token signs in (flag is true)', r.status_code == 200 and not actions('login_blocked_unverified'))

    # ---------- legacy + Google ----------
    db.reference(f'users/{LEG}').set({'uid': LEG, 'name': 'Zz Legacy', 'email': 'leg@example.invalid', 'role': 'patient',
                                      'status': 'active', 'created_at': 1})
    r = login(app.test_client(), token(LEG, 'leg@example.invalid', False), 'patient')
    check('older account without the flag is not blocked', r.status_code == 200 and r.get_json()['success'])
    verification_sends.clear()
    r = register(app.test_client(), token(G, 'g@example.invalid', True, 'google.com'), 'patient')
    j = r.get_json()
    check('Google sign-up: verified straight away, no email sent',
          j['success'] and not j.get('verify_email') and db.reference(f'users/{G}/email_verified').get() is True
          and not verification_sends, j)

    # ---------- 4. resend: 1 per minute per uid ----------
    db.reference(f'users/{P}/email_verified').set(False)
    db.reference(f'email_verification/{P}').delete()
    logs.clear()
    verification_sends.clear()
    rc = app.test_client()
    r1 = rc.post('/api/email-verification/resend', json={'idToken': tok_p_unv})
    check('first resend sent', r1.status_code == 200 and r1.get_json().get('retry_after') == 60, r1.get_json())
    r2 = rc.post('/api/email-verification/resend', json={'idToken': tok_p_unv})
    j2 = r2.get_json()
    check('second resend within a minute refused (429)', r2.status_code == 429 and j2.get('rate_limited') is True, j2)
    check('retry_after counts down from 60', 55 <= j2.get('retry_after', 0) <= 60, j2.get('retry_after'))
    check('only one email requested', len(verification_sends) == 1, verification_sends)
    resend_logs = actions('email_verification_sent')
    check('resend logged once, reason resend', len(resend_logs) == 1 and resend_logs[0]['details']['reason'] == 'resend', resend_logs)
    db.reference(f'email_verification/{P}/last_sent_at').set(int(time.time() * 1000) - 61_000)
    r3 = rc.post('/api/email-verification/resend', json={'idToken': tok_p_unv})
    check('allowed again after 60 s', r3.status_code == 200 and len(verification_sends) == 2)
    db.reference(f'email_verification/{P}').delete()
    send_result['value'] = 'error'
    r4 = rc.post('/api/email-verification/resend', json={'idToken': tok_p_unv})
    check('failed send reported (503) and logged as failed',
          r4.status_code == 503 and actions('email_verification_sent')[-1]['status'] == 'failed')
    check('failed send does not use up the minute', db.reference(f'email_verification/{P}/last_sent_at').get() is None)
    send_result['value'] = 'sent'
    check('resend with a bad token asks to sign in',
          rc.post('/api/email-verification/resend', json={'idToken': 'nope'}).get_json().get('needs_login') is True)
    jv = rc.post('/api/email-verification/resend', json={'idToken': tok_p_ver}).get_json()
    check('resend after verifying: already verified, flag set', jv.get('already_verified') is True
          and db.reference(f'users/{P}/email_verified').get() is True, jv)

    # ---------- 5. doctor needs verified email AND approval ----------
    dc = app.test_client()
    tok_d_unv, tok_d_ver = token(D, EMAIL_D, False), token(D, EMAIL_D, True)
    j = register(dc, tok_d_unv, 'doctor').get_json()
    check('doctor registration: pending + unverified', j['success'] and j.get('verify_email')
          and db.reference(f'users/{D}/status').get() == 'pending' and db.reference(f'users/{D}/email_verified').get() is False, j)
    check('/verify-email tells doctors approval is still needed', 'still needs to approve' in dc.get('/verify-email').get_data(as_text=True))
    j = login(app.test_client(), tok_d_unv, 'doctor').get_json()
    check('unverified + pending: told to verify first', j.get('unverified') is True, j)
    ac = admin_client()
    approvals = ac.get('/admin/approvals').get_data(as_text=True)
    check('approvals page warns "Email not yet verified" for this doctor', 'Email not yet verified' in approvals)
    check('dashboard pending card warns too', 'Email not yet verified' in ac.get('/admin').get_data(as_text=True))
    r = ac.post(f'/admin/doctors/{D}/approve', data={'csrf_token': 'csrf-ok'})
    check('admin can approve an unverified doctor', r.status_code in (302, 303) and db.reference(f'users/{D}/status').get() == 'approved')
    j = login(app.test_client(), tok_d_unv, 'doctor').get_json()
    check('approved but unverified: still blocked', j.get('unverified') is True and not j.get('success'), j)
    db.reference(f'users/{D}').update({'status': 'pending', 'email_verified': False})
    j = login(app.test_client(), tok_d_ver, 'doctor').get_json()
    check('verified but pending: blocked by approval', not j['success'] and 'pending approval' in j.get('error', '')
          and not j.get('unverified'), j)
    check('...and the verification was recorded', db.reference(f'users/{D}/email_verified').get() is True)
    db.reference(f'users/{D}/status').set('approved')
    r = login(app.test_client(), tok_d_ver, 'doctor')
    check('verified + approved: doctor signs in', r.status_code == 200 and r.get_json()['success'])

    # ---------- 6. admin bypass ----------
    db.reference(f'users/{P}/email_verified').set(False)
    accounts = ac.get(f'/admin/users?q={EMAIL_P}').get_data(as_text=True)
    check('accounts page has the verification column', 'Email verification</th>' in accounts)
    check('accounts page shows Not verified + Mark as verified',
          'Not verified' in accounts and f'/admin/users/{P}/verify-email' in accounts)
    logs.clear()
    auth_updates.clear()
    check('bypass without CSRF token refused', ac.post(f'/admin/users/{P}/verify-email', data={}).status_code == 400
          and db.reference(f'users/{P}/email_verified').get() is False)
    r = ac.post(f'/admin/users/{P}/verify-email', data={'csrf_token': 'csrf-ok'})
    check('bypass redirects back', r.status_code in (302, 303))
    check('bypass sets the flag (and records who)', db.reference(f'users/{P}/email_verified').get() is True
          and db.reference(f'users/{P}/email_verified_by').get() == ADMIN)
    check('bypass also marks Firebase Auth verified', auth_updates == [(P, {'email_verified': True})], auth_updates)
    by = actions('email_verification_bypassed')
    check('email_verification_bypassed logged with admin_uid + target_uid',
          len(by) == 1 and by[0]['details'] == {'admin_uid': ADMIN, 'target_uid': P}, by)
    r = login(app.test_client(), tok_p_unv, 'patient')
    check('bypassed account can sign in', r.status_code == 200 and r.get_json()['success'])
    pc2 = app.test_client()
    with pc2.session_transaction() as s:
        s['user'] = {'uid': P, 'name': 'Zz', 'role': 'patient', 'status': 'active'}
    check('non-admin cannot use the bypass', pc2.post(f'/admin/users/{D}/verify-email', data={'csrf_token': 'x'}).status_code in (302, 303)
          and not actions('email_verification_bypassed')[1:])

    # ---------- 7. admin-created accounts are auto-verified ----------
    auth_creates.clear()
    r = ac.post('/admin/users/create', data={'csrf_token': 'csrf-ok', 'name': 'Zz Created', 'email': f'created{STAMP}@example.invalid',
                                            'phone': '03001112222', 'password': 'secret123', 'role': 'patient'})
    check('admin create: Firebase user made with email_verified=True',
          len(auth_creates) == 1 and auth_creates[0].get('email_verified') is True, auth_creates)
    check('admin create: stored email_verified true', db.reference(f'users/{CREATED}/email_verified').get() is True)
    r = login(app.test_client(), token(CREATED, f'created{STAMP}@example.invalid', False), 'patient')
    check('admin-created account signs in without clicking a link', r.status_code == 200 and r.get_json()['success'])

    # ---------- 8. grandfather migration runs once ----------
    users_path, flag_path = f'{MIG}/users', f'{MIG}/flag'
    db.reference(users_path).set({
        'a_active': {'role': 'patient', 'status': 'active'},
        'b_approved': {'role': 'doctor', 'status': 'approved'},
        'c_pending': {'role': 'doctor', 'status': 'pending'},
        'd_suspended': {'role': 'patient', 'status': 'suspended'},
        'e_new_unverified': {'role': 'patient', 'status': 'active', 'email_verified': False},
    })
    logs.clear()
    n = appmod.migrate_email_verification(users_path, flag_path)
    got = {u: (v or {}).get('email_verified') for u, v in (db.reference(users_path).get() or {}).items()}
    check('migration marks active + approved accounts', n == 2 and got['a_active'] is True and got['b_approved'] is True, (n, got))
    check('migration leaves pending / suspended alone', got['c_pending'] is None and got['d_suspended'] is None, got)
    check('migration never overwrites an explicit false', got['e_new_unverified'] is False, got)
    flag = db.reference(flag_path).get() or {}
    check('migration flag recorded', flag.get('accounts_marked') == 2 and isinstance(flag.get('done_at'), int), flag)
    db.reference(f'{users_path}/f_later').set({'role': 'patient', 'status': 'active'})
    check('second run is a no-op', appmod.migrate_email_verification(users_path, flag_path) == 'already_done'
          and db.reference(f'{users_path}/f_later/email_verified').get() is None)
    check('migration logged once', len(actions('email_verification_migration')) == 1)
    check('real migration flag path', appmod.EMAIL_VERIFY_MIGRATION == 'system/migrations/email_verification_v1')
    src = open('app.py', encoding='utf-8').read()
    check('migration starts at app startup', 'run_in_background(migrate_email_verification)' in src)
    check('DEPLOY.md documents the Firebase template', os.path.isfile('DEPLOY.md') and
          'Verify your PsyMonitor account' in open('DEPLOY.md', encoding='utf-8').read())
finally:
    db.reference().update({**{f'users/{u}': None for u in (P, D, LEG, G, CREATED)},
                           **{f'email_verification/{u}': None for u in (P, D, LEG, G, CREATED)},
                           **{f'presence/{u}': None for u in (P, D, LEG, G, CREATED, ADMIN)},
                           MIG: None})

print(f'\n{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
