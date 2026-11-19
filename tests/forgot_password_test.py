"""Forgot password flow. Run from the project root:  venv\\Scripts\\python tests\\forgot_password_test.py
The Firebase email call is replaced by a stub, so no email is sent. Log entries it creates are removed."""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.getcwd())
import app as appmod  # noqa: E402
from app import app, db  # noqa: E402

results = []


def check(name, ok, extra=''):
    results.append(bool(ok))
    print(('PASS ' if ok else 'FAIL ') + name + (f' -> {extra}' if extra != '' else ''))


T0 = int(time.time() * 1000)
EMAIL = f'zz.reset.{T0}@example.invalid'
sent_to, reply = [], {'value': 'sent'}
real_send = appmod.send_password_reset_email
appmod.send_password_reset_email = lambda email: (sent_to.append(email), reply['value'])[1]


def page_token(client, query=''):
    h = client.get('/forgot-password' + query).get_data(as_text=True)
    m = re.search(r'name="csrf_token" value="([^"]+)"', h)
    return h, (m.group(1) if m else '')


try:
    # ---- the links ----
    for portal in ('patient', 'doctor', 'admin'):
        h = app.test_client().get(f'/{portal}/login').get_data(as_text=True)
        check(f'{portal} sign-in links to the reset page', f'href="/forgot-password?portal={portal}"' in h and 'Forgot password?' in h
              and 'sendPasswordResetEmail' not in h and 'resetModal' not in h)

    # ---- the page ----
    appmod.limiter.enabled = False
    c = app.test_client()
    h, token = page_token(c, '?portal=doctor')
    check('reset page renders with a form + token', 'Reset your password' in h and 'type="email"' in h and token
          and 'name="portal" value="doctor"' in h and 'href="/doctor/login"' in h)
    check('GET does not count toward the limit (no limiter on GET)', all(app.test_client().get('/forgot-password').status_code == 200 for _ in range(5)))

    # ---- validation ----
    r = c.post('/forgot-password', data={'email': EMAIL})
    check('missing token rejected', r.status_code == 400 and 'session expired' in r.get_data(as_text=True) and not sent_to)
    for bad in ('', 'not-an-email', 'a@b', 'x' * 250 + '@example.com'):
        r = c.post('/forgot-password', data={'csrf_token': token, 'email': bad})
        check(f'invalid email rejected: {bad[:20]!r}', r.status_code == 400 and 'valid email' in r.get_data(as_text=True))
    check('nothing sent for invalid input', not sent_to)

    # ---- success: same answer whether or not the account exists ----
    r = c.post('/forgot-password', data={'csrf_token': token, 'email': EMAIL, 'portal': 'doctor'})
    check('valid request redirects to the success page', r.status_code == 302 and 'sent=1' in r.headers['Location']
          and 'portal=doctor' in r.headers['Location'] and sent_to == [EMAIL])
    h = c.get(r.headers['Location']).get_data(as_text=True)
    check('success message', 'If this email exists, a reset link has been sent. Check your inbox.' in h and 'href="/doctor/login"' in h
          and EMAIL not in h)

    # ---- Firebase replies ----
    reply['value'] = 'busy'
    r = c.post('/forgot-password', data={'csrf_token': token, 'email': EMAIL})
    check('Firebase throttling -> friendly retry message', r.status_code == 503 and 'wait a few minutes' in r.get_data(as_text=True))
    reply['value'] = 'error'
    r = c.post('/forgot-password', data={'csrf_token': token, 'email': EMAIL})
    check('Firebase error -> friendly message, email kept in the box', r.status_code == 503 and 'couldn&#39;t send' in r.get_data(as_text=True)
          and f'value="{EMAIL}"' in r.get_data(as_text=True))
    reply['value'] = 'sent'

    # ---- real Firebase reply parsing (no email is sent: the HTTP call is faked) ----
    import io
    import urllib.error
    import urllib.request
    real_urlopen = urllib.request.urlopen
    try:
        for code, expected in (('EMAIL_NOT_FOUND', 'sent'), ('RESET_PASSWORD_EXCEED_LIMIT', 'busy'),
                               ('TOO_MANY_ATTEMPTS_TRY_LATER', 'busy'), ('INVALID_EMAIL', 'error')):
            def fake(req, timeout=None, code=code):
                raise urllib.error.HTTPError(req.full_url, 400, 'Bad Request', {},
                                             io.BytesIO(json.dumps({'error': {'message': code}}).encode()))
            urllib.request.urlopen = fake
            check(f'Firebase {code} -> {expected}' + (' (unknown address: same reply as a real one)' if code == 'EMAIL_NOT_FOUND' else ''), real_send(EMAIL) == expected)
        captured = {}

        def ok(req, timeout=None):
            captured['url'], captured['body'] = req.full_url, json.loads(req.data.decode())
            return io.BytesIO(b'{"email": "x"}')
        urllib.request.urlopen = ok
        check('success -> sent, PASSWORD_RESET request to Firebase Auth', real_send(EMAIL) == 'sent'
              and 'identitytoolkit.googleapis.com/v1/accounts:sendOobCode' in captured['url']
              and captured['body'] == {'requestType': 'PASSWORD_RESET', 'email': EMAIL})

        def down(req, timeout=None):
            raise urllib.error.URLError('offline')
        urllib.request.urlopen = down
        check('network failure -> error', real_send(EMAIL) == 'error')
    finally:
        urllib.request.urlopen = real_urlopen

    # ---- rate limit: 3 per 15 minutes per IP ----
    appmod.limiter.enabled = True
    appmod.limiter.reset()
    rl = app.test_client()
    _, tok = page_token(rl)
    ip = {'REMOTE_ADDR': '203.0.113.77'}
    codes = [rl.post('/forgot-password', data={'csrf_token': tok, 'email': EMAIL}, environ_base=ip).status_code for _ in range(4)]
    check('4th request in 15 minutes -> 429', codes == [302, 302, 302, 429], codes)
    other = app.test_client()
    _, tok2 = page_token(other)
    check('limit is per IP', other.post('/forgot-password', data={'csrf_token': tok2, 'email': EMAIL},
                                        environ_base={'REMOTE_ADDR': '203.0.113.78'}).status_code == 302)
    check('limit shown on the Security page list', appmod.RATE_LIMITS.get('forgot_password_submit') == '3 per 15 minutes')

    # ---- activity log: no email address ----
    time.sleep(2)
    logs = db.reference('activity_logs').order_by_key().start_at(appmod.push_key_prefix(T0 - 60000)).get() or {}
    mine = {k: v for k, v in logs.items() if v.get('action') == 'password_reset_requested' and v.get('timestamp', 0) >= T0}
    check('password_reset_requested logged', len(mine) >= 7, len(mine))
    check('the email address is never logged', not any(EMAIL in json.dumps(v) or 'zz.reset' in json.dumps(v) for v in mine.values()))
    check('result recorded (sent / busy / error)', {v['details']['result'] for v in mine.values()} >= {'sent', 'busy', 'error'})
finally:
    appmod.send_password_reset_email = real_send
    appmod.limiter.enabled = True
    time.sleep(1)
    logs = db.reference('activity_logs').order_by_key().start_at(appmod.push_key_prefix(T0 - 60000)).get() or {}
    rm = {f'activity_logs/{k}': None for k, v in logs.items() if v.get('timestamp', 0) >= T0
          and (v.get('action') == 'password_reset_requested' or (v.get('action') == 'rate_limit_exceeded'
                                                                 and (v.get('details') or {}).get('path') == '/forgot-password'))}
    if rm:
        db.reference().update(rm)
    check('cleanup', True, f'{len(rm)} log entries removed')

print(f'\n{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
