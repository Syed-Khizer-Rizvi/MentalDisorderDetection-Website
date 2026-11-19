"""Custom 404 / 500 pages. Run from the project root:  venv\\Scripts\\python tests\\error_pages_test.py"""
import os
import sys

sys.path.insert(0, os.getcwd())
import app as appmod  # noqa: E402
from app import app  # noqa: E402

results = []


def check(name, ok, extra=''):
    results.append(bool(ok))
    print(('PASS ' if ok else 'FAIL ') + name + (f' -> {extra}' if extra != '' else ''))


appmod.limiter.enabled = False


def boom():
    raise RuntimeError('test failure')


def db_down():
    def fail(*a, **k):
        raise ConnectionError('database unreachable')
    appmod.db.reference = fail   # restored below
    raise RuntimeError('db down')


# Test-only routes, added before the first request
app.add_url_rule('/zz-test-boom', 'zz_test_boom', boom)
app.add_url_rule('/zz-test-db-down', 'zz_test_db_down', db_down)
real_reference = appmod.db.reference

anon = app.test_client()
signed = app.test_client()
with signed.session_transaction() as s:
    s['user'] = {'uid': 'zz_err_pat', 'name': 'Zz', 'role': 'patient', 'status': 'active'}
    s['_presence_at'] = 9e12

r = anon.get('/no-such-page-zz')
h = r.get_data(as_text=True)
check('404 status + custom page', r.status_code == 404 and 'Page not found' in h and 'ERROR 404' in h, r.status_code)
check('404 for visitors: home + sign in + help', 'Back to home' in h and 'href="/login"' in h and 'href="/help"' in h)
check('404 has site branding, favicon, ui.css, title', 'Mental Disorder Detection' in h and 'favicon-32.png' in h and 'css/ui.css' in h
      and '<title>Page not found | Mental Disorder Detection</title>' in h and 'Plus+Jakarta+Sans' in h)
h = signed.get('/patient/no-such-page').get_data(as_text=True)
check('404 for signed-in users links to the dashboard', 'Go to my dashboard' in h and 'href="/dashboard"' in h)
r = anon.get('/api/no-such-endpoint')
check('API 404 stays JSON', r.status_code == 404 and r.is_json and r.get_json()['success'] is False)

r = anon.get('/zz-test-boom')
h = r.get_data(as_text=True)
check('500 status + custom page', r.status_code == 500 and 'Something went wrong' in h and 'ERROR 500' in h, r.status_code)
check('500 has support email + home link', 'psymonitor.support@gmail.com' in h and 'mailto:psymonitor.support@gmail.com' in h
      and 'Back to home' in h)
check('500 never shows the exception', 'RuntimeError' not in h and 'test failure' not in h and 'Traceback' not in h)
h = signed.get('/zz-test-boom').get_data(as_text=True)
check('500 for signed-in users links to the dashboard', 'Go to my dashboard' in h)
r = anon.get('/zz-test-boom', headers={'Accept': 'application/json'})
check('500 JSON for API clients', r.status_code == 500 and r.is_json)
try:
    r = signed.get('/zz-test-db-down')
    check('500 page still renders when the database is down', r.status_code == 500 and 'Something went wrong' in r.get_data(as_text=True))
finally:
    appmod.db.reference = real_reference

check('real pages unaffected', anon.get('/').status_code == 200 and anon.get('/login').status_code == 200)

print(f'\n{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
