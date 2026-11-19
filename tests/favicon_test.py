"""Favicons on every page. Run from the project root:  venv\\Scripts\\python tests\\favicon_test.py
Renders pages as throwaway zz_fav_* users (removed at the end)."""
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import app as appmod  # noqa: E402
from app import app, db  # noqa: E402
from PIL import Image  # noqa: E402

results = []


def check(name, ok, extra=''):
    results.append(bool(ok))
    print(('PASS ' if ok else 'FAIL ') + name + (f' -> {extra}' if extra != '' else ''))


appmod.limiter.enabled = False
TAGS = ('rel="icon" type="image/png" sizes="32x32" href="/static/images/favicon-32.png"',
        'rel="icon" type="image/png" sizes="16x16" href="/static/images/favicon-16.png"',
        'rel="apple-touch-icon" sizes="180x180" href="/static/images/apple-touch-icon.png"')

# ---- the files themselves ----
for path, size in (('static/images/favicon-16.png', (16, 16)), ('static/images/favicon-32.png', (32, 32)),
                   ('static/images/favicon-192.png', (192, 192)), ('static/images/apple-touch-icon.png', (180, 180))):
    ok = os.path.exists(path) and Image.open(path).size == size
    check(f'{os.path.basename(path)} is {size[0]}x{size[1]}', ok)
check('apple-touch-icon has no transparency (iOS would show black)', Image.open('static/images/apple-touch-icon.png').mode == 'RGB')
ico = Image.open('static/favicon.ico')
check('favicon.ico holds 16/32/48', {(16, 16), (32, 32), (48, 48)} <= set(ico.info.get('sizes', set())), ico.info.get('sizes'))

c = app.test_client()
r = c.get('/favicon.ico')
check('/favicon.ico served', r.status_code == 200 and r.mimetype == 'image/vnd.microsoft.icon' and r.data[:4] == b'\x00\x00\x01\x00')
for p in ('/static/images/favicon-32.png', '/static/images/apple-touch-icon.png'):
    check(f'{p} served', c.get(p).status_code == 200)

# ---- every page carries the tags ----
st = str(int(time.time()))
D, P = f'zz_fav_doc_{st}', f'zz_fav_pat_{st}'
try:
    db.reference().update({
        f'users/{D}': {'role': 'doctor', 'status': 'approved', 'name': 'Dr. Zz Fav', 'email': 'f@example.invalid', 'created_at': 1},
        f'users/{P}': {'role': 'patient', 'status': 'active', 'name': 'Zz Fav', 'email': 'g@example.invalid', 'created_at': 1, 'assigned_doctor_uid': D},
    })

    def as_user(uid, role, status='active'):
        cl = app.test_client()
        with cl.session_transaction() as s:
            s['user'] = {'uid': uid, 'name': 'Zz', 'role': role, 'status': status}
            s['_presence_at'] = 9e12
        return cl
    pages = {
        'public': (app.test_client(), ['/', '/login', '/register', '/patient/login', '/doctor/login', '/admin/login',
                                       '/patient/register', '/doctor/register', '/forgot-password', '/help', '/no-such-page']),
        'patient': (as_user(P, 'patient'), ['/patient/dashboard', '/patient/history', '/patient/activities', '/patient/profile',
                                            '/patient/wristband', '/patient/consultations', '/patient/chat', '/patient/games/breathing']),
        'doctor': (as_user(D, 'doctor', 'approved'), ['/doctor/dashboard', '/doctor/patients', '/doctor/monitor', '/doctor/consultations',
                                                      f'/doctor/patients/{P}', f'/doctor/patients/{P}/treatment-plan']),
        'admin': (as_user('zz_fav_admin', 'admin'), ['/admin', '/admin/users', '/admin/settings', '/admin/security']),
    }
    for role, (cl, urls) in pages.items():
        missing = []
        for u in urls:
            h = cl.get(u).get_data(as_text=True)
            if not all(t in h for t in TAGS) or 'images/logo.png\') }}' in h:
                missing.append(u)
        check(f'{role}: favicon tags on all {len(urls)} pages', not missing, missing)
    landing = app.test_client().get('/').get_data(as_text=True)
    check('landing page favicon (was missing before)', all(t in landing for t in TAGS))
finally:
    db.reference().update({f'users/{D}': None, f'users/{P}': None, f'presence/{D}': None, f'presence/{P}': None,
                           'presence/zz_fav_admin': None})

print(f'\n{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
