"""Show / hide password toggle. Run from the project root:  venv\\Scripts\\python tests\\password_toggle_test.py
Renders the login, register and admin account pages (the admin one as a throwaway zz_pwt_admin session) and checks
that every password field is marked for the toggle, keeps its autocomplete value and loads the script."""
import glob
import os
import re
import sys

sys.path.insert(0, os.getcwd())
import app as appmod  # noqa: E402
from app import app, db  # noqa: E402

results = []


def check(name, ok, extra=''):
    results.append(bool(ok))
    print(('PASS ' if ok else 'FAIL ') + name + (f' -> {extra}' if extra != '' else ''))


appmod.limiter.enabled = False
SCRIPT = '/static/js/password_toggle.js'
PW_INPUT = re.compile(r'<input\b[^>]*\btype="password"[^>]*>')

# ---- no template has a password field without the toggle class ----
unmarked = []
for path in glob.glob('templates/**/*.html', recursive=True):
    for tag in PW_INPUT.findall(open(path, encoding='utf-8').read()):
        if 'pw-field' not in tag:
            unmarked.append(f'{os.path.basename(path)}: {tag[:60]}')
check('every type="password" input in templates has class pw-field', not unmarked, unmarked)


def pw_inputs(html):
    return PW_INPUT.findall(html)


# ---- the rendered pages ----
c = app.test_client()
expect = {  # url -> autocomplete value of each password field, in page order
    '/patient/login': ['current-password'],
    '/doctor/login': ['current-password'],
    '/admin/login': ['current-password'],
    '/patient/register': ['new-password', 'new-password'],
    '/doctor/register': ['new-password', 'new-password'],
}
for url, autos in expect.items():
    h = c.get(url).get_data(as_text=True)
    fields = pw_inputs(h)
    check(f'{url}: {len(autos)} password field(s), all marked', len(fields) == len(autos) and all('pw-field' in f for f in fields),
          len(fields))
    check(f'{url}: autocomplete kept', [re.search(r'autocomplete="([^"]+)"', f).group(1) for f in fields] == autos)
    check(f'{url}: loads password_toggle.js once', h.count(SCRIPT) == 1)
    check(f'{url}: fields sit in a positioned .input-wrap',
          all(re.search(r'<div class="input-wrap">\s*<span class="input-icon"[^>]*>[^<]*</span>\s*' + re.escape(f), h) for f in fields))

try:
    admin = app.test_client()
    with admin.session_transaction() as s:
        s['user'] = {'uid': 'zz_pwt_admin', 'name': 'Zz', 'role': 'admin', 'status': 'active'}
        s['_presence_at'] = 9e12
    r = admin.get('/admin/users')
    h = r.get_data(as_text=True)
    fields = pw_inputs(h)
    check('/admin/users renders', r.status_code == 200, r.status_code)
    check('/admin/users: create-account password field marked + wrapped',
          bool(fields) and all('pw-field' in f for f in fields) and '<span class="pw-wrap"><input type="password"' in h, len(fields))
    check('/admin/users: autocomplete new-password kept', all('autocomplete="new-password"' in f for f in fields))
    check('/admin/users: loads password_toggle.js once', h.count(SCRIPT) == 1)
finally:
    db.reference().update({'presence/zz_pwt_admin': None})

# ---- pages without password fields don't pay for the script ----
for url in ('/', '/help', '/forgot-password'):
    h = c.get(url).get_data(as_text=True)
    check(f'{url}: no password fields, no toggle script', not pw_inputs(h) and SCRIPT not in h)

# ---- the script and its styles ----
r = c.get(SCRIPT)
js = r.get_data(as_text=True)
check('password_toggle.js served as JavaScript', r.status_code == 200 and 'javascript' in r.mimetype)
check('targets input.pw-field', "querySelectorAll('input.pw-field')" in js)
check('button is type="button" (never submits)', "button.type = 'button'" in js)
check('toggles type password <-> text', "input.type = show ? 'text' : 'password'" in js)
check('aria-label Show password / Hide password', "'Hide password' : 'Show password'" in js)
check('aria-pressed reflects the state', "aria-pressed" in js)
check('eye and eye-off icons', "eye:" in js and "'eye-off':" in js)
check('state change announced via a polite live region', "'aria-live', 'polite'" in js and 'Password shown' in js and 'Password hidden' in js)
check('field reset to hidden before submit', "addEventListener('submit'" in js)
check('script never touches autocomplete', 'autocomplete' not in js.split('*/', 1)[1])
css = open('static/css/ui.css', encoding='utf-8').read()
block = css[css.find('Show / hide password toggle'):]
check('toggle button is 36x36', 'width: 36px; height: 36px' in block)
check('vertically centred at the right', 'top: 50%' in block and 'translateY(-50%)' in block and 'right: 4px' in block)
check('muted by default, brand green on hover', 'color: var(--ui-faint)' in block and ':hover' in block and 'color: var(--ui-green)' in block)
check('no background', 'background: none' in block)
check('input gets right padding so text never runs under the icon', re.search(r'input\.pw-has-toggle \{ padding-right: 44px', block))
check('keyboard focus ring', '.pw-toggle:focus-visible' in block)

print(f'\n{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
