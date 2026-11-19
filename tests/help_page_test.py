"""Help / FAQ page. Run from the project root:  venv\\Scripts\\python tests\\help_page_test.py"""
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


appmod.limiter.enabled = False
QUESTIONS = ['How do I take an assessment?', 'What does my risk score mean?', 'How does the wristband connect?',
             'How can I contact my doctor?', 'How do I change my password?', 'Is my data private?',
             'Can I delete my account?', 'How often should I check in?']

anon = app.test_client()
r = anon.get('/help')
h = r.get_data(as_text=True)
check('/help is public (200 without signing in)', r.status_code == 200)
check('all 8 requested questions', all(f'<summary>{q}</summary>' in h for q in QUESTIONS),
      [q for q in QUESTIONS if f'<summary>{q}</summary>' not in h])
summaries = re.findall(r'<summary>', h)
check('8-10 collapsible sections', 8 <= len(summaries) <= 10 and h.count('<details class="faq"') == len(summaries), len(summaries))
check('collapsed by default (no "open")', '<details class="faq" open' not in h and ' open>' not in h)
check('no external JS', '<script src="http' not in h and 'cdn' not in h.lower())
check('risk bands come from the live thresholds', all(f"<strong>{r['label']}</strong> {r['range']}" in h for r in appmod.risk_ranges()))
check('wristband answer uses the live interval + product', f"about every {appmod.setting('wristband', 'interval_s')} seconds" in h
      and 'PsyMonitor Band v2' in h and 'Generate Pairing Code' in h)
check('links to the reset page and support email', 'href="/forgot-password"' in h and 'mailto:psymonitor.support@gmail.com' in h)
check('helplines in the contact answer', 'Umang 0317-4288665' in h and 'Rozan 051-2890505' in h)
check('title + description + favicon', '<title>Help & FAQ | Mental Disorder Detection</title>' in h
      and 'name="description" content="Answers to common questions' in h and 'favicon-32.png' in h)
home = anon.get('/').get_data(as_text=True)
check('footer links to Help', 'href="/help">Help &amp; FAQ</a>' in home)
nav = home[home.index('<div class="nav-links">'):home.index('<div class="nav-buttons">')]
check('home page top nav: Help after Contact, same style, before the login buttons',
      re.search(r'href="/#contact" class="nav-link">Contact</a>\s*<a href="/help" class="nav-link">Help</a>\s*</div>', nav) is not None
      and home.index('class="nav-link">Help</a>') < home.index('Patient Login</a>'), nav)
check('Help link in the nav on other public pages too', 'href="/help" class="nav-link">Help</a>' in anon.get('/help').get_data(as_text=True))

st = str(int(time.time()))
D, P = f'zz_help_doc_{st}', f'zz_help_pat_{st}'
try:
    db.reference().update({
        f'users/{D}': {'role': 'doctor', 'status': 'approved', 'name': 'Dr. Zz Help', 'email': 'h@example.invalid', 'created_at': 1},
        f'users/{P}': {'role': 'patient', 'status': 'active', 'name': 'Zz Help', 'email': 'i@example.invalid', 'created_at': 1, 'assigned_doctor_uid': D},
    })

    def as_user(uid, role, status='active'):
        cl = app.test_client()
        with cl.session_transaction() as s:
            s['user'] = {'uid': uid, 'name': 'Zz', 'role': role, 'status': status}
            s['_presence_at'] = 9e12
        return cl
    pat, doc = as_user(P, 'patient'), as_user(D, 'doctor', 'approved')
    link = 'href="/help" class="nav-item"><span class="nav-icon">❓</span><span class="nav-label">Help &amp; FAQ</span>'
    for who, cl, urls in (('patient', pat, ['/patient/dashboard', '/patient/history', '/patient/profile', '/patient/activities']),
                          ('doctor', doc, ['/doctor/dashboard', '/doctor/patients', '/doctor/monitor', f'/doctor/patients/{P}'])):
        missing = [u for u in urls if link not in cl.get(u).get_data(as_text=True)]
        check(f'{who} sidebar links to Help', not missing, missing)
    hd = doc.get('/doctor/dashboard').get_data(as_text=True)
    nav = re.findall(r'nav-label">([^<]+)<', hd)
    check('Help sits just above Logout', nav[-2:] == ['Help &amp; FAQ', 'Logout'], nav[-3:])
    h = pat.get('/help').get_data(as_text=True)
    check('signed-in visitors see "My dashboard" instead of the login buttons', 'My dashboard' in h and 'Patient Login</a>' not in h.split('</nav>')[0])
finally:
    db.reference().update({f'users/{D}': None, f'users/{P}': None, f'presence/{D}': None, f'presence/{P}': None})

print(f'\n{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
