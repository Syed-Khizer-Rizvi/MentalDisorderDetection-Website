"""Treatment plan feature, end to end against the project's Firebase database.

Run from the project root:   venv\\Scripts\\python tests\\treatment_plan_test.py
Needs firebase_key.json (as the app does). Uses throwaway zz_tp_* users, which are deleted at the end
together with their activity-log entries.
"""
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


appmod.limiter.enabled = False   # rate limits have their own suite
T0 = int(time.time() * 1000)
st = str(int(time.time()))
D, D2, P, P2, P3 = (f'zz_tp_{x}_{st}' for x in ('doc', 'doc2', 'pat', 'pat2', 'pat3'))
ALL = (D, D2, P, P2, P3)
SECRET_OBS = 'SECRET-OBSERVATION-' + st
INSTR = 'Breathe slowly each morning and write one line in your journal at night. ' + st
PK = 300   # Pakistan: JavaScript getTimezoneOffset() is -300


def client(uid, role, status='active'):
    c = app.test_client()
    with c.session_transaction() as s:
        s['user'] = {'uid': uid, 'name': 'Dr. Zz Plan' if uid == D else ('Dr. Zz Other' if uid == D2 else 'Zz Patient'),
                     'role': role, 'status': status}
        s['_presence_at'] = 9e12
    return c


def plan_node(uid):
    return db.reference(f'users/{uid}/treatment_plan').get() or {}


try:
    db.reference().update({
        f'users/{D}': {'role': 'doctor', 'status': 'approved', 'name': 'Dr. Zz Plan', 'email': 'd@example.invalid', 'created_at': T0},
        f'users/{D2}': {'role': 'doctor', 'status': 'approved', 'name': 'Dr. Zz Other', 'email': 'd2@example.invalid', 'created_at': T0},
        f'users/{P}': {'role': 'patient', 'status': 'active', 'name': 'Zz Patient', 'email': 'p@example.invalid', 'created_at': T0,
                       'assigned_doctor_uid': D},
        f'users/{P2}': {'role': 'patient', 'status': 'active', 'name': 'Zz Other Patient', 'email': 'p2@example.invalid',
                        'created_at': T0, 'assigned_doctor_uid': D2},
        f'users/{P3}': {'role': 'patient', 'status': 'active', 'name': 'Zz No Plan', 'email': 'p3@example.invalid',
                        'created_at': T0, 'assigned_doctor_uid': D},
    })
    doc, other, pat, nop = client(D, 'doctor', 'approved'), client(D2, 'doctor', 'approved'), client(P, 'patient'), client(P3, 'patient')
    url = f'/doctor/patients/{P}/treatment-plan'

    # ---------- helpers: the patient's local day ----------
    day = 86_400_000
    noon_utc = (T0 // day) * day + 12 * 3_600_000
    check('local midnight (UTC+5)', appmod.local_day_start_ms(noon_utc, -PK) == (T0 // day) * day - 5 * 3_600_000)
    check('local midnight (UTC)', appmod.local_day_start_ms(noon_utc, 0) == (T0 // day) * day)
    check('local midnight (UTC-8)', appmod.local_day_start_ms(noon_utc - 10 * 3_600_000, 480) == (T0 // day) * day - day + 8 * 3_600_000)
    rf = appmod.reflow_paragraphs
    check('reflow: hard-wrapped lines join into one paragraph',
          rf('Take a short walk every\nmorning and note how\nyou feel.') == ['Take a short walk every morning and note how you feel.'])
    check('reflow: blank lines keep paragraphs; list items keep their lines',
          rf('First para\nwrapped.\n\nSteps:\n- breathe\n- journal\n1. sleep early') == ['First para wrapped.', 'Steps:\n- breathe\n- journal\n1. sleep early']
          and rf('a\r\nb') == ['a b'] and rf('') == [] and rf(None) == [])
    check('tz offset validation', appmod.clean_tz_offset('-300') == -300 and appmod.clean_tz_offset(9999) is None
          and appmod.clean_tz_offset('x') is None)

    # ---------- doctor: page before a plan exists ----------
    h = doc.get(url).get_data(as_text=True)
    check('doctor plan page renders (no plan yet)', 'New treatment plan' in h and 'Save plan' in h and 'Current plan' not in h
          and all(f'value="{a}"' in h for a in appmod.TREATMENT_ACTIVITIES) and all(f'value="{g}"' in h for g in appmod.GAMES))
    check('"Guided meditation" is no longer offered', 'value="meditation"' not in h and 'Guided meditation' not in h)
    check('a new plan cannot include it', doc.post(f'/doctor/patients/{P3}/treatment-plan', json={
        'instructions': 'x', 'recommended_activities': ['breathing', 'meditation']}).status_code == 400 and not plan_node(P3))
    hd = doc.get(f'/doctor/patients/{P}').get_data(as_text=True)
    check('patient detail page has the Treatment Plan section', 'id="treatment-plan"' in hd and '+ Create plan' in hd
          and f'{url}' in hd and hd.index('id="treatment-plan"') < hd.index('id="clinical-notes"'))

    # ---------- doctor: validation ----------
    good = {'observations': SECRET_OBS, 'instructions': INSTR, 'recommended_activities': ['breathing', 'journal', 'gratitude'],
            'recommended_games': ['memory', 'zen'], 'follow_up_date': time.strftime('%Y-%m-%d', time.localtime(time.time() + 14 * 86400))}
    check('form posts rejected (JSON only)', doc.post(url, data={'instructions': 'x'}).status_code == 415)
    for label, change, field in (('no instructions', {'instructions': '  '}, 'instructions'),
                                 ('too long', {'observations': 'x' * 2001}, 'observations'),
                                 ('unknown activity', {'recommended_activities': ['breathing', 'skydiving']}, 'recommended_activities'),
                                 ('unknown game', {'recommended_games': ['chess']}, 'recommended_games'),
                                 ('activities not a list', {'recommended_activities': 'breathing'}, 'recommended_activities'),
                                 ('nothing recommended', {'recommended_activities': [], 'recommended_games': []}, 'recommended_activities'),
                                 ('bad date', {'follow_up_date': '2026-02-31'}, 'follow_up_date'),
                                 ('date too far', {'follow_up_date': '2099-01-01'}, 'follow_up_date')):
        r = doc.post(url, json={**good, **change})
        check(f'save rejects {label}', r.status_code == 400 and field in (r.get_json().get('errors') or {}), r.get_json())
    check('nothing stored after rejections', not plan_node(P))

    # ---------- doctor: create ----------
    r = doc.post(url, json={**good, 'created_by': 'hacker', 'completed_today': {'breathing': 1}})
    j = r.get_json()
    p = plan_node(P)
    check('doctor creates the plan', r.status_code == 200 and j['success'] and j['created'] is True, j)
    check('stored fields', p['observations'] == SECRET_OBS and p['instructions'] == INSTR
          and p['recommended_activities'] == ['breathing', 'journal', 'gratitude'] and p['recommended_games'] == ['memory', 'zen']
          and p['follow_up_date'] == good['follow_up_date'] and p['created_by'] == D and p['created_by_name'] == 'Zz Plan'
          and p['updated_by'] == D and isinstance(p['created_at'], int) and isinstance(p['updated_at'], int)
          and 'completed_today' not in p, p)
    h = doc.get(url).get_data(as_text=True)
    check('doctor reads the plan back', 'Current plan' in h and SECRET_OBS in h and INSTR in h and 'Last updated' in h
          and 'by Dr. Zz Plan' in h and 'Update plan' in h)
    hd = doc.get(f'/doctor/patients/{P}').get_data(as_text=True)
    check('detail page shows the plan summary', '✎ Edit plan' in hd and INSTR in hd and SECRET_OBS in hd and 'Memory Match' in hd)

    # ---------- access control ----------
    r = other.get(url)
    check("another doctor can't open the plan page", r.status_code == 302 and r.headers['Location'].endswith('/doctor/patients'))
    r = other.post(url, json={**good, 'instructions': 'hijack'})
    check("another doctor can't save it (403)", r.status_code == 403 and plan_node(P)['instructions'] == INSTR)
    check("another doctor can't open the patient detail page", other.get(f'/doctor/patients/{P}').status_code == 302)
    check("doctor can't touch an unassigned patient's plan", doc.post(f'/doctor/patients/{P2}/treatment-plan', json=good).status_code == 403
          and not plan_node(P2))
    check('unknown / unsafe patient ids', doc.post('/doctor/patients/nobody_here_zz/treatment-plan', json=good).status_code == 404
          and doc.get('/doctor/patients/a.b/treatment-plan').status_code in (302, 404))
    check('patients and signed-out visitors are blocked', pat.post(url, json=good).status_code in (302, 403)
          and pat.get(url).status_code in (302, 403) and app.test_client().get(url).status_code == 302)

    # ---------- patient: dashboard card (no observations) ----------
    h = pat.get('/patient/dashboard').get_data(as_text=True)
    card = h[h.index('id="treatment-plan"'):h.index('trend-panel')]
    check('patient card: instructions, checklist, games, follow-up, last updated',
          INSTR in card and 'Daily breathing exercise' in card and 'Journal entry' in card and 'Gratitude jar note' in card
          and '/patient/games/memory' in card and '/patient/games/zen' in card and '▶ Play' in card
          and 'Next follow-up' in card and 'Last updated by Dr. Zz Plan on' in card, card[:300])
    check('patient never sees the observations', SECRET_OBS not in h)
    check('badge 0/3 completed today', re.search(r'id="tpBadgeDone">0</span>/3 completed today', h) is not None)
    check('existing dashboard cards still there', all(x in h for x in ('Risk Trend', 'My Consultation Requests', 'Quick Actions', 'Emergency Helplines')))
    r = pat.get('/patient/treatment-plan')
    check('/patient/treatment-plan in a browser goes to the dashboard card', r.status_code == 302 and r.headers['Location'].endswith('/patient/dashboard#treatment-plan'))
    jv = pat.get('/patient/treatment-plan?format=json').get_json()
    check('patient JSON has no observations', jv['success'] and 'observations' not in jv['plan'] and SECRET_OBS not in json.dumps(jv)
          and jv['plan']['instructions'] == INSTR and [a['id'] for a in jv['plan']['activities']] == ['breathing', 'journal', 'gratitude'])
    check('patient with no plan sees the empty state', 'No treatment plan yet. Your doctor will create one after your consultation.'
          in nop.get('/patient/dashboard').get_data(as_text=True))
    check('/patient/treatment-plan JSON with no plan', nop.get('/patient/treatment-plan?format=json').get_json()['plan'] is None)

    # ---------- patient: ticking activities ----------
    done = '/api/treatment-plan/complete'
    check('JSON only', pat.post(done, data={'activity': 'breathing'}).status_code == 415)
    check('activity must be in the plan', pat.post(done, json={'activity': 'zen'}).status_code == 400)
    check('done must be true/false', pat.post(done, json={'activity': 'breathing', 'done': 'yes'}).status_code == 400)
    check('no plan -> 404', nop.post(done, json={'activity': 'breathing', 'done': True}).status_code == 404)
    check('doctors cannot tick', doc.post(done, json={'activity': 'breathing'}).status_code in (302, 403))
    r = pat.post(done, json={'activity': 'breathing', 'done': True, 'tz_offset': -PK})
    jr = r.get_json()
    stamp = plan_node(P).get('completed_today', {}).get('breathing')
    check('tick an activity', r.status_code == 200 and jr['done'] is True and jr['done_count'] == 1 and jr['total'] == 3
          and isinstance(stamp, int) and abs(stamp - time.time() * 1000) < 120_000, jr)
    pat.post(done, json={'activity': 'journal', 'done': True, 'tz_offset': -PK})
    h = pat.get('/patient/dashboard').get_data(as_text=True)
    check('dashboard shows 2/3 and ticked boxes', re.search(r'id="tpBadgeDone">2</span>/3 completed today', h) is not None
          and re.search(r'value="breathing" checked', h) and re.search(r'value="journal" checked', h) and not re.search(r'value="gratitude" checked', h))
    r = pat.post(done, json={'activity': 'journal', 'done': False, 'tz_offset': -PK})
    check('untick', r.get_json()['done_count'] == 1 and 'journal' not in plan_node(P).get('completed_today', {}))
    h = doc.get(url).get_data(as_text=True)
    check("doctor sees today's progress", '1 of 3 done today' in h)

    # ---------- midnight reset (simulated) ----------
    pk_midnight = appmod.local_day_start_ms(int(time.time() * 1000), -PK)
    db.reference(f'users/{P}/treatment_plan/completed_today').update({'breathing': pk_midnight - 60_000,   # 23:59 yesterday (PKT)
                                                                       'journal': pk_midnight + 60_000})   # 00:01 today
    plan = plan_node(P)
    now = int(time.time() * 1000)
    check('tick from 23:59 yesterday no longer counts', set(appmod.done_today(plan, now, -PK)) == {'journal'})
    check('the same ticks seen from another time zone use that local midnight',
          appmod.done_today(plan, appmod.local_day_start_ms(now, -PK) + day + 3_600_000, -PK) == {})
    jv = pat.get(f'/patient/treatment-plan?format=json&tz={-PK}').get_json()['plan']
    check('patient JSON after midnight: only today\'s tick', jv['done_count'] == 1
          and [a['id'] for a in jv['activities'] if a['done']] == ['journal'])
    db.reference(f'users/{P}/treatment_plan/completed_today/gratitude').set(now - 40 * 3_600_000)   # two days ago
    pat.post(done, json={'activity': 'breathing', 'done': True, 'tz_offset': -PK})
    left = plan_node(P).get('completed_today', {})
    check('ticking clears ticks older than 36 h', 'gratitude' not in left and 'breathing' in left and 'journal' in left, left)

    # ---------- doctor: update ----------
    created_at = plan_node(P)['created_at']
    time.sleep(1.1)
    upd = {**good, 'instructions': INSTR + ' Updated.', 'recommended_activities': ['journal', 'gratitude'], 'follow_up_date': ''}
    r = doc.post(url, json=upd)
    p = plan_node(P)
    check('doctor updates the plan', r.status_code == 200 and r.get_json()['created'] is False and p['instructions'].endswith('Updated.')
          and p['recommended_activities'] == ['journal', 'gratitude'] and 'follow_up_date' not in p
          and p['created_at'] == created_at and p['updated_at'] > created_at, p)
    check('ticks for removed activities are dropped on update', set(p.get('completed_today', {})) == {'journal'})
    jv = pat.get(f'/patient/treatment-plan?format=json&tz={-PK}').get_json()['plan']
    check('patient sees the update', jv['total'] == 2 and jv['done_count'] == 1 and jv['follow_up_label'] is None)
    # a reassigned patient: the new doctor can edit, the old one no longer can
    db.reference(f'users/{P}/assigned_doctor_uid').set(D2)
    check('after reassignment the old doctor is locked out', doc.post(url, json=upd).status_code == 403)
    r = other.post(url, json={**upd, 'instructions': 'New doctor plan'})
    p = plan_node(P)
    check('the new doctor can update; creator kept', r.status_code == 200 and p['created_by'] == D and p['updated_by'] == D2
          and p['updated_by_name'] == 'Zz Other')   # stored without "Dr.", which the pages add
    h = pat.get('/patient/dashboard').get_data(as_text=True)
    check('patient sees who updated it last', 'Last updated by Dr. Zz Other on' in h)
    db.reference(f'users/{P}/assigned_doctor_uid').set(D)

    # ---------- privacy in the admin Database Browser ----------
    admin = client('zz_tp_admin', 'admin')
    h = admin.get(f'/admin/database?path=users/{P}/treatment_plan').get_data(as_text=True)
    check('database browser hides observations + instructions', SECRET_OBS not in h and 'New doctor plan' not in h
          and h.count('***note hidden***') >= 2 and 'recommended_activities' in h)

    # ---------- a plan saved before "Guided meditation" was retired ----------
    db.reference(f'users/{P}/treatment_plan/recommended_activities').set(['journal', 'gratitude', 'meditation'])
    h = doc.get(url).get_data(as_text=True)
    check('doctor form keeps it checked, marked "Feature removed"', h.count('value="meditation"') == 1
          and re.search(r'value="meditation" checked><span>Guided meditation <span class="tp-tag tp-tag-removed">Feature removed', h))
    hd = doc.get(f'/doctor/patients/{P}').get_data(as_text=True)
    check('doctor summary marks it', re.search(r'Guided meditation <span class="tp-tag tp-tag-removed">Feature removed', hd) is not None)
    hp = pat.get('/patient/dashboard').get_data(as_text=True)
    check('patient still sees it, marked, and counts it', 'Guided meditation <span class="tp-tag tp-tag-removed">Feature removed' in hp
          and re.search(r'/3 completed today', hp) is not None)
    r = pat.post(done, json={'activity': 'meditation', 'done': True, 'tz_offset': -PK})
    check('patient can still tick it (no data broken)', r.status_code == 200 and r.get_json()['total'] == 3)
    keep = {**upd, 'instructions': 'New doctor plan', 'recommended_activities': ['journal', 'gratitude', 'meditation']}
    r = doc.post(url, json=keep)
    check('doctor can save the plan and keep it', r.status_code == 200 and 'meditation' in plan_node(P)['recommended_activities'])
    r = doc.post(url, json={**keep, 'recommended_activities': ['journal', 'gratitude']})
    check('doctor can untick it to drop it', r.status_code == 200 and plan_node(P)['recommended_activities'] == ['journal', 'gratitude']
          and 'meditation' not in plan_node(P).get('completed_today', {}))
    check('once dropped it cannot be added back', doc.post(url, json=keep).status_code == 400
          and 'meditation' not in plan_node(P)['recommended_activities'])

    # ---------- instructions pasted with a line break every ~50 characters ----------
    wrapped = ('Please continue the breathing exercise every morning\nfor the next two weeks and note how your stress\n'
               'levels change. Start a short journal entry before\nbed.\n\nIf anything feels worse, message me.')
    r = doc.post(url, json={**keep, 'recommended_activities': ['journal', 'gratitude'], 'instructions': wrapped})
    check('wrapped instructions are stored exactly as typed', r.status_code == 200 and plan_node(P)['instructions'] == wrapped)
    hp = pat.get('/patient/dashboard').get_data(as_text=True)
    box = hp[hp.index('class="tp-instructions"'):hp.index('class="tp-patient-grid"')]
    check('patient card shows them as full-width paragraphs (no forced breaks)',
          '<p>Please continue the breathing exercise every morning for the next two weeks and note how your stress levels change. '
          'Start a short journal entry before bed.</p><p>If anything feels worse, message me.</p>' in box
          and not any('\n' in p for p in re.findall(r'<p>(.*?)</p>', box, re.S)), box[:400])
    hd = doc.get(f'/doctor/patients/{P}').get_data(as_text=True)
    check('doctor summary re-flows them too', 'note how your stress levels change. Start a short journal' in hd)
    check('doctor edit box keeps the original line breaks', 'every morning\nfor the next two weeks' in doc.get(url).get_data(as_text=True))

    # ---------- activity log ----------
    time.sleep(2)
    logs = db.reference('activity_logs').order_by_key().start_at(appmod.push_key_prefix(T0 - 60000)).get() or {}
    mine = [v for v in logs.values() if str(v.get('user_uid', '')).startswith('zz_tp_')]
    acts = [v['action'] for v in mine]
    check('plan created / updated logged', acts.count('treatment_plan_created') == 1 and acts.count('treatment_plan_updated') == 5, acts)
    check('patient completions logged (ticks only, not unticks)', acts.count('treatment_activity_completed') == 4, acts)
    check('logs never contain the plan text', not any(s in json.dumps(mine) for s in (SECRET_OBS, INSTR)))
finally:
    time.sleep(1)
    db.reference().update({**{f'users/{u}': None for u in ALL}, **{f'presence/{u}': None for u in ALL + ('zz_tp_admin',)}})
    logs = db.reference('activity_logs').order_by_key().start_at(appmod.push_key_prefix(T0 - 60000)).get() or {}
    rm = {f'activity_logs/{k}': None for k, v in logs.items() if str(v.get('user_uid', '')).startswith('zz_tp_')}
    if rm:
        db.reference().update(rm)
    check('cleanup', all(db.reference(f'users/{u}').get() is None for u in ALL), f'{len(rm)} log entries removed')

print(f'\n{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
