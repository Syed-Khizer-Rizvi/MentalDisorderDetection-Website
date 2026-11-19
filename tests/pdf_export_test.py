"""Patient assessment PDF. Run from the project root:  venv\\Scripts\\python tests\\pdf_export_test.py
Uses throwaway zz_pdf_* users (removed at the end). Page compression is switched off so the PDF text can be
checked without a PDF library."""
import os
import re
import sys
import time

sys.path.insert(0, os.getcwd())
import reportlab.rl_config  # noqa: E402
import app as appmod  # noqa: E402
from app import app, db  # noqa: E402

reportlab.rl_config.pageCompression = 0
results = []


def check(name, ok, extra=''):
    results.append(bool(ok))
    print(('PASS ' if ok else 'FAIL ') + name + (f' -> {extra}' if extra != '' else ''))


def pdf_has(data, text):
    """Text drawn with the built-in fonts appears in the uncompressed content streams as (...) strings."""
    escaped = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
    return escaped.encode('latin-1') in data


appmod.limiter.enabled = False
st = str(int(time.time()))
D, P, P2 = f'zz_pdf_doc_{st}', f'zz_pdf_pat_{st}', f'zz_pdf_pat2_{st}'
AID, AID2 = '1700000000000', '1700000500000'
SECRET_OBS = 'SECRET-OBS-' + st
try:
    db.reference().update({
        f'users/{D}': {'role': 'doctor', 'status': 'approved', 'name': 'Dr. Zz Pdf', 'email': 'd@example.invalid', 'created_at': 1,
                       'profile': {'specialization': 'Psychiatry', 'affiliation': 'Zz Clinic'}},
        f'users/{P}': {'role': 'patient', 'status': 'active', 'name': 'Ayesha Khan', 'email': 'p@example.invalid', 'created_at': 1,
                       'assigned_doctor_uid': D, 'latest_assessment_id': AID,
                       'assessments': {
                           AID: {'phq_score': 16, 'gad_score': 13, 'pss_score': 27, 'created_at': int(AID),
                                 'facial_emotions': {'sad': 45.2, 'neutral': 35.1, 'happy': 19.7}, 'facial_peak': {'emotion': 'sad', 'value': 71}},
                           AID2: {'phq_score': 3, 'gad_score': 2, 'pss_score': 9, 'created_at': int(AID2), 'facial_skipped': True}},
                       'treatment_plan': {'observations': SECRET_OBS, 'instructions': 'Breathe slowly every\nmorning.\n\nWrite in your journal.',
                                          'recommended_activities': ['breathing', 'journal'], 'recommended_games': ['memory'],
                                          'follow_up_date': '2026-12-01', 'created_at': 1, 'updated_at': 2, 'created_by': D,
                                          'created_by_name': 'Zz Pdf', 'updated_by': D, 'updated_by_name': 'Zz Pdf'}},
        f'users/{P2}': {'role': 'patient', 'status': 'active', 'name': 'Zz Other', 'email': 'q@example.invalid', 'created_at': 1,
                        'assessments': {AID: {'phq_score': 1, 'gad_score': 1, 'pss_score': 1, 'created_at': int(AID)}}},
    })

    def as_user(uid, role, status='active', **extra):
        cl = app.test_client()
        with cl.session_transaction() as s:
            s['user'] = {'uid': uid, 'name': 'Ayesha Khan' if uid == P else 'Zz', 'role': role, 'status': status}
            s['_presence_at'] = 9e12
            s.update(extra)
        return cl
    pat = as_user(P, 'patient', assessment={'phq_score': 16, 'gad_score': 13, 'pss_score': 27, 'id': AID,
                                            'facial_emotions': {'sad': 45.2}, 'facial_peak': {'emotion': 'sad', 'value': 71}})

    # ---- buttons ----
    h = pat.get('/patient/results').get_data(as_text=True)
    check('results page has a Download PDF button', f'href="/patient/assessments/{AID}/pdf"' in h and 'Download PDF' in h)
    h = pat.get(f'/patient/history/{AID2}').get_data(as_text=True)
    check('past results (History) have it too', f'href="/patient/assessments/{AID2}/pdf"' in h)
    old = as_user(P, 'patient', assessment={'phq_score': 1, 'gad_score': 1, 'pss_score': 1})   # session from before ids were kept
    check('older sessions fall back to the latest assessment', f'href="/patient/assessments/{AID}/pdf"' in old.get('/patient/results').get_data(as_text=True))
    doc = as_user(D, 'doctor', 'approved')
    hd = doc.get(f'/doctor/patients/{P}/assessments/{AID}').get_data(as_text=True)
    check("doctor's view of the results has no patient PDF button", 'Download PDF' not in hd)

    # ---- the PDF ----
    r = pat.get(f'/patient/assessments/{AID}/pdf')
    data = r.data
    check('PDF download', r.status_code == 200 and r.mimetype == 'application/pdf' and data.startswith(b'%PDF'), r.status_code)
    check('filename assessment_<name>_<date>.pdf', r.headers['Content-Disposition'] ==
          f'attachment; filename="assessment_Ayesha_Khan_{appmod.format_timestamp(int(AID), "%Y-%m-%d")}.pdf"', r.headers['Content-Disposition'])
    check('not cached', 'no-store' in r.headers.get('Cache-Control', ''))
    check('letterhead', pdf_has(data, 'Mental Disorder Detection') and pdf_has(data, 'psymonitor.support@gmail.com')
          and pdf_has(data, 'Assessment Report') and b'/Subtype /Image' in data)
    check('patient name + assessment date', pdf_has(data, 'Ayesha Khan') and pdf_has(data, appmod.format_timestamp(int(AID), '%d %b %Y')))
    check('PHQ-9 / GAD-7 / PSS scores with severity', pdf_has(data, '16 / 27') and pdf_has(data, '13 / 21') and pdf_has(data, '27 / 40')
          and pdf_has(data, 'Mod. Severe') and pdf_has(data, 'Moderate') and pdf_has(data, 'High'))
    check('facial emotion summary', pdf_has(data, 'Sad') and pdf_has(data, '45%') and pdf_has(data, 'Video is never stored'))
    composite = appmod.compute_composite({'phq_score': 16, 'gad_score': 13, 'pss_score': 27,
                                          'facial_emotions': {'sad': 45.2, 'neutral': 35.1, 'happy': 19.7}})
    band = appmod.risk_band(composite)[1]
    check('composite risk score + band', pdf_has(data, str(composite)) and pdf_has(data, f'{band} risk'), (composite, band))
    check("assigned doctor's name", pdf_has(data, 'Dr. Zz Pdf') and pdf_has(data, 'Psychiatry'))
    check('treatment plan instructions (re-flowed)', pdf_has(data, 'Breathe slowly every morning.') and pdf_has(data, 'Write in your journal.')
          and pdf_has(data, 'Next follow-up: Tue 01 Dec 2026'))
    check("doctor's private observations never in the PDF", SECRET_OBS.encode() not in data)
    check('disclaimer + helplines', pdf_has(data, 'not a diagnosis') and pdf_has(data, 'Umang 0317-4288665'))
    r = pat.get(f'/patient/assessments/{AID2}/pdf')
    check('skipped facial scan is explained', r.status_code == 200 and pdf_has(r.data, 'The facial scan was skipped'))

    # ---- access ----
    other = as_user(P2, 'patient')
    r = other.get(f'/patient/assessments/{AID2}/pdf')
    check("a patient can't download someone else's assessment", r.status_code == 302)
    check('no plan / no doctor still works', other.get(f'/patient/assessments/{AID}/pdf').status_code == 200
          and pdf_has(other.get(f'/patient/assessments/{AID}/pdf').data, 'No treatment plan yet'))
    check('bad ids', pat.get('/patient/assessments/abc/pdf').status_code == 302 and pat.get('/patient/assessments/123/pdf').status_code == 302)
    check('doctors and visitors are blocked', doc.get(f'/patient/assessments/{AID}/pdf').status_code in (302, 403)
          and app.test_client().get(f'/patient/assessments/{AID}/pdf').status_code == 302)

    # ---- results page message for each risk band ----
    bands = {'low': (0, 0, 0), 'moderate': (9, 7, 18), 'high': (18, 14, 28), 'critical': (25, 20, 38)}
    seen = {}
    for level, (phq, gad, pss) in bands.items():
        a = {'phq_score': phq, 'gad_score': gad, 'pss_score': pss, 'facial_skipped': True, 'id': AID}
        check(f'{level} sample really is {level}', appmod.risk_band(appmod.compute_composite(a))[0] == level, appmod.compute_composite(a))
        seen[level] = as_user(P, 'patient', assessment=a).get('/patient/results').get_data(as_text=True)
    check('Low: "You\'re Doing Well!"', "You're Doing Well!" in seen['low'] and 'Pay Attention' not in seen['low'])
    check('Moderate: "Pay Attention" with the new text, amber styling, not "Doing Well"',
          '<h3>Pay Attention</h3>' in seen['moderate'] and 'rec-moderate' in seen['moderate'] and '🌤' in seen['moderate']
          and 'Your results suggest some mild stress or low mood. Try the recommended activities and check in with your doctor if symptoms persist.' in seen['moderate']
          and "You're Doing Well!" not in seen['moderate'] and 'Consult Your Doctor Soon' not in seen['moderate']
          and 'Check in with Dr. Zz Pdf' in seen['moderate'])
    check('High: existing message kept', 'Consult Your Doctor Soon' in seen['high'] and 'Pay Attention' not in seen['high']
          and "You're Doing Well!" not in seen['high'])
    check('Critical: existing message kept', 'Immediate Attention Required' in seen['critical'] and 'Pay Attention' not in seen['critical'])

    # ---- unit: filename ----
    f = appmod.assessment_pdf_filename
    check('filename slug', f('Ayesha  Khan!', int(AID)).startswith('assessment_Ayesha_Khan_') and f('', None).startswith('assessment_patient_')
          and re.fullmatch(r'assessment_[A-Za-z0-9_]+_\d{4}-\d{2}-\d{2}\.pdf', f('Zz / ../x', int(AID))) is not None)
finally:
    db.reference().update({**{f'users/{u}': None for u in (D, P, P2)}, **{f'presence/{u}': None for u in (D, P, P2)}})

print(f'\n{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
