from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import firebase_admin
from firebase_admin import credentials, auth, db
import os
from functools import wraps
from deepface import DeepFace
import base64
import numpy as np
import cv2

# ========== INITIALIZE FLASK ==========
app = Flask(__name__)
app.secret_key = 'mental-disorder-detection-secret-key-change-in-production'

# ========== INITIALIZE FIREBASE ADMIN ==========
cred = credentials.Certificate('firebase_key.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://mentaldisorderdetection-31d60-default-rtdb.firebaseio.com/'
})

# ========== FIREBASE WEB CONFIG (for frontend) ==========
FIREBASE_WEB_CONFIG = {
    'apiKey': 'AIzaSyCOMIkDlZ3tyJJxGuooTRp5ra7bI0JWEbg',
    'authDomain': 'mentaldisorderdetection-31d60.firebaseapp.com',
    'projectId': 'mentaldisorderdetection-31d60',
    'storageBucket': 'mentaldisorderdetection-31d60.firebasestorage.app',
    'messagingSenderId': '1099104870756',
    'appId': '1:1099104870756:web:f1be22b10bbd0b1900da71',
    'databaseURL': 'https://mentaldisorderdetection-31d60-default-rtdb.firebaseio.com/'
}


# ========== DECORATORS ==========
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'user' not in session:
                return redirect(url_for('login'))
            if session['user'].get('role') not in roles:
                flash('Access denied: You do not have permission to view this page.', 'error')
                return redirect(url_for('home'))
            return f(*args, **kwargs)
        return decorated
    return decorator


# ========== SCORING HELPERS ==========
def compute_composite(assessment):
    phq = assessment.get('phq_score', 0)
    gad = assessment.get('gad_score', 0)
    pss = assessment.get('pss_score', 0)
    emotions = assessment.get('facial_emotions', {}) or {}
    neg = sum(emotions.get(k, 0) for k in ['sad', 'angry', 'fear', 'disgust'])
    return round(((phq/27 + gad/21 + pss/40) * 100 + neg) / 4)


def risk_band(composite):
    if composite <= 25:
        return 'low', 'Low', '#059669'
    elif composite <= 50:
        return 'moderate', 'Moderate', '#D97706'
    elif composite <= 70:
        return 'high', 'High', '#EA580C'
    return 'critical', 'Critical', '#DC2626'


def phq_severity(s):
    if s <= 4: return ('Minimal', '#059669')
    if s <= 9: return ('Mild', '#2563EB')
    if s <= 14: return ('Moderate', '#D97706')
    if s <= 19: return ('Mod. Severe', '#EA580C')
    return ('Severe', '#DC2626')


def gad_severity(s):
    if s <= 4: return ('Minimal', '#059669')
    if s <= 9: return ('Mild', '#2563EB')
    if s <= 14: return ('Moderate', '#D97706')
    return ('Severe', '#DC2626')


def pss_severity(s):
    if s <= 13: return ('Low', '#059669')
    if s <= 26: return ('Moderate', '#D97706')
    return ('High', '#DC2626')


def format_timestamp(ms, fmt):
    from datetime import datetime
    return datetime.fromtimestamp(ms / 1000).strftime(fmt) if isinstance(ms, (int, float)) else '-'


def time_ago(ms):
    import time
    if not isinstance(ms, (int, float)):
        return '-'
    seconds = max(0, int(time.time() - ms / 1000))
    if seconds < 60:
        return 'Just now'
    if seconds >= 30 * 86400:
        return format_timestamp(ms, '%d %b %Y')
    for unit, size in [('day', 86400), ('hour', 3600), ('minute', 60)]:
        if seconds >= size:
            n = seconds // size
            return f"{n} {unit}{'s' if n != 1 else ''} ago"


def build_results_context(assessment):
    """Template variables for results.html, from a session or stored assessment."""
    phq = assessment.get('phq_score', 0)
    gad = assessment.get('gad_score', 0)
    pss = assessment.get('pss_score', 0)
    emotions = assessment.get('facial_emotions', {}) or {}

    dominant = max(emotions, key=emotions.get) if emotions else 'neutral'
    dominant_value = round(emotions.get(dominant, 0)) if emotions else 0

    peak = assessment.get('facial_peak', {}) or {}
    peak_emotion = peak.get('emotion') or dominant
    peak_value = round(peak.get('value', 0))

    phq_s, phq_c = phq_severity(phq)
    gad_s, gad_c = gad_severity(gad)
    pss_s, pss_c = pss_severity(pss)

    composite = compute_composite(assessment)
    risk_level, _, composite_color = risk_band(composite)

    return dict(
        phq_score=phq, gad_score=gad, pss_score=pss,
        phq_severity=phq_s, gad_severity=gad_s, pss_severity=pss_s,
        phq_color=phq_c, gad_color=gad_c, pss_color=pss_c,
        facial_dominant=dominant,
        facial_dominant_pct=dominant_value,
        facial_peak=peak_emotion,
        facial_peak_pct=peak_value,
        composite_score=composite,
        composite_color=composite_color,
        risk_level=risk_level
    )


# ========== PUBLIC ROUTES ==========
@app.route('/')
def home():
    return render_template('home.html', firebase_config=FIREBASE_WEB_CONFIG)


@app.route('/login')
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html', firebase_config=FIREBASE_WEB_CONFIG)


@app.route('/register')
def register():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('register.html', firebase_config=FIREBASE_WEB_CONFIG)


# ========== API ROUTES (called by frontend JS) ==========
@app.route('/api/verify-token', methods=['POST'])
def verify_token():
    try:
        data = request.get_json()
        id_token = data.get('idToken')

        decoded = auth.verify_id_token(id_token, clock_skew_seconds=10)
        uid = decoded['uid']
        email = decoded.get('email', '')

        user_ref = db.reference(f'users/{uid}')
        profile = user_ref.get()

        if not profile:
            return jsonify({'success': False, 'error': 'User profile not found. Please contact support.'}), 404

        if profile.get('role') == 'doctor' and profile.get('status') != 'approved':
            return jsonify({'success': False, 'error': 'Your doctor account is pending approval by admin.'}), 403

        session['user'] = {
            'uid': uid,
            'email': email,
            'name': profile.get('name', ''),
            'role': profile.get('role', 'patient'),
            'status': profile.get('status', 'active')
        }

        return jsonify({'success': True, 'role': profile.get('role'), 'redirect': url_for('dashboard')})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/register-profile', methods=['POST'])
def register_profile():
    try:
        data = request.get_json()
        id_token = data.get('idToken')
        name = data.get('name', '').strip()
        phone = data.get('phone', '').strip()
        role = data.get('role', 'patient').strip().lower()

        decoded = auth.verify_id_token(id_token, clock_skew_seconds=10)
        uid = decoded['uid']
        email = decoded.get('email', '')

        if role == 'admin':
            try:
                auth.delete_user(uid)
            except:
                pass
            return jsonify({'success': False, 'error': 'Admin accounts cannot be created through public registration.'}), 403

        status = 'pending' if role == 'doctor' else 'active'

        user_ref = db.reference(f'users/{uid}')
        user_ref.set({
            'uid': uid,
            'name': name,
            'email': email,
            'phone': phone,
            'role': role,
            'status': status,
            'created_at': {'.sv': 'timestamp'}
        })

        message = (
            'Account created! Your doctor account is pending admin approval. You will be notified once approved.'
            if role == 'doctor'
            else 'Account created successfully! You can now log in.'
        )

        return jsonify({'success': True, 'message': message, 'status': status})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ========== PROTECTED ROUTES ==========
@app.route('/dashboard')
@login_required
def dashboard():
    role = session['user'].get('role')
    if role == 'patient':
        return redirect(url_for('patient_dashboard'))
    elif role == 'doctor':
        return redirect(url_for('doctor_dashboard'))
    elif role == 'admin':
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('home'))


@app.route('/patient/dashboard')
@login_required
@role_required('patient')
def patient_dashboard():
    from datetime import datetime
    user_name = session['user'].get('name', 'Patient')
    user_initials = ''.join([word[0].upper() for word in user_name.split()[:2]]) or 'P'
    uid = session['user']['uid']

    latest_id = db.reference(f'users/{uid}/latest_assessment_id').get()
    latest = db.reference(f'users/{uid}/assessments/{latest_id}').get() if latest_id else None

    data = {
        'has_assessment': False, 'phq_score': 0, 'gad_score': 0, 'pss_score': 0,
        'phq_severity': '-', 'gad_severity': '-', 'pss_severity': '-',
        'facial_dominant': '-', 'composite_score': 0, 'risk_level': 'none',
        'risk_label': 'No Data', 'risk_color': '#64748B', 'assessment_date': 'No assessments yet'
    }

    if latest:
        phq = latest.get('phq_score', 0)
        gad = latest.get('gad_score', 0)
        pss = latest.get('pss_score', 0)
        emotions = latest.get('facial_emotions', {})
        dominant = max(emotions, key=emotions.get) if emotions else 'neutral'

        phq_sev = 'Minimal' if phq <= 4 else 'Mild' if phq <= 9 else 'Moderate' if phq <= 14 else 'Mod. Severe' if phq <= 19 else 'Severe'
        gad_sev = 'Minimal' if gad <= 4 else 'Mild' if gad <= 9 else 'Moderate' if gad <= 14 else 'Severe'
        pss_sev = 'Low' if pss <= 13 else 'Moderate' if pss <= 26 else 'High'

        composite = compute_composite(latest)
        risk_level, risk_label, risk_color = risk_band(composite)

        created = latest.get('created_at', 0)
        date_str = datetime.fromtimestamp(created / 1000).strftime('%d %b %Y') if isinstance(created, int) else '-'

        data.update({
            'has_assessment': True, 'phq_score': phq, 'gad_score': gad, 'pss_score': pss,
            'phq_severity': phq_sev, 'gad_severity': gad_sev, 'pss_severity': pss_sev,
            'facial_dominant': dominant.capitalize(), 'composite_score': composite,
            'risk_level': risk_level, 'risk_label': risk_label, 'risk_color': risk_color,
            'assessment_date': date_str
        })

    # Assessment keys are millisecond timestamps, so key order is chronological
    recent = db.reference(f'users/{uid}/assessments').order_by_key().limit_to_last(7).get() or {}
    recent = [a for a in recent.values() if isinstance(a.get('created_at'), (int, float))]
    recent.sort(key=lambda a: a['created_at'])

    risk_trend = []
    for a in recent:
        score = compute_composite(a)
        created = datetime.fromtimestamp(a['created_at'] / 1000)
        risk_trend.append({
            'score': score,
            'risk_label': risk_band(score)[1],
            'date': created.strftime('%d %b'),
            'time': created.strftime('%I:%M %p'),
            'full_date': created.strftime('%d %b %Y, %I:%M %p')
        })

    # Only show the time under a date label when that day has more than one assessment
    day_counts = {}
    for p in risk_trend:
        day_counts[p['date']] = day_counts.get(p['date'], 0) + 1
    for p in risk_trend:
        if day_counts[p['date']] == 1:
            p['time'] = ''

    wristband_order = db.reference(f'wristband_orders/{uid}').get()

    return render_template('patient_dashboard.html',
                           user_name=user_name, user_initials=user_initials,
                           wristband_order=wristband_order, risk_trend=risk_trend, **data)


@app.route('/patient/assessment')
@login_required
@role_required('patient')
def patient_assessment():
    return render_template('assessment.html')


@app.route('/api/submit-assessment', methods=['POST'])
@login_required
@role_required('patient')
def submit_assessment():
    try:
        data = request.get_json()
        phq_answers = data.get('phq', [])
        gad_answers = data.get('gad', [])
        pss_answers = data.get('pss', [])

        phq_score = sum(phq_answers)
        gad_score = sum(gad_answers)

        pss_reverse = [3, 4, 6, 7]
        pss_score = 0
        for i, val in enumerate(pss_answers):
            pss_score += (4 - val) if i in pss_reverse else val

        session['assessment'] = {
            'phq_answers': phq_answers,
            'gad_answers': gad_answers,
            'pss_answers': pss_answers,
            'phq_score': phq_score,
            'gad_score': gad_score,
            'pss_score': pss_score
        }

        return jsonify({'success': True, 'redirect': url_for('patient_facial_scan')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/patient/facial-scan')
@login_required
@role_required('patient')
def patient_facial_scan():
    if 'assessment' not in session:
        return redirect(url_for('patient_assessment'))
    return render_template('facial_scan.html')


@app.route('/api/analyze-emotion', methods=['POST'])
@login_required
@role_required('patient')
def analyze_emotion():
    try:
        data = request.get_json()
        image_data = data.get('image', '')

        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image data'}), 400

        result = DeepFace.analyze(
            img_path=img,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result.get('emotion', {})
        dominant = result.get('dominant_emotion', 'neutral')

        return jsonify({
            'success': True,
            'emotions': {k: float(v) for k, v in emotions.items()},
            'dominant': dominant
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/submit-facial', methods=['POST'])
@login_required
@role_required('patient')
def submit_facial():
    try:
        import time
        from datetime import datetime
        data = request.get_json()
        emotions = data.get('emotions', {})
        readings = data.get('readings', 0)
        peak = data.get('peak', {})
        uid = session['user']['uid']

        if 'assessment' in session:
            session['assessment']['facial_emotions'] = emotions
            session['assessment']['facial_readings'] = readings
            session['assessment']['facial_peak'] = peak
            session.modified = True

            assessment_id = str(int(time.time() * 1000))
            full_assessment = {
                'phq_score': session['assessment'].get('phq_score', 0),
                'gad_score': session['assessment'].get('gad_score', 0),
                'pss_score': session['assessment'].get('pss_score', 0),
                'phq_answers': session['assessment'].get('phq_answers', []),
                'gad_answers': session['assessment'].get('gad_answers', []),
                'pss_answers': session['assessment'].get('pss_answers', []),
                'facial_emotions': emotions,
                'facial_peak': peak,
                'facial_readings': readings,
                'created_at': {'.sv': 'timestamp'}
            }
            db.reference(f'users/{uid}/assessments/{assessment_id}').set(full_assessment)
            db.reference(f'users/{uid}/latest_assessment_id').set(assessment_id)

        return jsonify({'success': True, 'redirect': url_for('patient_results')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/patient/results')
@login_required
@role_required('patient')
def patient_results():
    assessment = session.get('assessment', {})
    if not assessment:
        return redirect(url_for('patient_assessment'))
    return render_template('results.html', **build_results_context(assessment))


@app.route('/patient/history')
@login_required
@role_required('patient')
def patient_history():
    user_name = session['user'].get('name', 'Patient')
    user_initials = ''.join([word[0].upper() for word in user_name.split()[:2]]) or 'P'
    uid = session['user']['uid']

    all_assessments = db.reference(f'users/{uid}/assessments').get() or {}
    items = sorted(all_assessments.items(),
                   key=lambda kv: kv[1].get('created_at', 0) if isinstance(kv[1].get('created_at'), (int, float)) else 0,
                   reverse=True)

    history = []
    for assessment_id, a in items:
        phq = a.get('phq_score', 0)
        gad = a.get('gad_score', 0)
        pss = a.get('pss_score', 0)
        emotions = a.get('facial_emotions', {}) or {}
        composite = compute_composite(a)
        risk_level, risk_label, risk_color = risk_band(composite)
        history.append({
            'id': assessment_id,
            'date': format_timestamp(a.get('created_at'), '%d %b %Y'),
            'time': format_timestamp(a.get('created_at'), '%I:%M %p'),
            'composite_score': composite,
            'risk_level': risk_level, 'risk_label': risk_label, 'risk_color': risk_color,
            'phq_score': phq, 'gad_score': gad, 'pss_score': pss,
            'phq': phq_severity(phq), 'gad': gad_severity(gad), 'pss': pss_severity(pss),
            'facial_dominant': max(emotions, key=emotions.get).capitalize() if emotions else '-'
        })

    return render_template('history.html', user_name=user_name, user_initials=user_initials, history=history)


@app.route('/patient/history/<assessment_id>')
@login_required
@role_required('patient')
def patient_history_detail(assessment_id):
    uid = session['user']['uid']
    # IDs are millisecond timestamps; reject anything else before it reaches a Firebase path
    assessment = db.reference(f'users/{uid}/assessments/{assessment_id}').get() if assessment_id.isdigit() else None
    if not assessment:
        return redirect(url_for('patient_history'))

    return render_template('results.html', read_only=True,
                           assessment_date=format_timestamp(assessment.get('created_at'), '%d %b %Y, %I:%M %p'),
                           **build_results_context(assessment))


@app.route('/patient/order-wristband')
@login_required
@role_required('patient')
def patient_order_wristband():
    if db.reference(f"wristband_orders/{session['user']['uid']}").get():
        return redirect(url_for('patient_wristband'))
    user_name = session['user'].get('name', '')
    return render_template('order_wristband.html', user_name=user_name)


WRISTBAND_STATUSES = ['pending', 'confirmed', 'shipped', 'delivered']


@app.route('/patient/wristband')
@login_required
@role_required('patient')
def patient_wristband():
    user_name = session['user'].get('name', 'Patient')
    user_initials = ''.join([word[0].upper() for word in user_name.split()[:2]]) or 'P'
    uid = session['user']['uid']

    order = db.reference(f'wristband_orders/{uid}').get()
    if order:
        status = (order.get('status') or 'pending').lower()
        order = dict(order,
                     status=status,
                     status_index=WRISTBAND_STATUSES.index(status) if status in WRISTBAND_STATUSES else 0,
                     order_date=format_timestamp(order.get('created_at'), '%d %b %Y, %I:%M %p'))

    return render_template('wristband.html', user_name=user_name, user_initials=user_initials,
                           order=order, statuses=WRISTBAND_STATUSES)


GENDER_OPTIONS = ['Male', 'Female', 'Other', 'Prefer not to say']
PROFILE_FIELDS = {  # field -> max length; stored under users/{uid}/profile
    'dob': 10, 'gender': 20, 'address': 300,
    'emergency_name': 100, 'emergency_phone': 30
}


@app.route('/patient/profile')
@login_required
@role_required('patient')
def patient_profile():
    uid = session['user']['uid']
    user = db.reference(f'users/{uid}').get() or {}
    user_name = user.get('name') or session['user'].get('name', 'Patient')
    user_initials = ''.join([word[0].upper() for word in user_name.split()[:2]]) or 'P'

    return render_template('profile.html',
                           user_name=user_name, user_initials=user_initials,
                           email=session['user'].get('email') or user.get('email', ''),
                           phone=user.get('phone', ''),
                           profile=user.get('profile') or {},
                           member_since=format_timestamp(user.get('created_at'), '%d %b %Y'),
                           gender_options=GENDER_OPTIONS)


@app.route('/api/update-profile', methods=['POST'])
@login_required
@role_required('patient')
def update_profile():
    from datetime import date
    try:
        data = request.get_json() or {}
        uid = session['user']['uid']

        name = str(data.get('name', '')).strip()
        phone = str(data.get('phone', '')).strip()
        if not name:
            return jsonify({'success': False, 'error': 'Full name is required.'}), 400
        if len(name) > 100 or len(phone) > 30:
            return jsonify({'success': False, 'error': 'Name or phone is too long.'}), 400

        profile = {}
        for field, max_len in PROFILE_FIELDS.items():
            value = str(data.get(field, '')).strip()
            if len(value) > max_len:
                return jsonify({'success': False, 'error': f'{field.replace("_", " ").capitalize()} is too long.'}), 400
            if value:
                profile[field] = value

        if 'dob' in profile:
            try:
                if date.fromisoformat(profile['dob']) > date.today():
                    return jsonify({'success': False, 'error': 'Date of birth cannot be in the future.'}), 400
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid date of birth.'}), 400
        if 'gender' in profile and profile['gender'] not in GENDER_OPTIONS:
            return jsonify({'success': False, 'error': 'Invalid gender option.'}), 400

        # name/phone stay at the top level where registration and login read them
        db.reference(f'users/{uid}').update({'name': name, 'phone': phone, 'profile': profile or None})

        session['user']['name'] = name
        session.modified = True
        initials = ''.join([word[0].upper() for word in name.split()[:2]]) or 'P'

        return jsonify({'success': True, 'message': 'Profile updated successfully.', 'name': name, 'initials': initials})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


JOURNAL_MOODS = ['Great', 'Good', 'Okay', 'Low', 'Struggling']
JOURNAL_MAX_LENGTH = 2000


@app.route('/patient/activities')
@login_required
@role_required('patient')
def patient_activities():
    user_name = session['user'].get('name', 'Patient')
    user_initials = ''.join([word[0].upper() for word in user_name.split()[:2]]) or 'P'
    uid = session['user']['uid']

    # Journal keys are millisecond timestamps, so key order is chronological
    recent = db.reference(f'users/{uid}/journal').order_by_key().limit_to_last(5).get() or {}
    journal = [{
        'mood': entry.get('mood', ''),
        'text': entry.get('text', ''),
        'date': format_timestamp(entry.get('created_at'), '%d %b %Y, %I:%M %p')
    } for _, entry in sorted(recent.items(), reverse=True)]

    return render_template('activities.html', user_name=user_name, user_initials=user_initials,
                           journal=journal, moods=JOURNAL_MOODS, journal_max=JOURNAL_MAX_LENGTH)


@app.route('/api/save-journal', methods=['POST'])
@login_required
@role_required('patient')
def save_journal():
    import time
    from datetime import datetime
    try:
        data = request.get_json() or {}
        uid = session['user']['uid']
        text = str(data.get('text', '')).strip()
        mood = str(data.get('mood', '')).strip()

        if not text:
            return jsonify({'success': False, 'error': 'Please write something before saving.'}), 400
        if len(text) > JOURNAL_MAX_LENGTH:
            return jsonify({'success': False, 'error': f'Entries are limited to {JOURNAL_MAX_LENGTH} characters.'}), 400
        if mood and mood not in JOURNAL_MOODS:
            return jsonify({'success': False, 'error': 'Invalid mood option.'}), 400

        now_ms = int(time.time() * 1000)
        entry = {'text': text, 'created_at': {'.sv': 'timestamp'}}
        if mood:
            entry['mood'] = mood
        db.reference(f'users/{uid}/journal/{now_ms}').set(entry)

        return jsonify({'success': True, 'message': 'Journal entry saved.',
                        'entry': {'mood': mood, 'text': text,
                                  'date': datetime.fromtimestamp(now_ms / 1000).strftime('%d %b %Y, %I:%M %p')}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/order-wristband', methods=['POST'])
@login_required
@role_required('patient')
def submit_wristband_order():
    try:
        data = request.get_json()
        uid = session['user']['uid']

        if db.reference(f'wristband_orders/{uid}').get():
            return jsonify({'success': False,
                            'error': 'You already have an active order. View it on the Wristband page.',
                            'redirect': url_for('patient_wristband')}), 409

        order = {
            'uid': uid,
            'fullname': data.get('fullname', ''),
            'phone': data.get('phone', ''),
            'address': data.get('address', ''),
            'city': data.get('city', ''),
            'size': data.get('size', 'M'),
            'notes': data.get('notes', ''),
            'status': 'pending',
            'created_at': {'.sv': 'timestamp'}
        }

        order_ref = db.reference(f'wristband_orders/{uid}')
        order_ref.set(order)

        return jsonify({'success': True, 'message': 'Order placed successfully', 'redirect': url_for('patient_wristband')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

RISK_LEVELS = ['low', 'moderate', 'high', 'critical']


def doctor_identity():
    user_name = session['user'].get('name', 'Doctor')
    # Avoid "Dr. Dr. ..." when the stored name already carries the title
    if user_name.lower().replace('.', '').split()[:1] == ['dr']:
        user_name = user_name.split(None, 1)[1] if ' ' in user_name else user_name
    user_initials = ''.join([word[0].upper() for word in user_name.split()[:2]]) or 'D'
    return user_name, user_initials


def load_active_patients():
    """Summaries of all active patients (scored on their latest assessment) plus every
    timestamped assessment across them, from a single read of the users tree."""
    users = db.reference('users').get() or {}

    patients, recent = [], []
    for uid, u in users.items():
        if not isinstance(u, dict) or u.get('role') != 'patient' or u.get('status') != 'active':
            continue
        name = u.get('name') or 'Unnamed patient'
        assessments = u.get('assessments') or {}

        latest = assessments.get(u.get('latest_assessment_id')) if u.get('latest_assessment_id') else None
        patient = {'uid': uid, 'name': name, 'email': u.get('email', ''),
                   'composite_score': None, 'risk_level': None, 'last_assessment': None, 'facial_dominant': None}
        if latest:
            score = compute_composite(latest)
            risk_level, risk_label, risk_color = risk_band(score)
            emotions = latest.get('facial_emotions', {}) or {}
            patient.update(composite_score=score, risk_level=risk_level, risk_label=risk_label, risk_color=risk_color,
                           last_assessment=format_timestamp(latest.get('created_at'), '%d %b %Y'),
                           facial_dominant=max(emotions, key=emotions.get).capitalize() if emotions else None)
        patients.append(patient)

        for a in assessments.values():
            if isinstance(a, dict) and isinstance(a.get('created_at'), (int, float)):
                score = compute_composite(a)
                _, risk_label, risk_color = risk_band(score)
                recent.append({'name': name, 'composite_score': score, 'risk_label': risk_label,
                               'risk_color': risk_color, 'created_at': a['created_at']})

    return patients, recent


def risk_distribution(patients):
    counts = {level: 0 for level in RISK_LEVELS}
    for p in patients:
        if p['risk_level']:
            counts[p['risk_level']] += 1
    return counts


@app.route('/doctor/patients')
@login_required
@role_required('doctor')
def doctor_patients():
    user_name, user_initials = doctor_identity()
    patients, _ = load_active_patients()
    counts = risk_distribution(patients)

    risk = request.args.get('risk', 'all').lower()
    if risk not in RISK_LEVELS:
        risk = 'all'
    query = request.args.get('q', '').strip()
    selected_id = request.args.get('id', '')

    shown = patients
    if risk != 'all':
        shown = [p for p in shown if p['risk_level'] == risk]
    if query:
        q = query.lower()
        shown = [p for p in shown if q in p['name'].lower() or q in p['email'].lower()]
    # Highest risk first; patients without an assessment go last, alphabetically
    shown = sorted(shown, key=lambda p: (p['composite_score'] is None, -(p['composite_score'] or 0), p['name'].lower()))

    return render_template('doctor_patients.html',
                           user_name=user_name, user_initials=user_initials,
                           patients=shown, total_patients=len(patients),
                           counts=counts, unassessed_count=sum(1 for p in patients if p['risk_level'] is None),
                           risk=risk, query=query, selected_id=selected_id, risk_levels=RISK_LEVELS)


@app.route('/doctor/dashboard')
@login_required
@role_required('doctor')
def doctor_dashboard():
    user_name, user_initials = doctor_identity()
    patients, recent = load_active_patients()

    assessed = [p for p in patients if p['composite_score'] is not None]
    distribution = risk_distribution(patients)

    avg_score = round(sum(p['composite_score'] for p in assessed) / len(assessed)) if assessed else None

    recent.sort(key=lambda a: a['created_at'], reverse=True)
    recent = recent[:5]
    for a in recent:
        a['time_ago'] = time_ago(a['created_at'])

    return render_template('doctor_dashboard.html',
                           user_name=user_name, user_initials=user_initials,
                           total_patients=len(patients),
                           unassessed_count=len(patients) - len(assessed),
                           critical_count=distribution['critical'],
                           high_count=distribution['high'],
                           avg_score=avg_score,
                           avg_label=risk_band(avg_score)[1] if avg_score is not None else None,
                           distribution=distribution,
                           distribution_total=len(assessed),
                           top_patients=sorted(assessed, key=lambda p: p['composite_score'], reverse=True)[:3],
                           recent_assessments=recent)


@app.route('/admin/dashboard')
@login_required
@role_required('admin')
def admin_dashboard():
    return f"<h1>Admin Dashboard</h1><p>Welcome, {session['user']['name']}!</p><p>Role: {session['user']['role']}</p><a href='/logout'>Logout</a>"


# ========== LOGOUT ==========
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)