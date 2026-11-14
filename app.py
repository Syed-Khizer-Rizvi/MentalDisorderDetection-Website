from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, has_request_context
import firebase_admin
from firebase_admin import credentials, auth, db
import os
import hashlib
import hmac
import secrets
import time
from functools import wraps
from deepface import DeepFace
import base64
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv()  # local settings such as ADMIN_UID live in .env (not committed)

# ========== INITIALIZE FLASK ==========
app = Flask(__name__)
# Session-signing key from .env (SECRET_KEY, generate with: python -c "import secrets; print(secrets.token_hex(32))").
# Without it, a random key is made at each start: nobody can forge sessions, but everyone is signed out on restart.
SECRET_KEY_FROM_ENV = bool(os.environ.get('SECRET_KEY', '').strip())
if SECRET_KEY_FROM_ENV:
    app.secret_key = os.environ['SECRET_KEY'].strip()
else:
    app.secret_key = secrets.token_hex(32)
    print('WARNING: SECRET_KEY is not set in .env, so a temporary random key is being used. '
          'Sessions will end whenever the server restarts. Add SECRET_KEY to .env.', flush=True)
# Placeholder never matches a real Firebase UID, so no one is admin until .env sets it
ADMIN_UID = os.environ.get('ADMIN_UID', 'SET_IN_ENV_FILE').strip()

# ========== RATE LIMITING ==========
# Per client IP. In-memory storage: counters live in this process and reset on restart, and each
# worker process counts separately; production with several workers should use Redis
# (storage_uri='redis://...'). Behind a reverse proxy, remote_addr is the proxy's address, so
# ProxyFix would be needed for real client IPs.
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Endpoints that poll or stream frames: the default 50/hour would break them in normal use
# (the alert badge polls every minute, Live Mode every 10 s); they're still behind login.
RATE_LIMIT_EXEMPT = {'static', 'alerts_unread', 'chat_unread', 'chat_token', 'admin_activity_logs'}
RATE_LIMITS = {   # shown on the Security page; applied with @limiter.limit on each route
    'verify_token': '10 per minute', 'register_profile': '5 per hour', 'wristband_pair': '10 per hour',
    'wristband_data': '120 per minute', 'admin_login': '10 per minute', 'admin_mfa_verify': '5 per 15 minutes',
    'analyze_emotion': '60 per minute',
}
limiter = Limiter(
    get_remote_address, app=app,
    default_limits=['200 per day', '50 per hour'],
    default_limits_exempt_when=lambda: request.endpoint in RATE_LIMIT_EXEMPT or request.endpoint in RATE_LIMITS,
    storage_uri='memory://', headers_enabled=True, strategy='fixed-window',
)

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


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('user', {}).get('role') != 'admin':
            if 'user' not in session:
                return redirect(url_for('login'))
            flash('Access denied: administrator access is required.', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated


def admin_csrf_valid():
    submitted = request.form.get('csrf_token', '')
    expected = session.get('_admin_csrf_token', '')
    return bool(submitted and expected and hmac.compare_digest(submitted, expected))


# ========== FACIAL EMOTION MODEL ==========
# Custom CNN trained for this FYP on the FER-2013 dataset (48x48 grayscale faces, 7 emotions).
# Version 2: 7.19M parameters, 60 epochs: 67.5% test accuracy, 67.1% validation accuracy.
# Details in Models/model_metadata.json. Loaded once here, not per request.
# If the model file is missing or fails to load, the scan falls back to DeepFace, which is why
# deepface stays in requirements.txt.
FER_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']   # model output order
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# The folder is "Models" on disk; check "models" too for case-sensitive (Linux) hosts
FER_MODEL_PATH = next((p for p in (os.path.join(APP_DIR, d, 'fer_emotion_model.keras') for d in ('Models', 'models'))
                       if os.path.isfile(p)), None)


def load_fer_model(path):
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        # TensorFlow has no build for Python 3.14, so run the same .keras file on Keras 3 + JAX.
        # The backend must be chosen before keras is first imported.
        os.environ.setdefault('KERAS_BACKEND', 'jax')
        from keras.models import load_model
    model = load_model(path, compile=False)
    model.predict(np.zeros((1, 48, 48, 1), dtype='float32'), verbose=0)   # warm up, so the first scan isn't slow
    return model


def load_face_cascade():
    # OpenCV 4 wheels bundle the cascade; OpenCV 5 wheels don't, so a copy lives in Models/
    name = 'haarcascade_frontalface_default.xml'
    candidates = [os.path.join(cv2.data.haarcascades, name)] if hasattr(cv2, 'data') else []
    candidates += [os.path.join(APP_DIR, d, name) for d in ('Models', 'models')]
    for path in candidates:
        if os.path.isfile(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade
    return None


FER_MODEL = FACE_CASCADE = None
if FER_MODEL_PATH is None:
    app.logger.warning('Custom emotion model not found at Models/fer_emotion_model.keras; using DeepFace')
else:
    try:
        FACE_CASCADE = load_face_cascade()
        if FACE_CASCADE is None:
            raise FileNotFoundError('haarcascade_frontalface_default.xml not found')
        FER_MODEL = load_fer_model(FER_MODEL_PATH)
    except Exception as e:
        FER_MODEL = None
        app.logger.warning('Could not load the custom emotion model (%s); using DeepFace', e)


def analyze_facial_emotion(image):
    """Emotions for the largest face in `image` (encoded image bytes or a BGR/gray array), using the custom CNN.

    Returns DeepFace's shape, {'emotion': {name: percentage 0-100}, 'dominant_emotion': name},
    or None when no face is found.
    """
    if isinstance(image, (bytes, bytearray)):
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = cv2.resize(gray[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA)
    batch = (face.astype('float32') / 255.0).reshape(1, 48, 48, 1)
    probs = FER_MODEL.predict(batch, verbose=0)[0]
    emotion = {name: float(p) * 100 for name, p in zip(FER_CLASSES, probs)}
    return {'emotion': emotion, 'dominant_emotion': max(emotion, key=emotion.get)}


# ========== SCORING HELPERS ==========
# ========== TUNABLE SCORING SETTINGS (admin ML Models page) ==========
# Stored at system/fusion_weights and system/risk_thresholds; these defaults apply when unset or invalid.
FUSION_KEYS = ('phq', 'gad', 'pss', 'facial')
DEFAULT_FUSION_WEIGHTS = {'phq': 0.25, 'gad': 0.25, 'pss': 0.25, 'facial': 0.25}
DEFAULT_RISK_THRESHOLDS = {'low': 25, 'moderate': 50, 'high': 70}
RISK_COLORS = {'low': '#059669', 'moderate': '#D97706', 'high': '#EA580C', 'critical': '#DC2626'}
SETTINGS_TTL = 30   # seconds; risk_band() runs many times per page, so settings are cached briefly
_scoring_cache = {'at': 0.0, 'weights': DEFAULT_FUSION_WEIGHTS, 'thresholds': DEFAULT_RISK_THRESHOLDS}


def clean_fusion_weights(raw):
    """Weights rounded to 0.05 steps, each 0-1 and summing to 1; None if `raw` isn't a valid set."""
    if not isinstance(raw, dict):
        return None
    try:
        weights = {k: round(float(raw[k]) * 20) / 20 for k in FUSION_KEYS}
    except (KeyError, TypeError, ValueError):
        return None
    if any(not 0 <= w <= 1 for w in weights.values()) or abs(sum(weights.values()) - 1) > 1e-6:
        return None
    return weights


def clean_risk_thresholds(raw):
    """Upper bounds for Low/Moderate/High (Critical is above High); None if out of the allowed ranges."""
    if not isinstance(raw, dict):
        return None
    try:
        values = {k: raw[k] for k in DEFAULT_RISK_THRESHOLDS}
        if any(isinstance(v, bool) or float(v) != int(float(v)) for v in values.values()):
            return None
        low, moderate, high = (int(float(values[k])) for k in ('low', 'moderate', 'high'))
    except (KeyError, TypeError, ValueError):
        return None
    if not (10 <= low <= 40 and low + 5 <= moderate <= 70 and moderate + 5 <= high <= 90):
        return None
    return {'low': low, 'moderate': moderate, 'high': high}


def scoring_settings(refresh=False):
    """Current weights and thresholds from Firebase, cached for SETTINGS_TTL seconds."""
    if refresh or time.time() - _scoring_cache['at'] > SETTINGS_TTL:
        try:
            system = db.reference('system').get() or {}
            system = system if isinstance(system, dict) else {}
            _scoring_cache['weights'] = clean_fusion_weights(system.get('fusion_weights')) or DEFAULT_FUSION_WEIGHTS
            _scoring_cache['thresholds'] = clean_risk_thresholds(system.get('risk_thresholds')) or DEFAULT_RISK_THRESHOLDS
        except Exception:
            pass   # keep the last known settings (defaults at startup) if Firebase is unreachable
        _scoring_cache['at'] = time.time()
    return _scoring_cache


def fusion_weights():
    return dict(scoring_settings()['weights'])


def risk_thresholds():
    return dict(scoring_settings()['thresholds'])


def risk_ranges():
    """The four bands with their score ranges, for labels, legends and charts."""
    t = risk_thresholds()
    bounds = [('low', 'Low', 0, t['low']), ('moderate', 'Moderate', t['low'] + 1, t['moderate']),
              ('high', 'High', t['moderate'] + 1, t['high']), ('critical', 'Critical', t['high'] + 1, 100)]
    return [{'level': level, 'label': label, 'min': lo, 'max': hi, 'range': f'{lo}-{hi}', 'color': RISK_COLORS[level]}
            for level, label, lo, hi in bounds]


def compute_composite(assessment):
    phq = assessment.get('phq_score', 0)
    gad = assessment.get('gad_score', 0)
    pss = assessment.get('pss_score', 0)
    emotions = assessment.get('facial_emotions', {}) or {}
    neg = sum(emotions.get(k, 0) for k in ['sad', 'angry', 'fear', 'disgust'])
    weights = clean_fusion_weights(assessment.get('fusion_weights'))
    if weights is None:
        # Assessments saved before fusion weights were configurable: the original equal-weight
        # formula, so their scores never change
        questionnaires = (phq/27 + gad/21 + pss/40) * 100
        # Skipped facial scan: average the three questionnaires only, so the score stays on 0-100
        if assessment.get('facial_skipped'):
            return round(questionnaires / 3)
        return round((questionnaires + neg) / 4)
    # Newer assessments carry the weights they were scored with (see submit_facial)
    parts = {'phq': phq / 27 * 100, 'gad': gad / 21 * 100, 'pss': pss / 40 * 100}
    if assessment.get('facial_skipped'):
        # No facial part: share its weight across the questionnaires in proportion
        total = sum(weights[k] for k in parts)
        if total <= 0:
            return round(sum(parts.values()) / 3)
        return round(sum(weights[k] * parts[k] for k in parts) / total)
    parts['facial'] = neg
    return round(sum(weights[k] * parts[k] for k in FUSION_KEYS))


def facial_mood(assessment, skipped_text='Skipped', missing=None):
    """Dominant facial emotion for display, `skipped_text` when the patient skipped the scan."""
    if assessment.get('facial_skipped'):
        return skipped_text
    emotions = assessment.get('facial_emotions', {}) or {}
    return max(emotions, key=emotions.get).capitalize() if emotions else missing


def risk_band(composite):
    """(level, label, colour) using the admin-set thresholds (default Low <=25, Moderate <=50, High <=70)."""
    t = risk_thresholds()
    if composite <= t['low']:
        return 'low', 'Low', RISK_COLORS['low']
    elif composite <= t['moderate']:
        return 'moderate', 'Moderate', RISK_COLORS['moderate']
    elif composite <= t['high']:
        return 'high', 'High', RISK_COLORS['high']
    return 'critical', 'Critical', RISK_COLORS['critical']


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


def age_from_dob(dob):
    """Whole years since an ISO date string (YYYY-MM-DD); None if missing or invalid."""
    from datetime import date
    try:
        born = date.fromisoformat(str(dob)) if dob else None
    except ValueError:
        return None
    if not born:
        return None
    today = date.today()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    return age if age >= 0 else None


def build_risk_trend(assessments, limit, detail_url=None):
    """Chart points for templates/_risk_trend.html: the latest `limit` timestamped
    assessments ({id: assessment}), oldest first. `detail_url(id)` links each point."""
    from datetime import datetime
    dated = sorted(((aid, a) for aid, a in (assessments or {}).items()
                    if isinstance(a, dict) and isinstance(a.get('created_at'), (int, float))),
                   key=lambda item: item[1]['created_at'])[-limit:]

    risk_trend = []
    for aid, a in dated:
        score = compute_composite(a)
        level, label, color = risk_band(score)
        created = datetime.fromtimestamp(a['created_at'] / 1000)
        risk_trend.append({
            'score': score,
            'risk_level': level,
            'risk_label': label,
            'risk_color': color,
            'url': detail_url(aid) if detail_url else '',
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
    return risk_trend


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
        facial_skipped=bool(assessment.get('facial_skipped')),
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


# ========== DOCTOR ASSIGNMENT ==========
# users/{patient_uid}/assigned_doctor_uid, assigned_at (server timestamp) and assigned_by:
# 'self' (patient chose at registration), 'auto' (least-loaded doctor) or 'admin:<admin_uid>'.

def strip_dr(name):
    """'Dr. Sarah Ahmed' -> 'Sarah Ahmed', so templates can add the title once."""
    name = (name or '').strip()
    if name.lower().replace('.', '').split()[:1] == ['dr'] and ' ' in name:
        return name.split(None, 1)[1]
    return name


def doctor_summary(uid, user):
    profile = user.get('profile') or {}
    name = strip_dr(user.get('name')) or 'Doctor'
    return {
        'uid': uid, 'name': name,
        'initials': ''.join(w[0].upper() for w in name.split()[:2]) or 'D',
        'specialization': profile.get('specialization', ''),
        'affiliation': profile.get('affiliation', ''),
        'years_experience': profile.get('years_experience'),
    }


def load_doctor_summary(doctor_uid):
    """Public summary of a patient's assigned doctor, or None if unset or the account is gone."""
    if not doctor_uid or not is_safe_key(doctor_uid):
        return None
    doctor = db.reference(f'users/{doctor_uid}').get()
    if not isinstance(doctor, dict) or doctor.get('role') != 'doctor':
        return None
    return doctor_summary(doctor_uid, doctor)


def load_approved_doctors(users=None):
    """Approved doctors with the public details patients see when choosing one."""
    users = users if users is not None else (db.reference('users').get() or {})
    doctors = [doctor_summary(uid, u) for uid, u in users.items()
               if isinstance(u, dict) and u.get('role') == 'doctor' and u.get('status') == 'approved']
    return sorted(doctors, key=lambda d: d['name'].lower())


def get_approved_doctor(doctor_uid):
    """The doctor's summary if doctor_uid belongs to an approved doctor, else None."""
    if not is_safe_key(doctor_uid):
        return None
    user = db.reference(f'users/{doctor_uid}').get()
    if not isinstance(user, dict) or user.get('role') != 'doctor' or user.get('status') != 'approved':
        return None
    return doctor_summary(doctor_uid, user)


def auto_assign_doctor(patient_uid):
    """The approved doctor with the fewest active patients (random among ties), or None.
    Only picks; the caller saves the assignment with assign_doctor()."""
    import random
    users = db.reference('users').get() or {}
    load = {d['uid']: 0 for d in load_approved_doctors(users)}
    if not load:
        return None
    for uid, u in users.items():
        if (uid != patient_uid and isinstance(u, dict) and u.get('role') == 'patient'
                and u.get('status') == 'active' and u.get('assigned_doctor_uid') in load):
            load[u['assigned_doctor_uid']] += 1
    fewest = min(load.values())
    return random.choice([uid for uid, count in load.items() if count == fewest])


def assign_doctor(patient_uid, doctor_uid, assigned_by):
    previous = db.reference(f'users/{patient_uid}/assigned_doctor_uid').get()
    updates = {
        f'users/{patient_uid}/assigned_doctor_uid': doctor_uid,
        f'users/{patient_uid}/assigned_at': {'.sv': 'timestamp'},   # always server time, never client input
        f'users/{patient_uid}/assigned_by': assigned_by,
    }
    if previous and previous != doctor_uid and is_safe_key(previous):
        # The old conversation becomes inaccessible; drop its unread counters so no badge lingers
        old_chat = chat_id_for(previous, patient_uid)
        updates[f'chat_unread/{previous}/{old_chat}'] = None
        updates[f'chat_unread/{patient_uid}/{old_chat}'] = None
    db.reference().update(updates)


def patient_assigned_to(patient, doctor_uid):
    return isinstance(patient, dict) and bool(doctor_uid) and patient.get('assigned_doctor_uid') == doctor_uid


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
    # Public page: only the details a patient needs to choose (no contact info)
    doctors = [{'uid': d['uid'], 'name': d['name'], 'specialization': d['specialization'],
                'affiliation': d['affiliation']} for d in load_approved_doctors()]
    return render_template('register.html', firebase_config=FIREBASE_WEB_CONFIG, doctors=doctors)


@app.route('/admin/login', methods=['GET', 'POST'])
@limiter.limit(RATE_LIMITS['admin_login'])
def admin_login():
    return redirect(url_for('login'))


# ========== API ROUTES (called by frontend JS) ==========
@app.route('/api/verify-token', methods=['POST'])
@limiter.limit(RATE_LIMITS['verify_token'])
def verify_token():
    try:
        data = request.get_json()
        id_token = data.get('idToken')

        decoded = auth.verify_id_token(id_token, clock_skew_seconds=10)
        uid = decoded['uid']
        email = decoded.get('email', '')

        user_ref = db.reference(f'users/{uid}')
        profile = user_ref.get()

        if uid == ADMIN_UID:
            profile = profile if isinstance(profile, dict) else {}
            admin_profile = {
                'uid': uid,
                'email': email or profile.get('email', ''),
                'name': profile.get('name') or decoded.get('name') or 'Administrator',
                'role': 'admin',
                'status': 'active'
            }
            if not profile.get('created_at'):
                admin_profile['created_at'] = {'.sv': 'timestamp'}
            user_ref.update(admin_profile)
            profile = {**profile, **admin_profile}

        if not isinstance(profile, dict):
            # Signed in to Firebase (e.g. first Google sign-in) but never registered here
            activity_log('login_failed', 'Sign-in failed: no registered profile', 'failed',
                         {'email': email, 'reason': 'not_registered'}, user={'uid': uid, 'name': email or 'Unknown', 'role': 'guest'})
            return jsonify({'success': False, 'needs_registration': True,
                            'error': 'User profile not found. Please create your account first.'}), 404

        if profile.get('role') == 'doctor' and profile.get('status') != 'approved':
            activity_log('login_failed', f"Sign-in blocked: doctor account {profile.get('status') or 'not approved'}", 'failed',
                         {'email': email, 'reason': f"doctor_{profile.get('status') or 'unknown'}"},
                         user={'uid': uid, 'name': profile.get('name') or email, 'role': 'doctor'})
            message = ('Your doctor account is pending approval by admin.'
                       if profile.get('status') == 'pending'
                       else 'Your doctor account was not approved. Please contact the administrator.')
            return jsonify({'success': False, 'error': message}), 403

        user_record = {
            'uid': uid,
            'email': email,
            'name': profile.get('name', ''),
            'role': profile.get('role', 'patient'),
            'status': profile.get('status', 'active')
        }
        if uid == ADMIN_UID and mfa_enrolled_profile(profile):
            # Password accepted, but no session yet: /admin/mfa-verify creates it after the second factor
            session.pop('user', None)
            session['_mfa_pending'] = {'user': user_record, 'at': time.time()}
            activity_log('mfa_challenge', 'Password accepted; 2FA code requested', 'info', details={'email': email}, user=user_record)
            return jsonify({'success': True, 'mfa_required': True, 'redirect': url_for('admin_mfa_verify')})

        session['user'] = user_record
        session['_login_toast'] = True   # one-time: popped by the next dashboard load
        try:
            # Daily login total for the admin analytics "Activities Completed" figure (server-local day)
            db.reference(f"login_counts/{time.strftime('%Y-%m-%d')}").set({'.sv': {'increment': 1}})
        except Exception:
            pass   # analytics only; never block a login on it
        activity_log('login', 'Logged in', details={'email': email})

        return jsonify({'success': True, 'role': profile.get('role'), 'redirect': url_for('dashboard')})

    except Exception as e:
        # Usually an invalid or expired ID token. Wrong passwords never get here: Firebase
        # rejects them in the browser before any token is sent.
        activity_log('login_failed', 'Sign-in failed: token rejected', 'failed',
                     {'reason': type(e).__name__}, user={'uid': 'anonymous', 'name': 'Unknown', 'role': 'guest'})
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/register-profile', methods=['POST'])
@limiter.limit(RATE_LIMITS['register_profile'])
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

        # Signed-in users (e.g. Google) can call this again; never overwrite an existing profile
        if db.reference(f'users/{uid}').get() is not None:
            return jsonify({'success': False, 'error': 'This account is already registered. Please sign in.'}), 409

        # Only remove an Auth account created by this sign-up form. A Google account would just be
        # re-created by the next popup, while the client still holds its old token.
        password_signup = decoded.get('firebase', {}).get('sign_in_provider') == 'password'

        def discard_auth_account():
            if password_signup:
                try:
                    auth.delete_user(uid)
                except Exception:
                    pass

        if role not in ('patient', 'doctor'):
            discard_auth_account()
            if role == 'admin':
                return jsonify({'success': False, 'error': 'Admin accounts cannot be created through public registration.'}), 403
            return jsonify({'success': False, 'error': 'Choose Patient or Doctor.'}), 400

        status = 'pending' if role == 'doctor' else 'active'

        record = {
            'uid': uid,
            'name': name,
            'email': email,
            'phone': phone,
            'role': role,
            'status': status,
            'created_at': {'.sv': 'timestamp'}
        }

        if role == 'patient':
            doctor_uid = str(data.get('doctor_uid') or '').strip()
            error = None
            if doctor_uid:
                if not get_approved_doctor(doctor_uid):
                    error = 'The selected doctor is not available. Please choose another doctor.'
            elif load_approved_doctors():
                error = 'Please choose your doctor.'
            if error:
                # The Firebase Auth account was created before this call; remove it so the
                # patient can retry with the same email instead of being locked out
                discard_auth_account()
                return jsonify({'success': False, 'error': error}), 400
            if doctor_uid:
                record.update(assigned_doctor_uid=doctor_uid, assigned_at={'.sv': 'timestamp'}, assigned_by='self')
            # No approved doctors yet: register unassigned; the dashboard auto-assigns later

        db.reference(f'users/{uid}').set(record)

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

    # Patients without a doctor (registered before assignment existed, or when no doctor
    # was approved yet) are given the least-loaded approved doctor on their next visit
    doctor_uid = db.reference(f'users/{uid}/assigned_doctor_uid').get()
    if not doctor_uid:
        doctor_uid = auto_assign_doctor(uid)
        if doctor_uid:
            assign_doctor(uid, doctor_uid, 'auto')
    assigned_doctor = load_doctor_summary(doctor_uid)

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
            'facial_dominant': facial_mood(latest, 'Not scanned', 'Neutral'),
            'facial_skipped': bool(latest.get('facial_skipped')), 'composite_score': composite,
            'risk_level': risk_level, 'risk_label': risk_label, 'risk_color': risk_color,
            'assessment_date': date_str
        })

    # Assessment keys are millisecond timestamps, so key order is chronological
    recent = db.reference(f'users/{uid}/assessments').order_by_key().limit_to_last(7).get() or {}
    risk_trend = build_risk_trend(recent, 7, lambda aid: url_for('patient_history_detail', assessment_id=aid))

    wristband_order = db.reference(f'wristband_orders/{uid}').get()

    return render_template('patient_dashboard.html',
                           user_name=user_name, user_initials=user_initials,
                           wristband_order=wristband_order, risk_trend=risk_trend, wb=load_wristband_view(uid, 1),
                           assigned_doctor=assigned_doctor,
                           login_just_happened=session.pop('_login_toast', False), **data)


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
@limiter.limit(RATE_LIMITS['analyze_emotion'])
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

        if FER_MODEL is not None:
            result = analyze_facial_emotion(img)
            if result is None:
                return jsonify({'success': False, 'error': 'No face detected'}), 400
        else:
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
        data = request.get_json() or {}
        emotions = data.get('emotions') or {}
        # The scan is optional: an explicit skip, or no usable readings, saves a questionnaire-only assessment
        skipped = data.get('skipped') is True or not isinstance(emotions, dict) or not emotions
        readings = 0 if skipped else data.get('readings', 0)
        peak = {} if skipped else data.get('peak', {})
        if skipped:
            emotions = {}
        uid = session['user']['uid']

        if 'assessment' in session:
            session['assessment']['facial_emotions'] = emotions
            session['assessment']['facial_readings'] = readings
            session['assessment']['facial_peak'] = peak
            session['assessment']['facial_skipped'] = skipped
            session.modified = True

            assessment_id = str(int(time.time() * 1000))
            full_assessment = {
                'phq_score': session['assessment'].get('phq_score', 0),
                'gad_score': session['assessment'].get('gad_score', 0),
                'pss_score': session['assessment'].get('pss_score', 0),
                'phq_answers': session['assessment'].get('phq_answers', []),
                'gad_answers': session['assessment'].get('gad_answers', []),
                'pss_answers': session['assessment'].get('pss_answers', []),
                'created_at': {'.sv': 'timestamp'}
            }
            # Score with the fusion weights in force now, kept on the assessment so later
            # weight changes don't alter it (compute_composite reads them back)
            full_assessment['fusion_weights'] = fusion_weights()
            if skipped:
                # Firebase drops None/empty values, so record the skip explicitly
                full_assessment['facial_skipped'] = True
            else:
                full_assessment.update(facial_emotions=emotions, facial_peak=peak, facial_readings=readings)
            db.reference(f'users/{uid}/assessments/{assessment_id}').set(full_assessment)
            db.reference(f'users/{uid}/latest_assessment_id').set(assessment_id)
            score = compute_composite(full_assessment)
            level, risk_label, _color = risk_band(score)
            activity_log('assessment_completed', 'Completed an assessment',
                         details={'composite': score, 'risk': risk_label, 'facial_scan': 'skipped' if skipped else 'completed'})
            if level == 'critical':
                patient_name = session['user'].get('name') or 'a patient'
                activity_log('critical_alert', f'Critical alert raised for {patient_name} (composite {score}/100)', 'alert',
                             details={'patient_uid': uid, 'alert_id': alert_display_id(uid, assessment_id), 'composite': score},
                             user=SYSTEM_ACTOR)

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
    return render_template('results.html', assigned_doctor=patient_assigned_doctor(),
                           **build_results_context(assessment))


def patient_assigned_doctor():
    """The signed-in patient's current doctor (results pages show and 'alert' this doctor)."""
    return load_doctor_summary(db.reference(f"users/{session['user']['uid']}/assigned_doctor_uid").get())


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
            'facial_dominant': facial_mood(a, missing='-'),
            'facial_skipped': bool(a.get('facial_skipped'))
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

    return render_template('results.html', read_only=True, assigned_doctor=patient_assigned_doctor(),
                           assessment_date=format_timestamp(assessment.get('created_at'), '%d %b %Y, %I:%M %p'),
                           **build_results_context(assessment))


@app.route('/patient/order-wristband')
@login_required
@role_required('patient')
def patient_order_wristband():
    if db.reference(f"wristband_orders/{session['user']['uid']}").get():
        return redirect(url_for('patient_wristband'))
    user_name = session['user'].get('name', '')
    return render_template('order_wristband.html', user_name=user_name, price=WRISTBAND_PRICE_PKR,
                           product=WRISTBAND_PRODUCT, delivery=WRISTBAND_DELIVERY, payment=WRISTBAND_PAYMENT)


WRISTBAND_STATUSES = ['pending', 'confirmed', 'shipped', 'delivered']

# ========== WRISTBAND PRICING ==========
# Fixed price, cash on delivery. Set only here: neither patients nor admins can change it,
# and the server never trusts a price sent by the browser.
WRISTBAND_PRICE_PKR = 25_000
WRISTBAND_PRODUCT = 'Mental Wellness Wristband'
WRISTBAND_DELIVERY = '7-10 business days'
WRISTBAND_PAYMENT = 'Cash on Delivery'
WRISTBAND_INCLUDED = ['ESP32-based smart wristband', 'Heart rate monitoring (MAX30102)', 'GSR stress sensor',
                      'Body temperature sensor (MLX90614)', 'OLED display', '3D printed comfortable strap',
                      'Companion web app access', '6 months warranty']


def format_pkr(amount):
    """25000 -> 'Rs. 25,000'."""
    try:
        return f'Rs. {int(amount):,}'
    except (TypeError, ValueError):
        return 'Rs. —'


app.jinja_env.filters['pkr'] = format_pkr


def order_price(order):
    """The stored price; orders placed before pricing existed are the standard price."""
    price = order.get('price') if isinstance(order, dict) else None
    return price if isinstance(price, int) and not isinstance(price, bool) and price > 0 else WRISTBAND_PRICE_PKR


def _base36(n):
    digits = ''
    n = int(n)
    while True:
        n, r = divmod(n, 36)
        digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[r] + digits
        if not n:
            return digits


def new_order_reference():
    return f'MW-{_base36(int(time.time() * 1000))[-6:]}-{secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")}{secrets.randbelow(10)}'


def order_reference(uid, order):
    """The stored reference, or a stable one for orders placed before references existed."""
    if isinstance(order, dict) and order.get('ref'):
        return order['ref']
    created = order.get('created_at') if isinstance(order, dict) else None
    return f"MW-{_base36(created)[-6:] if isinstance(created, (int, float)) else uid[:6].upper()}"


def new_order_record(uid, fields):
    """Order fields plus the server-set price, reference and payment method."""
    return {**fields, 'uid': uid, 'price': WRISTBAND_PRICE_PKR, 'currency': 'PKR', 'payment_method': 'cod',
            'ref': new_order_reference(), 'status': 'pending', 'created_at': {'.sv': 'timestamp'}}


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
                     order_date=format_timestamp(order.get('created_at'), '%d %b %Y, %I:%M %p'),
                     price=order_price(order), ref=order_reference(uid, order))

    return render_template('wristband.html', user_name=user_name, user_initials=user_initials,
                           order=order, statuses=WRISTBAND_STATUSES, wb=load_wristband_view(uid, 1),
                           price=WRISTBAND_PRICE_PKR, product=WRISTBAND_PRODUCT, included=WRISTBAND_INCLUDED,
                           delivery=WRISTBAND_DELIVERY, payment=WRISTBAND_PAYMENT)


# ========== WRISTBAND DEVICE (ESP32) ==========
# users/{uid}/wristband: device_id, status, paired_at, last_sync, latest. The patient and their
#   assigned doctor may read it, so it never holds the API key.
# wristband_devices/{device_id}: {uid, api_key_hash}. Server-only lookup, so each device POST is
#   one small read instead of a scan of every user.
# users/{uid}/wristband_readings/{key}: every reading. users/{uid}/wristband_hourly/h{epoch hour}:
#   running totals per hour, which the charts read instead of thousands of raw readings.

WRISTBAND_RANGES = {'gsr': (0, 100000), 'heart_rate': (20, 250), 'temperature': (20, 45)}
WRISTBAND_ACTIVE_MS = 10 * 60 * 1000   # synced within 10 minutes: Active; within an hour: Idle
WRISTBAND_IDLE_MS = 60 * 60 * 1000
# Stress indicator (0-100) from heart rate and GSR. NOT clinically calibrated: tune these two
# ranges once the real sensors are on a wrist. Each maps (calm, stressed) readings to 0 and 100;
# if the GSR sensor reads LOWER under stress, swap its pair.
STRESS_HR = (60, 120)
STRESS_GSR = (200, 800)


def stress_indicator(gsr, heart_rate):
    def scale(value, calm, stressed):
        return min(max((value - calm) / (stressed - calm), 0), 1)
    return round(100 * (scale(heart_rate, *STRESS_HR) + scale(gsr, *STRESS_GSR)) / 2)


def stress_band(score):
    if score < 35:
        return 'calm', 'Calm', '#059669'
    if score < 65:
        return 'elevated', 'Elevated', '#D97706'
    return 'high', 'High', '#EA580C'


def wristband_key_hash(api_key):
    return hashlib.sha256(api_key.encode()).hexdigest()


@app.route('/api/wristband/pair', methods=['POST'])
@limiter.limit(RATE_LIMITS['wristband_pair'])
@login_required
@role_required('patient')
def wristband_pair():
    """Creates (or rotates) the device credentials. The API key is returned once and only its
    hash is stored, so a lost key means regenerating."""
    if not request.is_json:   # JSON only, so a cross-site form post can't rotate the key
        return chat_json_error('Send a JSON request.', 400)
    uid = session['user']['uid']
    old_device = db.reference(f'users/{uid}/wristband/device_id').get()

    device_id = secrets.token_hex(6).upper()
    api_key = secrets.token_urlsafe(24)
    updates = {
        f'wristband_devices/{device_id}': {'uid': uid, 'api_key_hash': wristband_key_hash(api_key)},
        # Replaces the whole node: the old device's latest reading no longer counts as this one's
        f'users/{uid}/wristband': {'device_id': device_id, 'status': 'paired', 'paired_at': {'.sv': 'timestamp'}},
    }
    if isinstance(old_device, str) and is_safe_key(old_device) and old_device != device_id:
        updates[f'wristband_devices/{old_device}'] = None   # the old key stops working immediately
    db.reference().update(updates)
    if isinstance(old_device, str) and old_device:
        activity_log('wristband_revoked', f'Revoked wristband {old_device} and paired {device_id}',
                     details={'old_device_id': old_device, 'device_id': device_id})
    else:
        activity_log('wristband_paired', f'Paired wristband {device_id}', details={'device_id': device_id})
    return jsonify({'success': True, 'device_id': device_id, 'api_key': api_key,
                    'endpoint': url_for('wristband_data', _external=True)})


def wristband_device_config():
    """Sent back with every reading so the ESP32 can follow the admin's System Settings."""
    return {'interval_s': setting('wristband', 'interval_s'), 'buffer_min': setting('wristband', 'buffer_min')}


@app.route('/api/wristband/data', methods=['POST'])
@limiter.limit(RATE_LIMITS['wristband_data'])
def wristband_data():
    """Sensor readings from a paired ESP32. Sending the device's `timestamp` makes retries safe:
    a reading with the same timestamp is stored once."""
    import math
    import time
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return chat_json_error('Send a JSON object with Content-Type: application/json.', 400)

    device_id, api_key = data.get('device_id'), data.get('api_key')
    if not (isinstance(device_id, str) and is_safe_key(device_id) and isinstance(api_key, str) and api_key):
        return chat_json_error('Invalid device credentials.', 401)
    device = db.reference(f'wristband_devices/{device_id}').get()
    if not isinstance(device, dict) or not hmac.compare_digest(str(device.get('api_key_hash', '')),
                                                               wristband_key_hash(api_key)):
        return chat_json_error('Invalid device credentials.', 401)
    uid = device.get('uid')
    if not (isinstance(uid, str) and is_safe_key(uid)) or db.reference(f'users/{uid}/status').get() != 'active':
        return chat_json_error('Invalid device credentials.', 401)

    values = {}
    for field, (low, high) in WRISTBAND_RANGES.items():
        value = data.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or not low <= value <= high:
            return chat_json_error(f'{field} must be a number from {low} to {high}.', 400)
        values[field] = round(value, 2)

    now = int(time.time() * 1000)
    reading = dict(values, stress=stress_indicator(values['gsr'], values['heart_rate']), received_at=now)
    # Device clock in seconds (milliseconds accepted); ignored unless plausible, e.g. before NTP sync
    stamp = data.get('timestamp')
    if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and math.isfinite(stamp):
        seconds = int(stamp / 1000 if stamp > 1e11 else stamp)
        if 1_600_000_000 <= seconds <= now // 1000 + 86400:
            reading['device_ts'] = seconds

    if 'device_ts' in reading:
        key = f"t{reading['device_ts']}"
        if db.reference(f'users/{uid}/wristband_readings/{key}').get(shallow=True) is not None:
            return jsonify({'success': True, 'duplicate': True, 'config': wristband_device_config()})
    else:
        key = f'r{now}{secrets.token_hex(3)}'

    hour = f'users/{uid}/wristband_hourly/h{now // 3_600_000}'
    increment = lambda n: {'.sv': {'increment': n}}
    db.reference().update({   # one atomic multi-path write
        f'users/{uid}/wristband_readings/{key}': reading,
        f'users/{uid}/wristband/latest': reading,
        f'users/{uid}/wristband/last_sync': now,
        f'{hour}/count': increment(1),
        f'{hour}/stress_sum': increment(reading['stress']),
        f'{hour}/gsr_sum': increment(values['gsr']),
        f'{hour}/heart_rate_sum': increment(values['heart_rate']),
        f'{hour}/temperature_sum': increment(values['temperature']),
    })
    return jsonify({'success': True, 'config': wristband_device_config()})


def build_wristband_trend(hourly, now_ms, days):
    """Stress chart points from hourly totals: 24 hourly slots when days == 1, otherwise one slot
    per day. Positions are percentages of the plot box; empty slots break the line."""
    from datetime import datetime, timedelta
    now = datetime.fromtimestamp(now_ms / 1000)
    if days == 1:
        first_hour = now_ms // 3_600_000 - 23
        slots = [datetime.fromtimestamp((first_hour + i) * 3600) for i in range(24)]
        slot_of = lambda hour: hour - first_hour
        label = lambda t: t.strftime('%H:00')
        full = lambda t: t.strftime('%d %b, %H:00') + '–' + (t + timedelta(hours=1)).strftime('%H:00')
    else:
        first_day = now.date() - timedelta(days=days - 1)
        slots = [datetime.combine(first_day + timedelta(days=i), datetime.min.time()) for i in range(days)]
        slot_of = lambda hour: (datetime.fromtimestamp(hour * 3600).date() - first_day).days
        label = lambda t: t.strftime('%a %d')
        full = lambda t: t.strftime('%a %d %b %Y')

    totals = [None] * len(slots)
    for key, bucket in (hourly or {}).items():
        if not (isinstance(bucket, dict) and key[1:].isdigit() and (bucket.get('count') or 0) > 0):
            continue
        i = slot_of(int(key[1:]))
        if 0 <= i < len(slots):
            t = totals[i] or {'count': 0, 'stress_sum': 0, 'gsr_sum': 0, 'heart_rate_sum': 0, 'temperature_sum': 0}
            for field in t:
                t[field] += bucket.get(field) or 0
            totals[i] = t

    n = len(slots)
    points, segments, run = [], [], []
    for i, t in enumerate(totals):
        x = round(i * 100 / (n - 1), 2)
        if not t:
            if run:
                segments.append(' '.join(run))
            run = []
            continue
        score = round(t['stress_sum'] / t['count'])
        level, band_label, color = stress_band(score)
        run.append(f'{x},{100 - score}')
        count = t['count']
        points.append({'x': x, 'score': score, 'level': level, 'label': band_label, 'color': color,
                       'date': full(slots[i]),
                       'detail': f"HR {t['heart_rate_sum'] / count:.0f} bpm · GSR {t['gsr_sum'] / count:.0f}"
                                 f" · {t['temperature_sum'] / count:.1f} °C · {count} reading{'s' if count != 1 else ''}"})
    if run:
        segments.append(' '.join(run))
    step = 6 if days == 1 else 1
    labels = [{'x': round(i * 100 / (n - 1), 2), 'text': label(slots[i])}
              for i in range(n) if i % step == 0 or i == n - 1]
    # A lone point (no neighbour on either side) has nothing to connect to
    return {'points': points, 'segments': [s for s in segments if ' ' in s], 'labels': labels}


def load_wristband_view(uid, days):
    """Template data for _wristband_panel.html, or None when no wristband is paired."""
    import time
    wristband = db.reference(f'users/{uid}/wristband').get()
    if not isinstance(wristband, dict) or not wristband.get('device_id'):
        return None
    now = int(time.time() * 1000)
    last_sync = wristband.get('last_sync')
    age = now - last_sync if isinstance(last_sync, (int, float)) else None
    if age is not None and age <= WRISTBAND_ACTIVE_MS:
        status = ('active', 'Active')
    elif age is not None and age <= WRISTBAND_IDLE_MS:
        status = ('idle', 'Idle')
    else:
        status = ('offline', 'Offline')

    latest = wristband.get('latest')
    if isinstance(latest, dict) and isinstance(latest.get('stress'), (int, float)):
        latest = dict(latest, stress_level=stress_band(latest['stress']))
    else:
        latest = None
    first_hour = now // 3_600_000 - 24 * days + 1
    hourly = db.reference(f'users/{uid}/wristband_hourly').order_by_key().start_at(f'h{first_hour}').get()
    return {
        'device_id': wristband['device_id'],
        'paired_at': format_timestamp(wristband.get('paired_at'), '%d %b %Y, %I:%M %p'),
        'status': status[0], 'status_label': status[1],
        'last_sync_ago': time_ago(last_sync) if age is not None else None,
        'last_sync_full': format_timestamp(last_sync, '%d %b %Y, %I:%M:%S %p') if age is not None else None,
        'latest': latest,
        'trend': build_wristband_trend(hourly, now, days),
    }


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

    assigned_doctor = load_doctor_summary(user.get('assigned_doctor_uid'))
    if assigned_doctor:
        assigned_doctor['assigned_since'] = format_timestamp(user.get('assigned_at'), '%d %b %Y')

    return render_template('profile.html',
                           user_name=user_name, user_initials=user_initials,
                           email=session['user'].get('email') or user.get('email', ''),
                           phone=user.get('phone', ''),
                           profile=user.get('profile') or {},
                           member_since=format_timestamp(user.get('created_at'), '%d %b %Y'),
                           gender_options=GENDER_OPTIONS, assigned_doctor=assigned_doctor)


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


# ========== WELLNESS GAMES (patient) ==========
# users/{uid}/game_stats/{game}: plays, last_played, best_score (or best_<level>), total_seconds
# users/{uid}/gratitude/{push_id}: text, created_at (private: doctors never see it; masked in the admin browser)
# 'better' says whether a lower or higher score is the best one; None = no score (relaxation only).
GAMES = {
    'breathing': {'title': 'Breathing Exercise', 'icon': '🫁', 'color': '#D1FAE5', 'better': None,
                  'blurb': 'Follow the circle through 4-7-8 breathing to settle your body.'},
    'memory': {'title': 'Memory Match', 'icon': '🧠', 'color': '#DBEAFE', 'better': 'low',
               'blurb': 'Flip cards and find matching pairs to sharpen focus.'},
    'color': {'title': 'Color Focus', 'icon': '🎨', 'color': '#EDE9FE', 'better': 'high',
              'blurb': 'Tap the colour the word names, not the colour it\'s written in.'},
    'zen': {'title': 'Zen Garden', 'icon': '🪨', 'color': '#FEF3C7', 'better': None,
            'blurb': 'Rake sand and place stones in a peaceful garden.'},
    'bubbles': {'title': 'Bubble Pop', 'icon': '🫧', 'color': '#E0F2FE', 'better': 'high',
                'blurb': 'Pop slowly rising bubbles: simple, soothing, no pressure.'},
    'mandala': {'title': 'Mandala Draw', 'icon': '🌸', 'color': '#FCE7F3', 'better': None,
                'blurb': 'Draw and watch your strokes mirror into a mandala.'},
    'gratitude': {'title': 'Gratitude Jar', 'icon': '🫙', 'color': '#FFEDD5', 'better': None,
                  'blurb': 'Drop a note about something good into your jar.'},
}
GRATITUDE_MAX = 280


def record_game_play(uid, game, duration=None, score=None, level=None):
    """Updates users/{uid}/game_stats/{game} and returns (stats, is_new_best)."""
    ref = db.reference(f'users/{uid}/game_stats/{game}')
    current = ref.get() or {}
    current = current if isinstance(current, dict) else {}
    updates = {'plays': {'.sv': {'increment': 1}}, 'last_played': {'.sv': 'timestamp'}}
    if isinstance(duration, (int, float)):
        updates['total_seconds'] = {'.sv': {'increment': int(duration)}}
    new_best = False
    better = GAMES[game]['better']
    if better and isinstance(score, (int, float)):
        key = f'best_{level}' if level else 'best_score'
        old = current.get(key)
        if not isinstance(old, (int, float)) or (score < old if better == 'low' else score > old):
            updates[key] = score
            new_best = old is not None
    ref.update(updates)
    return ref.get() or {}, new_best


def gratitude_stats(entries):
    """Count, current day streak (ending today or yesterday) and an encouraging message."""
    from datetime import date, timedelta
    days = {date.fromtimestamp(e['created_at'] / 1000) for e in entries if isinstance(e.get('created_at'), (int, float))}
    streak, day = 0, date.today()
    if day not in days:
        day -= timedelta(days=1)    # a streak still counts until the end of today
    while day in days:
        streak += 1
        day -= timedelta(days=1)
    n = len(entries)
    milestones = {1: 'Your first gratitude note. A lovely start.', 10: '10 gratitudes!', 25: '25 gratitudes. Look how full your jar is getting!',
                  50: '50 gratitudes!', 100: '100 gratitudes! That\'s a lot of good things.'}
    message = milestones.get(n) or (f'{streak} days streak!' if streak >= 2 else
                                    'Every small good thing counts.' if n else 'Your jar is waiting for its first note.')
    return {'count': n, 'streak': streak, 'message': message}


def load_gratitude(uid, limit=500):
    raw = db.reference(f'users/{uid}/gratitude').order_by_key().limit_to_last(limit).get() or {}
    entries = [{'id': k, 'text': v.get('text', ''), 'created_at': v.get('created_at'),
                'date': format_timestamp(v.get('created_at'), '%d %b %Y, %I:%M %p')}
               for k, v in raw.items() if isinstance(v, dict) and isinstance(v.get('text'), str)]
    entries.sort(key=lambda e: e['created_at'] or 0)
    return entries


@app.route('/patient/games/<game>')
@login_required
@role_required('patient')
def patient_game(game):
    if game not in GAMES:
        return redirect(url_for('patient_activities'))
    stats = db.reference(f"users/{session['user']['uid']}/game_stats/{game}").get() or {}
    return render_template(f'games/{game}.html', game=game, meta=GAMES[game], stats=stats if isinstance(stats, dict) else {})


@app.route('/api/games/complete', methods=['POST'])
@login_required
@role_required('patient')
def game_complete():
    if not request.is_json:   # JSON only, so a cross-site form can't post it
        return chat_json_error('Expected a JSON request.', 415)
    data = request.get_json(silent=True) or {}
    game = data.get('game')
    if game not in GAMES or game == 'gratitude':   # gratitude plays are recorded by /api/gratitude/add
        return chat_json_error('Unknown game.', 400)
    def number(value, hi):
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= hi else None
    duration = number(data.get('duration'), 86_400)
    score = number(data.get('score'), 100_000)
    level = data.get('level') if data.get('level') in ('easy', 'medium', 'hard') else None
    stats, new_best = record_game_play(session['user']['uid'], game, duration, score, level)
    activity_log('game_completed', f"Completed {GAMES[game]['title']}",
                 details={'game': game, 'duration_s': int(duration) if duration is not None else None,
                          'score': score, 'level': level})
    return jsonify({'success': True, 'plays': stats.get('plays', 1), 'new_best': new_best,
                    'best': stats.get(f'best_{level}' if level else 'best_score')})


@app.route('/api/gratitude/add', methods=['POST'])
@login_required
@role_required('patient')
def gratitude_add():
    if not request.is_json:
        return chat_json_error('Expected a JSON request.', 415)
    text = str((request.get_json(silent=True) or {}).get('text') or '').strip()
    if not text:
        return chat_json_error('Write something you\'re grateful for first.', 400)
    if len(text) > GRATITUDE_MAX:
        return chat_json_error(f'Keep it to {GRATITUDE_MAX} characters or fewer.', 400)
    uid = session['user']['uid']
    ref = db.reference(f'users/{uid}/gratitude').push({'text': text, 'created_at': {'.sv': 'timestamp'}})
    saved = ref.get() or {}
    record_game_play(uid, 'gratitude')
    # The note itself stays private: only that one was added is logged
    activity_log('game_completed', 'Added a note to the Gratitude Jar', details={'game': 'gratitude'})
    entries = load_gratitude(uid)
    return jsonify({'success': True, 'entry': {'id': ref.key, 'text': text, 'created_at': saved.get('created_at'),
                                               'date': format_timestamp(saved.get('created_at'), '%d %b %Y, %I:%M %p')},
                    'stats': gratitude_stats(entries)})


@app.route('/api/gratitude/list')
@login_required
@role_required('patient')
def gratitude_list():
    entries = load_gratitude(session['user']['uid'])
    return jsonify({'success': True, 'entries': entries, 'stats': gratitude_stats(entries)})


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
        'date': format_timestamp(entry.get('created_at'), '%d %b %Y, %I:%M %p'),
        'share_with_doctor': entry.get('share_with_doctor') is True
    } for _, entry in sorted(recent.items(), reverse=True)]

    game_stats = db.reference(f'users/{uid}/game_stats').get() or {}
    game_stats = game_stats if isinstance(game_stats, dict) else {}
    games = [{'key': key, **meta, 'plays': (game_stats.get(key) or {}).get('plays', 0),
              'last_played': time_ago((game_stats.get(key) or {}).get('last_played')) if (game_stats.get(key) or {}).get('last_played') else ''}
             for key, meta in GAMES.items()]
    return render_template('activities.html', user_name=user_name, user_initials=user_initials,
                           journal=journal, moods=JOURNAL_MOODS, journal_max=JOURNAL_MAX_LENGTH, games=games)


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
        # Private unless the patient explicitly opts in; only a JSON true counts as sharing
        share = data.get('share_with_doctor') is True

        if not text:
            return jsonify({'success': False, 'error': 'Please write something before saving.'}), 400
        if len(text) > JOURNAL_MAX_LENGTH:
            return jsonify({'success': False, 'error': f'Entries are limited to {JOURNAL_MAX_LENGTH} characters.'}), 400
        if mood and mood not in JOURNAL_MOODS:
            return jsonify({'success': False, 'error': 'Invalid mood option.'}), 400

        now_ms = int(time.time() * 1000)
        entry = {'text': text, 'created_at': {'.sv': 'timestamp'}, 'share_with_doctor': share}
        if mood:
            entry['mood'] = mood
        db.reference(f'users/{uid}/journal/{now_ms}').set(entry)

        return jsonify({'success': True,
                        'message': 'Journal entry saved and shared with your doctor.' if share else 'Journal entry saved privately.',
                        'entry': {'mood': mood, 'text': text, 'share_with_doctor': share,
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

        data = data if isinstance(data, dict) else {}
        fields = {k: str(data.get(k) or '').strip() for k in ('fullname', 'phone', 'address', 'city', 'notes')}
        fields['size'] = str(data.get('size') or 'M').strip().upper()
        if (not fields['fullname'] or len(fields['fullname']) > 100 or not fields['phone'] or len(fields['phone']) > 30
                or not fields['address'] or len(fields['address']) > 300 or not fields['city'] or len(fields['city']) > 100
                or fields['size'] not in ('S', 'M', 'L') or len(fields['notes']) > 500):
            return jsonify({'success': False, 'error': 'Please fill in your name, phone, address and city, and choose a wrist size.'}), 400

        # Price, reference and payment are set here; anything the browser sends for them is ignored
        order = new_order_record(uid, fields)
        db.reference(f'wristband_orders/{uid}').set(order)
        activity_log('wristband_ordered', f"Ordered a wristband ({format_pkr(order['price'])}, cash on delivery)",
                     details={'ref': order['ref'], 'price': order['price'], 'city': fields['city']})

        return jsonify({'success': True, 'message': 'Order placed successfully', 'redirect': url_for('patient_wristband'),
                        'order': {'ref': order['ref'], 'product': WRISTBAND_PRODUCT, 'total': format_pkr(order['price']),
                                  'fullname': fields['fullname'], 'phone': fields['phone'], 'address': fields['address'],
                                  'city': fields['city'], 'size': fields['size'], 'payment': WRISTBAND_PAYMENT,
                                  'delivery': WRISTBAND_DELIVERY}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

RISK_LEVELS = ['low', 'moderate', 'high', 'critical']


def doctor_identity():
    # Avoid "Dr. Dr. ..." when the stored name already carries the title
    user_name = strip_dr(session['user'].get('name', 'Doctor')) or 'Doctor'
    user_initials = ''.join([word[0].upper() for word in user_name.split()[:2]]) or 'D'
    return user_name, user_initials


def load_active_patients(doctor_uid):
    """Summaries of the active patients assigned to doctor_uid (scored on their latest
    assessment) plus every timestamped assessment across them, from one read of users."""
    users = db.reference('users').get() or {}

    patients, recent = [], []
    for uid, u in users.items():
        if not isinstance(u, dict) or u.get('role') != 'patient' or u.get('status') != 'active':
            continue
        if not patient_assigned_to(u, doctor_uid):
            continue
        name = u.get('name') or 'Unnamed patient'
        assessments = u.get('assessments') or {}

        latest = assessments.get(u.get('latest_assessment_id')) if u.get('latest_assessment_id') else None
        patient = {'uid': uid, 'name': name, 'email': u.get('email', ''),
                   'composite_score': None, 'risk_level': None, 'last_assessment': None, 'facial_dominant': None}
        if latest:
            score = compute_composite(latest)
            risk_level, risk_label, risk_color = risk_band(score)
            patient.update(composite_score=score, risk_level=risk_level, risk_label=risk_label, risk_color=risk_color,
                           last_assessment=format_timestamp(latest.get('created_at'), '%d %b %Y'),
                           facial_dominant=facial_mood(latest))
        patients.append(patient)

        for a in assessments.values():
            if isinstance(a, dict) and isinstance(a.get('created_at'), (int, float)):
                score = compute_composite(a)
                risk_level, risk_label, risk_color = risk_band(score)
                recent.append({'patient_uid': uid, 'name': name, 'composite_score': score, 'risk_level': risk_level,
                               'risk_label': risk_label, 'risk_color': risk_color, 'created_at': a['created_at'],
                               'phq_score': a.get('phq_score'), 'gad_score': a.get('gad_score'),
                               'pss_score': a.get('pss_score'), 'facial_mood': facial_mood(a)})

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
    patients, _ = load_active_patients(session['user']['uid'])
    counts = risk_distribution(patients)

    risk = request.args.get('risk', 'all').lower()
    if risk not in RISK_LEVELS:
        risk = 'all'
    query = request.args.get('q', '').strip()

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
                           risk=risk, query=query, risk_levels=RISK_LEVELS)


def get_active_patient(patient_uid):
    """The users/{uid} record if it is an active patient, else None."""
    import re
    # Firebase Auth UIDs are URL-safe; reject anything that could alter the database path
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', patient_uid or ''):
        return None
    user = db.reference(f'users/{patient_uid}').get()
    if not isinstance(user, dict) or user.get('role') != 'patient' or user.get('status') != 'active':
        return None
    return user


@app.route('/doctor/patients/<patient_uid>')
@login_required
@role_required('doctor')
def doctor_patient_detail(patient_uid):
    user_name, user_initials = doctor_identity()
    patient = get_active_patient(patient_uid)
    if not patient:
        flash('That patient could not be found or is no longer active.', 'error')
        return redirect(url_for('doctor_patients'))
    if not patient_assigned_to(patient, session['user']['uid']):
        flash("That patient isn't assigned to you.", 'error')
        return redirect(url_for('doctor_patients'))

    name = patient.get('name') or 'Unnamed patient'
    assessments = {k: v for k, v in (patient.get('assessments') or {}).items() if isinstance(v, dict)}
    by_newest = sorted(assessments.items(),
                       key=lambda kv: kv[1].get('created_at') if isinstance(kv[1].get('created_at'), (int, float)) else 0,
                       reverse=True)
    latest = assessments.get(patient.get('latest_assessment_id')) or (by_newest[0][1] if by_newest else None)

    # Privacy: the doctor view gets only name, age and clinical data. Contact details,
    # date of birth, address and emergency contact are never passed to the template.
    header = {'name': name,
              'initials': ''.join(w[0].upper() for w in name.split()[:2]) or 'P',
              'age': age_from_dob((patient.get('profile') or {}).get('dob')),
              'composite_score': None}
    current = None
    if latest:
        score = compute_composite(latest)
        risk_level, risk_label, risk_color = risk_band(score)
        header.update(composite_score=score, risk_level=risk_level, risk_label=risk_label, risk_color=risk_color)
        current = {
            'date': format_timestamp(latest.get('created_at'), '%d %b %Y, %I:%M %p'),
            'phq': (latest.get('phq_score', 0),) + phq_severity(latest.get('phq_score', 0)),
            'gad': (latest.get('gad_score', 0),) + gad_severity(latest.get('gad_score', 0)),
            'pss': (latest.get('pss_score', 0),) + pss_severity(latest.get('pss_score', 0)),
            'facial_dominant': facial_mood(latest),
            'facial_skipped': bool(latest.get('facial_skipped')),
        }

    history = []
    for assessment_id, a in by_newest:
        score = compute_composite(a)
        _, risk_label, risk_color = risk_band(score)
        history.append({
            'id': assessment_id,
            'date': format_timestamp(a.get('created_at'), '%d %b %Y'),
            'time': format_timestamp(a.get('created_at'), '%I:%M %p'),
            'composite_score': score, 'risk_label': risk_label, 'risk_color': risk_color,
            'facial_skipped': bool(a.get('facial_skipped')),
            'phq': (a.get('phq_score', 0),) + phq_severity(a.get('phq_score', 0)),
            'gad': (a.get('gad_score', 0),) + gad_severity(a.get('gad_score', 0)),
            'pss': (a.get('pss_score', 0),) + pss_severity(a.get('pss_score', 0)),
            'facial_dominant': facial_mood(a, missing='-'),
            'viewable': assessment_id.isdigit(),
        })

    # Journal keys are millisecond timestamps, so key order is chronological
    # Only entries the patient explicitly shared; entries saved before sharing existed have no
    # flag and stay private. Journal keys are millisecond timestamps, so key order is chronological.
    shared = [e for _, e in sorted((patient.get('journal') or {}).items(), reverse=True)
              if isinstance(e, dict) and e.get('share_with_doctor') is True]
    journal = [{'mood': e.get('mood', ''), 'text': e.get('text', ''),
                'date': format_timestamp(e.get('created_at'), '%d %b %Y, %I:%M %p')} for e in shared[:5]]

    # Wristband: status only; the order's delivery address and phone stay with the patient
    order = db.reference(f'wristband_orders/{patient_uid}').get()
    wristband = None
    if isinstance(order, dict):
        status = (order.get('status') or 'pending').lower()
        wristband = {'status': status if status in WRISTBAND_STATUSES else 'pending',
                     'date': format_timestamp(order.get('created_at'), '%d %b %Y')}

    return render_template('doctor_patient_detail.html',
                           user_name=user_name, user_initials=user_initials,
                           patient_uid=patient_uid, patient=header, current=current,
                           risk_trend=build_risk_trend(
                               assessments, 10,
                               lambda aid: url_for('doctor_assessment_detail', patient_uid=patient_uid, assessment_id=aid)),
                           history=history, journal=journal, journal_total=len(shared),
                           wristband=wristband, wb=load_wristband_view(patient_uid, 7),
                           patient_alert_list=patient_alerts(patient_uid, patient),
                           clinical_notes=visible_clinical_notes(patient_uid, session['user']['uid']),
                           note_categories=NOTE_CATEGORIES, note_title_max=NOTE_TITLE_MAX, note_body_max=NOTE_BODY_MAX)


@app.route('/doctor/patients/<patient_uid>/assessments/<assessment_id>')
@login_required
@role_required('doctor')
def doctor_assessment_detail(patient_uid, assessment_id):
    patient = get_active_patient(patient_uid)
    if not patient:
        flash('That patient could not be found or is no longer active.', 'error')
        return redirect(url_for('doctor_patients'))
    if not patient_assigned_to(patient, session['user']['uid']):
        flash("That patient isn't assigned to you.", 'error')
        return redirect(url_for('doctor_patients'))
    assessment = (patient.get('assessments') or {}).get(assessment_id) if assessment_id.isdigit() else None
    if not isinstance(assessment, dict):
        flash('That assessment could not be found.', 'error')
        return redirect(url_for('doctor_patient_detail', patient_uid=patient_uid))

    return render_template('results.html', read_only=True, doctor_view=True,
                           patient_name=patient.get('name') or 'Patient',
                           back_url=url_for('doctor_patient_detail', patient_uid=patient_uid),
                           assessment_date=format_timestamp(assessment.get('created_at'), '%d %b %Y, %I:%M %p'),
                           **build_results_context(assessment))


NOTE_CATEGORIES = {
    'observation': 'Observation', 'diagnosis': 'Diagnosis', 'treatment_plan': 'Treatment Plan',
    'follow_up': 'Follow-up', 'medication': 'Medication', 'other': 'Other',
}
NOTE_VISIBILITIES = ('all_doctors', 'private')
NOTE_TITLE_MAX, NOTE_BODY_MAX = 100, 2000


def is_safe_key(value):
    """Firebase UIDs and push IDs are URL-safe; anything else could alter a database path."""
    import re
    return bool(re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value or ''))


# ========== RATE LIMIT RESPONSES ==========
_rate_limit_logged = {}   # (ip, endpoint) -> last time logged, so a flood can't flood the log too


@app.errorhandler(429)
def too_many_requests(error):
    now = time.time()
    key = (request.remote_addr, request.endpoint)
    if now - _rate_limit_logged.get(key, 0) > 60:
        if len(_rate_limit_logged) > 5000:
            _rate_limit_logged.clear()
        _rate_limit_logged[key] = now
        actor = session.get('user') or {'uid': 'anonymous', 'name': 'Unknown', 'role': 'guest'}
        activity_log('rate_limit_exceeded', f'Rate limit exceeded on {request.path}', 'alert',
                     details={'path': request.path, 'method': request.method, 'limit': str(error.description)}, user=actor)
    message = 'Too many requests. Please wait a little and try again.'
    if request.path.startswith('/api/') or request.is_json or request.accept_mimetypes.best == 'application/json':
        return jsonify({'success': False, 'error': message, 'rate_limited': True}), 429
    page = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Too many requests</title><body style="font-family:system-ui,sans-serif;max-width:520px;margin:15vh auto;'
            'padding:0 16px;color:#0F172A"><h1 style="font-size:22px">Too many requests</h1>'
            f'<p style="color:#475569">{message}</p><p><a href="/" style="color:#0F7A5F">Back to the home page</a></p></body>')
    return page, 429


# ========== PRESENCE (admin "Active Sessions Now") ==========
# presence/{uid} = last request time (server ms), written at most once a minute per session.
# Kept out of users/{uid} so a stale session for a deleted account can't recreate that node.
PRESENCE_WRITE_EVERY = 60          # seconds between writes for one session
ACTIVE_SESSION_MS = 5 * 60 * 1000  # "active now" window


@app.before_request
def track_presence():
    user = session.get('user')
    if not user or request.endpoint in (None, 'static'):
        return
    now = time.time()
    if now - session.get('_presence_at', 0) < PRESENCE_WRITE_EVERY or not is_safe_key(user.get('uid')):
        return
    try:
        db.reference(f"presence/{user['uid']}").set({'.sv': 'timestamp'})
        session['_presence_at'] = now
    except Exception:
        pass   # presence is best-effort; never block a page on it


# ========== ACTIVITY LOG (admin audit trail) ==========
# activity_logs/{push_id}: timestamp, user_uid, user_name, user_role, action, action_label, details,
#   ip, user_agent_summary, status. Only state changes and security events are logged, never page
#   views, and never passwords, API keys or tokens. Entries are kept indefinitely for now.
LOG_STATUSES = ('success', 'failed', 'alert', 'info')
LOG_ROLES = ('patient', 'doctor', 'admin', 'auto', 'guest')   # guest: not identified (e.g. failed sign-in)


def user_agent_summary(ua):
    """'Chrome, Windows' style summary from a User-Agent string (rough, regex-only)."""
    import re
    ua = ua or ''
    if not ua:
        return 'Unknown'
    if re.search(r'python-requests|python-urllib|curl|wget|httpie|postman|werkzeug', ua, re.I):
        return 'Script / API client'
    browser = next((name for pattern, name in (
        (r'Edg(e|A|iOS)?/', 'Edge'), (r'OPR/|Opera', 'Opera'), (r'SamsungBrowser/', 'Samsung Internet'),
        (r'Firefox/|FxiOS/', 'Firefox'), (r'Chrome/|CriOS/', 'Chrome'), (r'Version/[\d.]+.*Safari/', 'Safari'))
        if re.search(pattern, ua)), 'Other browser')
    system = next((name for pattern, name in (
        (r'iPhone|iPad|iPod', 'iOS'), (r'Android', 'Android'), (r'Windows', 'Windows'),
        (r'Mac OS X|Macintosh', 'macOS'), (r'CrOS', 'ChromeOS'), (r'Linux', 'Linux'))
        if re.search(pattern, ua)), 'Unknown OS')
    return f'{browser}, {system}'


def activity_log(action, label, status='success', details=None, user=None):
    """Appends one audit entry. `user` defaults to the signed-in user; pass
    {'uid': 'system', 'name': 'System', 'role': 'auto'} for automatic events. Never raises."""
    try:
        actor = user or (session.get('user') if has_request_context() else None) or {}
        clean = {}
        for key, value in (details or {}).items():
            if value is None or value == '':
                continue
            clean[str(key)[:40]] = value if isinstance(value, (int, float, bool)) else str(value)[:300]
        has_request = has_request_context()
        db.reference('activity_logs').push({
            'timestamp': {'.sv': 'timestamp'},
            'user_uid': str(actor.get('uid') or 'anonymous'),
            'user_name': str(actor.get('name') or 'Unknown user')[:100],
            'user_role': actor.get('role') if actor.get('role') in LOG_ROLES else 'auto',
            'action': action, 'action_label': str(label)[:200],
            'details': clean or None,
            'ip': (request.remote_addr or '') if has_request else '',
            'user_agent_summary': user_agent_summary(request.headers.get('User-Agent', '')) if has_request else 'Server',
            'status': status if status in LOG_STATUSES else 'info',
        })
    except Exception:
        pass   # auditing must never break the action being audited


SYSTEM_ACTOR = {'uid': 'system', 'name': 'System', 'role': 'auto'}


def validate_note(data):
    """(clean_fields, errors) for a clinical note request body."""
    title = str(data.get('title') or '').strip()
    body = str(data.get('body') or '').strip()
    category = str(data.get('category') or '').strip()
    visibility = str(data.get('visibility') or '').strip()

    errors = {}
    if not body:
        errors['body'] = 'Note text is required.'
    elif len(body) > NOTE_BODY_MAX:
        errors['body'] = f'Note text must be {NOTE_BODY_MAX} characters or fewer.'
    if len(title) > NOTE_TITLE_MAX:
        errors['title'] = f'Title must be {NOTE_TITLE_MAX} characters or fewer.'
    if category not in NOTE_CATEGORIES:
        errors['category'] = 'Choose a valid category.'
    if visibility not in NOTE_VISIBILITIES:
        errors['visibility'] = 'Visibility must be all_doctors or private.'
    # An empty title is stored as None so a multi-path update removes it
    return {'title': title or None, 'body': body, 'category': category, 'visibility': visibility}, errors


def note_for_display(note_id, note, viewer_uid):
    created = note.get('created_at') if isinstance(note.get('created_at'), (int, float)) else 0
    updated = note.get('updated_at') if isinstance(note.get('updated_at'), (int, float)) else created
    category = note.get('category') if note.get('category') in NOTE_CATEGORIES else 'other'
    return {
        'id': note_id, 'title': note.get('title') or '', 'body': note.get('body') or '',
        'category': category, 'category_label': NOTE_CATEGORIES[category],
        'visibility': note.get('visibility', 'all_doctors'),
        'is_private': note.get('visibility') == 'private',
        'is_mine': note.get('doctor_uid') == viewer_uid,
        'doctor_name': note.get('doctor_name') or 'Unknown doctor',
        'created_at': created, 'time_ago': time_ago(created) if created else '-',
        'created_full': format_timestamp(created, '%d %b %Y, %I:%M %p') if created else '-',
        'edited': bool(updated and created and updated - created > 60000),
    }


def clinical_notes_ref(patient_uid, note_id=None):
    """Clinical notes live at top-level clinical_notes/{patient_uid}/{note_id}, outside the
    patient's own users/{uid} record, so they are only reached through this server
    (Admin SDK) and never through a patient's read access to their own user node."""
    return db.reference(f'clinical_notes/{patient_uid}' + (f'/{note_id}' if note_id else ''))


def visible_clinical_notes(patient_uid, viewer_uid):
    """Notes this doctor may see on a patient: everyone's shared notes plus their own private
    ones. Other doctors' private notes are dropped here, before anything reaches a template."""
    notes = []
    for note_id, n in (clinical_notes_ref(patient_uid).get() or {}).items():
        if not isinstance(n, dict):
            continue
        if n.get('visibility') == 'private' and n.get('doctor_uid') != viewer_uid:
            continue
        notes.append(note_for_display(note_id, n, viewer_uid))
    return sorted(notes, key=lambda n: n['created_at'], reverse=True)


def note_json_error(message, status, errors=None):
    body = {'success': False, 'error': message}
    if errors:
        body['errors'] = errors
    return jsonify(body), status


@app.route('/api/patients/<patient_uid>/clinical-notes', methods=['POST'])
@login_required
@role_required('doctor')
def add_clinical_note(patient_uid):
    patient = get_active_patient(patient_uid)
    if not patient:
        return note_json_error('Patient not found or no longer active.', 404)
    if not patient_assigned_to(patient, session['user']['uid']):
        return note_json_error("That patient isn't assigned to you.", 403)
    fields, errors = validate_note(request.get_json(silent=True) or {})
    if errors:
        return note_json_error('Please fix the highlighted fields.', 400, errors)

    doctor_name, _ = doctor_identity()
    note = {k: v for k, v in fields.items() if v is not None}
    note.update(doctor_uid=session['user']['uid'], doctor_name=doctor_name,
                created_at={'.sv': 'timestamp'}, updated_at={'.sv': 'timestamp'})
    note_id = clinical_notes_ref(patient_uid).push(note).key

    flash('Clinical note added.', 'success')
    return jsonify({'success': True, 'note_id': note_id,
                    'redirect': url_for('doctor_patient_detail', patient_uid=patient_uid) + f'#note-{note_id}'})


def load_own_note(patient_uid, note_id):
    """(reference, note, error_response) for a note the current doctor wrote."""
    if not is_safe_key(patient_uid) or not is_safe_key(note_id):
        return None, None, note_json_error('Note not found.', 404)
    ref = clinical_notes_ref(patient_uid, note_id)
    note = ref.get()
    if not isinstance(note, dict):
        return None, None, note_json_error('Note not found.', 404)
    if note.get('doctor_uid') != session['user']['uid']:
        return None, None, note_json_error('You can only change notes you wrote.', 403)
    if db.reference(f'users/{patient_uid}/assigned_doctor_uid').get() != session['user']['uid']:
        return None, None, note_json_error("That patient isn't assigned to you.", 403)
    return ref, note, None


@app.route('/api/patients/<patient_uid>/clinical-notes/<note_id>', methods=['PUT'])
@login_required
@role_required('doctor')
def edit_clinical_note(patient_uid, note_id):
    ref, _, error = load_own_note(patient_uid, note_id)
    if error:
        return error
    fields, errors = validate_note(request.get_json(silent=True) or {})
    if errors:
        return note_json_error('Please fix the highlighted fields.', 400, errors)

    # doctor_uid, doctor_name and created_at are never changed
    ref.update(dict(fields, updated_at={'.sv': 'timestamp'}))
    flash('Clinical note updated.', 'success')
    return jsonify({'success': True, 'note_id': note_id,
                    'redirect': url_for('doctor_patient_detail', patient_uid=patient_uid) + f'#note-{note_id}'})


@app.route('/api/patients/<patient_uid>/clinical-notes/<note_id>', methods=['DELETE'])
@login_required
@role_required('doctor')
def delete_clinical_note(patient_uid, note_id):
    ref, _, error = load_own_note(patient_uid, note_id)
    if error:
        return error
    ref.delete()
    flash('Clinical note deleted.', 'success')
    return jsonify({'success': True,
                    'redirect': url_for('doctor_patient_detail', patient_uid=patient_uid) + '#clinical-notes'})


@app.route('/doctor/notes')
@login_required
@role_required('doctor')
def doctor_notes():
    user_name, user_initials = doctor_identity()
    me = session['user']['uid']
    all_notes = db.reference('clinical_notes').get() or {}

    query = request.args.get('q', '').strip()
    category = request.args.get('category', 'all')
    if category not in NOTE_CATEGORIES:
        category = 'all'

    groups, total = [], 0
    for uid, patient_notes in all_notes.items():
        if not isinstance(patient_notes, dict) or not is_safe_key(uid):
            continue
        mine = [note_for_display(nid, n, me) for nid, n in patient_notes.items()
                if isinstance(n, dict) and n.get('doctor_uid') == me]
        if not mine:
            continue
        # Only the fields this page shows, not the patient's whole record
        if db.reference(f'users/{uid}/role').get() != 'patient':
            continue
        # A patient reassigned to another doctor takes these notes out of this doctor's view
        if db.reference(f'users/{uid}/assigned_doctor_uid').get() != me:
            continue
        total += len(mine)
        name = db.reference(f'users/{uid}/name').get() or 'Unnamed patient'
        status = db.reference(f'users/{uid}/status').get()
        if category != 'all':
            mine = [n for n in mine if n['category'] == category]
        if query:
            q = query.lower()
            if q not in name.lower():
                mine = [n for n in mine if q in n['title'].lower() or q in n['body'].lower()]
        if mine:
            mine.sort(key=lambda n: n['created_at'], reverse=True)
            groups.append({'uid': uid, 'name': name, 'active': status == 'active', 'notes': mine})
    # Patients with the most recent note first
    groups.sort(key=lambda g: g['notes'][0]['created_at'], reverse=True)

    return render_template('doctor_notes.html', user_name=user_name, user_initials=user_initials,
                           groups=groups, total_notes=total, shown_notes=sum(len(g['notes']) for g in groups),
                           query=query, category=category, categories=NOTE_CATEGORIES)


DOCTOR_PROFILE_FIELDS = ['specialization', 'license_number', 'affiliation', 'years_experience', 'bio']
DOCTOR_EDITABLE_FIELDS = ['name', 'phone'] + DOCTOR_PROFILE_FIELDS
SPECIALIZATION_SUGGESTIONS = ['Clinical Psychology', 'Psychiatry', 'Child & Adolescent Psychiatry',
                              'Counselling Psychology', 'Neuropsychology', 'General Practice']


def validate_doctor_field(field, raw):
    """(clean_value, error) for one editable field; '' / None means clear the field."""
    import re
    value = '' if raw is None else str(raw).strip()
    if field == 'name':
        if not value:
            return None, 'Full name is required.'
        return (value, None) if len(value) <= 100 else (None, 'Name must be 100 characters or fewer.')
    if not value:
        return None, None  # optional field cleared
    if field == 'phone':
        # Accept common formatting ("+92 300 1234567", "0300-1234567") and store digits only
        digits = re.sub(r'[\s\-+]', '', value)
        return (digits, None) if re.fullmatch(r'\d{10,15}', digits) else (None, 'Phone must contain 10-15 digits.')
    if field == 'license_number':
        # PMDC formats use separators, e.g. "PMDC-12345" or "PMDC/RMP/12345"
        if len(value) > 50 or not re.fullmatch(r'[A-Za-z0-9/-]+', value) or not re.search(r'[A-Za-z0-9]', value):
            return None, 'License number can use letters, numbers, hyphens and slashes, up to 50 characters.'
        return value, None
    if field == 'years_experience':
        if not re.fullmatch(r'\d{1,2}', value) or int(value) > 70:
            return None, 'Years of experience must be a whole number from 0 to 70.'
        return int(value), None
    if field == 'bio':
        return (value, None) if len(value) <= 500 else (None, 'Bio must be 500 characters or fewer.')
    if field in ('specialization', 'affiliation'):
        return (value, None) if len(value) <= 100 else (None, 'Must be 100 characters or fewer.')
    return None, 'Unknown field.'


@app.route('/doctor/profile')
@login_required
@role_required('doctor')
def doctor_profile():
    user_name, user_initials = doctor_identity()
    uid = session['user']['uid']
    user = db.reference(f'users/{uid}').get() or {}
    profile = user.get('profile') or {}

    values = {'name': user.get('name') or session['user'].get('name', ''), 'phone': user.get('phone', '')}
    values.update({f: profile.get(f, '') for f in DOCTOR_PROFILE_FIELDS})

    return render_template('doctor_profile.html',
                           user_name=user_name, user_initials=user_initials,
                           values=values,
                           email=session['user'].get('email') or user.get('email', ''),
                           member_since=format_timestamp(user.get('created_at'), '%d %b %Y'),
                           status=(user.get('status') or session['user'].get('status') or '').lower(),
                           specializations=SPECIALIZATION_SUGGESTIONS)


@app.route('/api/update-doctor-profile', methods=['POST'])
@login_required
@role_required('doctor')
def update_doctor_profile():
    try:
        data = request.get_json() or {}
        uid = session['user']['uid']

        # Partial update: only fields present in the request are validated and written
        fields = [f for f in DOCTOR_EDITABLE_FIELDS if f in data]
        unknown = [k for k in data if k not in DOCTOR_EDITABLE_FIELDS]
        if unknown:
            return jsonify({'success': False, 'error': 'Unknown field(s): ' + ', '.join(sorted(unknown))}), 400
        if not fields:
            return jsonify({'success': False, 'error': 'No changes to save.'}), 400

        clean, errors = {}, {}
        for f in fields:
            value, error = validate_doctor_field(f, data[f])
            if error:
                errors[f] = error
            else:
                clean[f] = value
        if errors:
            return jsonify({'success': False, 'error': 'Please fix the highlighted fields.', 'errors': errors}), 400

        # name/phone stay at the top level where registration and login read them; a None value
        # in a multi-path update deletes that field
        db.reference(f'users/{uid}').update(
            {(f if f in ('name', 'phone') else f'profile/{f}'): v for f, v in clean.items()})

        if 'name' in clean:
            session['user']['name'] = clean['name']
            session.modified = True
        user_name, user_initials = doctor_identity()
        return jsonify({'success': True, 'message': 'Profile updated successfully',
                        'saved': {f: '' if v is None else v for f, v in clean.items()},
                        'name': user_name, 'initials': user_initials})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/doctor/dashboard')
@login_required
@role_required('doctor')
def doctor_dashboard():
    user_name, user_initials = doctor_identity()
    patients, recent = load_active_patients(session['user']['uid'])

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
                           recent_assessments=recent,
                           login_just_happened=session.pop('_login_toast', False))


# ========== ASSESSMENT REPORTS (doctor) ==========
@app.route('/doctor/reports')
@login_required
@role_required('doctor')
def doctor_reports():
    import time
    user_name, user_initials = doctor_identity()
    patients, assessments = load_active_patients(session['user']['uid'])

    assessments.sort(key=lambda a: a['created_at'], reverse=True)
    for a in assessments:
        a['date'] = format_timestamp(a['created_at'], '%d %b %Y')
        a['date_iso'] = format_timestamp(a['created_at'], '%Y-%m-%d')   # what the From Date filter compares against

    week_ago = (time.time() - 7 * 86400) * 1000
    avg_score = (round(sum(a['composite_score'] for a in assessments) / len(assessments))
                 if assessments else None)
    stats = {
        'total': len(assessments),
        'this_week': sum(1 for a in assessments if a['created_at'] >= week_ago),
        'avg_score': avg_score,
        'avg_label': risk_band(avg_score)[1] if avg_score is not None else None,
        'critical_high': sum(1 for a in assessments if a['risk_level'] in ('critical', 'high')),
    }

    return render_template('doctor_reports.html', user_name=user_name, user_initials=user_initials,
                           assessments=assessments, stats=stats, risk_levels=RISK_LEVELS,
                           patients=sorted(patients, key=lambda p: p['name'].lower()))


# ========== RISK ALERTS (doctor) ==========
# Alerts are derived from assessments in the High or Critical band of risk_band() (default 51-70 and 71+;
# admins can move the thresholds on the ML Models page).
# Only the acknowledgement is stored, at alerts/{patient_uid}/{assessment_id}:
#   patient_uid, doctor_uid, assessment_id, severity, composite_score, created_at,
#   acknowledged_at (server time), acknowledged_by, acknowledged_by_name, note
# The URL id of an alert is "{patient_uid}_{assessment_id}" (assessment ids are digits).
ALERT_TABS = ('all', 'unread', 'critical', 'high', 'acknowledged')
ALERT_RANGES = ('all', 'today', 'week', 'month', 'custom')
ALERT_NOTE_MAX = 500


def alert_thresholds():
    """Lowest composite score that raises each alert severity."""
    t = risk_thresholds()
    return {'critical': t['high'] + 1, 'high': t['moderate'] + 1}


def alert_severity(score):
    level = risk_band(score)[0]
    return level if level in ('critical', 'high') else None


def parse_alert_id(alert_id):
    patient_uid, _, assessment_id = (alert_id or '').rpartition('_')
    if not patient_uid or not is_safe_key(patient_uid) or not assessment_id.isdigit():
        return None
    return patient_uid, assessment_id


def format_alert_time(ts):
    """'Just now', '12 min ago', 'Today 10:24 AM', 'Yesterday 3:05 PM', 'Oct 30, 2:15 PM'."""
    from datetime import datetime, timedelta
    if not isinstance(ts, (int, float)) or not ts:
        return ''
    when = datetime.fromtimestamp(ts / 1000)
    now = datetime.now()
    seconds = (now - when).total_seconds()
    clock = when.strftime('%I:%M %p').lstrip('0')
    if 0 <= seconds < 60:
        return 'Just now'
    if 0 <= seconds < 3600:
        return f'{int(seconds // 60)} min ago'
    if when.date() == now.date():
        return f'Today {clock}'
    if when.date() == (now - timedelta(days=1)).date():
        return f'Yesterday {clock}'
    day = when.strftime('%b %d').replace(' 0', ' ') + ('' if when.year == now.year else when.strftime(', %Y'))
    return f'{day}, {clock}'


def format_duration(ms):
    minutes = int(ms // 60000)
    if minutes < 1:
        return '< 1 min'
    if minutes < 60:
        return f'{minutes} min'
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f'{hours}h {minutes}m' if minutes else f'{hours}h'
    days, hours = divmod(hours, 24)
    return f'{days}d {hours}h' if hours else f'{days}d'


def patient_alerts(patient_uid, patient, acks=None):
    """Alerts for one patient, newest first, with acknowledgement state."""
    if acks is None:
        acks = db.reference(f'alerts/{patient_uid}').get() or {}
    name = patient.get('name') or 'Unnamed patient'
    alerts = []
    for assessment_id, a in (patient.get('assessments') or {}).items():
        if not isinstance(a, dict) or not isinstance(a.get('created_at'), (int, float)) or not str(assessment_id).isdigit():
            continue
        score = compute_composite(a)
        severity = alert_severity(score)
        if not severity:
            continue
        ack = acks.get(assessment_id) if isinstance(acks, dict) else None
        ack = ack if isinstance(ack, dict) and ack.get('acknowledged_at') else None
        alerts.append({
            'id': f'{patient_uid}_{assessment_id}',
            'patient_uid': patient_uid, 'patient_name': name,
            'initials': ''.join(w[0].upper() for w in name.split()[:2]) or 'P',
            'assessment_id': assessment_id,
            'severity': severity, 'severity_label': severity.capitalize(),
            'score': score, 'threshold': alert_thresholds()[severity],
            'created_at': a['created_at'], 'when': format_alert_time(a['created_at']),
            'full_date': format_timestamp(a['created_at'], '%d %b %Y, %I:%M %p'),
            'facial_skipped': bool(a.get('facial_skipped')),
            'acknowledged': bool(ack),
            'ack_at': ack.get('acknowledged_at') if ack else None,
            'ack_when': format_alert_time(ack.get('acknowledged_at')) if ack else '',
            'ack_by_name': (ack.get('acknowledged_by_name') or 'a doctor') if ack else '',
            'note': ack.get('note', '') if ack else '',
        })
    alerts.sort(key=lambda x: x['created_at'], reverse=True)
    return alerts


def load_doctor_alerts(doctor_uid):
    """Alerts for the doctor's currently assigned, active patients (reassignment removes them)."""
    users = db.reference('users').get() or {}
    alerts = []
    for uid, u in users.items():
        if isinstance(u, dict) and u.get('role') == 'patient' and u.get('status') == 'active' \
                and patient_assigned_to(u, doctor_uid) and u.get('assessments'):
            alerts.extend(patient_alerts(uid, u))
    alerts.sort(key=lambda x: x['created_at'], reverse=True)
    return alerts


def alert_unread_total(doctor_uid):
    return sum(1 for a in load_doctor_alerts(doctor_uid) if not a['acknowledged'])


def alerts_in_range(alerts, range_key, date_from, date_to):
    from datetime import datetime, timedelta
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end = None
    if range_key == 'today':
        start = today
    elif range_key == 'week':
        start = today - timedelta(days=today.weekday())        # since Monday
    elif range_key == 'month':
        start = today.replace(day=1)
    elif range_key == 'custom':
        try:
            start = datetime.strptime(date_from, '%Y-%m-%d') if date_from else None
            end = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1) if date_to else None
        except ValueError:
            start = end = None
    lo = start.timestamp() * 1000 if start else None
    hi = end.timestamp() * 1000 if end else None
    return [a for a in alerts if (lo is None or a['created_at'] >= lo) and (hi is None or a['created_at'] < hi)]


@app.route('/doctor/risk-alerts')
@login_required
@role_required('doctor')
def doctor_risk_alerts():
    from datetime import datetime
    doctor_uid = session['user']['uid']
    user_name, user_initials = doctor_identity()
    alerts = load_doctor_alerts(doctor_uid)

    # Stats cover every visible alert, whatever filter is applied
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000
    acknowledged = [a for a in alerts if a['acknowledged']]
    responses = [a['ack_at'] - a['created_at'] for a in acknowledged
                 if isinstance(a['ack_at'], (int, float)) and a['ack_at'] >= a['created_at']]
    stats = {
        'today': sum(1 for a in alerts if a['created_at'] >= midnight),
        'critical_open': sum(1 for a in alerts if a['severity'] == 'critical' and not a['acknowledged']),
        'acknowledged': len(acknowledged),
        'avg_response': format_duration(sum(responses) / len(responses)) if responses else None,
    }

    tab = request.args.get('filter', 'all').lower()
    tab = tab if tab in ALERT_TABS else 'all'
    query = request.args.get('q', '').strip()[:100]
    range_key = request.args.get('range', 'all').lower()
    range_key = range_key if range_key in ALERT_RANGES else 'all'
    date_from, date_to = request.args.get('from', ''), request.args.get('to', '')

    scoped = alerts_in_range(alerts, range_key, date_from, date_to)
    if query:
        scoped = [a for a in scoped if query.lower() in a['patient_name'].lower()]
    in_tab = {
        'all': lambda a: True,
        'unread': lambda a: not a['acknowledged'],
        'critical': lambda a: a['severity'] == 'critical',
        'high': lambda a: a['severity'] == 'high',
        'acknowledged': lambda a: a['acknowledged'],
    }
    counts = {t: sum(1 for a in scoped if in_tab[t](a)) for t in ALERT_TABS}
    shown = [a for a in scoped if in_tab[tab](a)]
    # Tab links keep the search and date filters
    filter_args = {k: v for k, v in (('q', query), ('range', range_key if range_key != 'all' else ''),
                                     ('from', date_from if range_key == 'custom' else ''),
                                     ('to', date_to if range_key == 'custom' else '')) if v}

    return render_template('doctor_risk_alerts.html', user_name=user_name, user_initials=user_initials,
                           alerts=shown, total_alerts=len(alerts), counts=counts, stats=stats,
                           tab=tab, query=query, range_key=range_key, date_from=date_from, date_to=date_to,
                           filter_args=filter_args, alert_unread_total=sum(1 for a in alerts if not a['acknowledged']))


def alert_for_doctor(alert_id):
    """(patient_uid, assessment_id, alert) when alert_id is a real alert on one of the signed-in
    doctor's patients, else a JSON error response."""
    parsed = parse_alert_id(alert_id)
    if not parsed:
        return None, chat_json_error('Alert not found.', 404)
    patient_uid, assessment_id = parsed
    patient = get_active_patient(patient_uid)
    if not patient or not patient_assigned_to(patient, session['user']['uid']):
        return None, chat_json_error("That patient isn't assigned to you.", 403)
    alert = next((a for a in patient_alerts(patient_uid, patient) if a['assessment_id'] == assessment_id), None)
    if not alert:
        return None, chat_json_error('Alert not found.', 404)
    return (patient_uid, assessment_id, alert), None


@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
@login_required
@role_required('doctor')
def acknowledge_alert(alert_id):
    if not request.is_json:   # JSON only, so a cross-site form can't trigger it
        return chat_json_error('Expected a JSON request.', 415)
    data = request.get_json(silent=True) or {}
    note = data.get('note', '')
    if not isinstance(note, str) or len(note.strip()) > ALERT_NOTE_MAX:
        return chat_json_error(f'Note must be text of at most {ALERT_NOTE_MAX} characters.', 400)
    found, error = alert_for_doctor(alert_id)
    if error:
        return error
    patient_uid, assessment_id, alert = found
    if not alert['acknowledged']:
        me = session['user']
        db.reference(f'alerts/{patient_uid}/{assessment_id}').set({
            'patient_uid': patient_uid, 'doctor_uid': me['uid'], 'assessment_id': assessment_id,
            'severity': alert['severity'], 'composite_score': alert['score'], 'created_at': alert['created_at'],
            'acknowledged_at': {'.sv': 'timestamp'}, 'acknowledged_by': me['uid'],
            'acknowledged_by_name': strip_dr(me.get('name')) or 'Doctor', 'note': note.strip(),
        })
        activity_log('alert_acknowledged', f"Acknowledged {alert['severity']} alert for {alert['patient_name']}",
                     details={'alert_id': alert_display_id(patient_uid, assessment_id), 'composite': alert['score'],
                              'note': 'added' if note.strip() else None})
    return jsonify({'success': True, 'already': alert['acknowledged'],
                    'unread_total': alert_unread_total(session['user']['uid'])})


@app.route('/api/alerts/<alert_id>/unacknowledge', methods=['POST'])
@login_required
@role_required('doctor')
def unacknowledge_alert(alert_id):
    if not request.is_json:
        return chat_json_error('Expected a JSON request.', 415)
    found, error = alert_for_doctor(alert_id)
    if error:
        return error
    patient_uid, assessment_id, alert = found
    if not alert['acknowledged']:
        return chat_json_error('This alert is not acknowledged.', 409)
    db.reference(f'alerts/{patient_uid}/{assessment_id}').delete()
    activity_log('alert_unacknowledged', f"Marked {alert['severity']} alert for {alert['patient_name']} as unread", 'info',
                 details={'alert_id': alert_display_id(patient_uid, assessment_id)})
    return jsonify({'success': True, 'unread_total': alert_unread_total(session['user']['uid'])})


@app.route('/api/alerts/unread')
@login_required
@role_required('doctor')
def alerts_unread():
    """Sidebar badge refresh (static/js/alert_badge.js polls this)."""
    return jsonify({'success': True, 'unread_total': alert_unread_total(session['user']['uid'])})


# ========== CHAT (patient <-> assigned doctor) ==========
# chats/{chat_id}: participants {uid: true}, doctor_uid, patient_uid, created_at, last_message,
#   messages/{push_id}: text, sender_uid, sender_name, sender_role, timestamp, read_by/{uid}: timestamp
# chat_unread/{uid}/{chat_id}: unread count for that user (drives the sidebar badges)
# All writes go through Flask. Browsers only *read* chats/{id} and chat_unread/{uid} with the Firebase
# client SDK (signed in via /api/chat/token); database.rules.json limits those reads to the participants
# who are still assigned to each other.
CHAT_MESSAGE_MAX = 2000
CHAT_PREVIEW_MAX = 120


def chat_id_for(doctor_uid, patient_uid):
    """Same id whichever order the two UIDs are given in."""
    return '_'.join(sorted([doctor_uid, patient_uid]))


def chat_partner(user_uid, chat_id):
    """(doctor_uid, patient_uid) if user_uid may use chat_id right now, else None.
    The chat must belong to the user and the patient must still be assigned to that doctor."""
    if not is_safe_key(user_uid) or not is_safe_key(chat_id):
        return None
    user = db.reference(f'users/{user_uid}').get()
    if not isinstance(user, dict):
        return None
    if user.get('role') == 'patient':
        doctor_uid = user.get('assigned_doctor_uid')
        if user.get('status') == 'active' and doctor_uid and chat_id_for(doctor_uid, user_uid) == chat_id:
            return doctor_uid, user_uid
        return None
    if user.get('role') == 'doctor' and user.get('status') == 'approved':
        patient_uid = db.reference(f'chats/{chat_id}/patient_uid').get()
        if not patient_uid or not is_safe_key(patient_uid) or chat_id_for(user_uid, patient_uid) != chat_id:
            return None
        patient = get_active_patient(patient_uid)
        if patient and patient_assigned_to(patient, user_uid):
            return user_uid, patient_uid
    return None


def can_access_chat(user_uid, chat_id):
    return chat_partner(user_uid, chat_id) is not None


def ensure_chat(doctor_uid, patient_uid):
    """Creates the chat node (participants etc.) the first time either side opens it."""
    chat_id = chat_id_for(doctor_uid, patient_uid)
    ref = db.reference(f'chats/{chat_id}')
    if ref.child('patient_uid').get() is None:
        ref.update({'participants': {doctor_uid: True, patient_uid: True},
                    'doctor_uid': doctor_uid, 'patient_uid': patient_uid,
                    'created_at': {'.sv': 'timestamp'}})
    return chat_id


def unread_count(user_uid, chat_id, messages=None):
    """Messages from the other participant that user_uid hasn't read yet."""
    if messages is None:
        messages = db.reference(f'chats/{chat_id}/messages').get() or {}
    return sum(1 for msg in messages.values()
               if isinstance(msg, dict) and msg.get('sender_uid') != user_uid
               and not (msg.get('read_by') or {}).get(user_uid))


def mark_chat_read(user_uid, chat_id, message_ids=None):
    """Marks the given (or all) incoming messages as read and refreshes the user's unread counter."""
    messages = db.reference(f'chats/{chat_id}/messages').get() or {}
    wanted = None if message_ids is None else set(message_ids)
    updates = {}
    for msg_id, msg in messages.items():
        if not isinstance(msg, dict) or msg.get('sender_uid') == user_uid or (msg.get('read_by') or {}).get(user_uid):
            continue
        if wanted is None or msg_id in wanted:
            updates[f'chats/{chat_id}/messages/{msg_id}/read_by/{user_uid}'] = {'.sv': 'timestamp'}
            msg.setdefault('read_by', {})[user_uid] = True  # so the count below sees it
    remaining = unread_count(user_uid, chat_id, messages)
    updates[f'chat_unread/{user_uid}/{chat_id}'] = remaining or None
    db.reference().update(updates)
    return len(updates) - 1, remaining


def chat_unread_total(user_uid):
    counts = db.reference(f'chat_unread/{user_uid}').get() or {}
    return sum(v for v in counts.values() if isinstance(v, int)) if isinstance(counts, dict) else 0


def format_chat_timestamp(ts):
    """'Just now', '5 min ago', '2:30 PM' (today), 'Yesterday', 'Oct 30' (or 'Oct 30, 2025')."""
    from datetime import datetime, timedelta
    if not isinstance(ts, (int, float)) or not ts:
        return ''
    when = datetime.fromtimestamp(ts / 1000)
    now = datetime.now()
    seconds = (now - when).total_seconds()
    if 0 <= seconds < 60:
        return 'Just now'
    if 0 <= seconds < 3600:
        return f'{int(seconds // 60)} min ago'
    if when.date() == now.date():
        return when.strftime('%I:%M %p').lstrip('0')
    if when.date() == (now - timedelta(days=1)).date():
        return 'Yesterday'
    return when.strftime('%b %d').replace(' 0', ' ') + ('' if when.year == now.year else when.strftime(', %Y'))


def chat_message_json(msg_id, msg):
    return {'id': msg_id, 'text': msg.get('text', ''), 'sender_uid': msg.get('sender_uid'),
            'sender_name': msg.get('sender_name', ''), 'sender_role': msg.get('sender_role'),
            'timestamp': msg.get('timestamp'), 'read_by': msg.get('read_by') or {}}


def chat_json_error(message, status):
    return jsonify({'success': False, 'error': message}), status


@app.context_processor
def inject_risk_ranges():
    """risk_ranges() in every template, so zone labels follow the admin-set thresholds (called lazily)."""
    return {'risk_ranges': risk_ranges}


@app.context_processor
def inject_chat_unread():
    """Initial sidebar badge; chat_badge.js keeps it live."""
    user = session.get('user') or {}
    if user.get('role') not in ('patient', 'doctor'):
        return {}
    boot = {'config': FIREBASE_WEB_CONFIG, 'me': user['uid'],
            'tokenUrl': url_for('chat_token'), 'unreadUrl': url_for('chat_unread')}
    try:
        return {'chat_unread_total': chat_unread_total(user['uid']), 'chat_boot': boot}
    except Exception:
        return {'chat_unread_total': 0, 'chat_boot': boot}


@app.context_processor
def inject_alert_unread():
    """Unacknowledged risk alerts for the doctor's sidebar badge (alert_badge.js refreshes it)."""
    user = session.get('user') or {}
    if user.get('role') != 'doctor':
        return {}
    try:
        return {'alert_unread_total': alert_unread_total(user['uid'])}
    except Exception:
        return {'alert_unread_total': 0}


@app.route('/api/chat/token')
@login_required
@role_required('patient', 'doctor')
def chat_token():
    """Firebase custom token for the signed-in user, so the browser's Firebase identity always
    matches the Flask session (database rules key off it)."""
    token = auth.create_custom_token(session['user']['uid'])
    return jsonify({'success': True, 'token': token.decode() if isinstance(token, bytes) else token})


def chat_page_context(chat_id, other, other_role):
    me = session['user']
    return {
        'chat': {
            'id': chat_id,
            'me': me['uid'],
            'my_role': me['role'],
            'other_uid': other['uid'],
            'other_name': other['name'],
            'other_role': other_role,
        },
        'chat_firebase_config': FIREBASE_WEB_CONFIG,
    }


@app.route('/patient/chat')
@login_required
@role_required('patient')
def patient_chat():
    uid = session['user']['uid']
    user_name = session['user'].get('name', 'Patient')
    user_initials = ''.join([word[0].upper() for word in user_name.split()[:2]]) or 'P'
    doctor = load_doctor_summary(db.reference(f'users/{uid}/assigned_doctor_uid').get())
    context = {}
    if doctor:
        chat_id = ensure_chat(doctor['uid'], uid)
        mark_chat_read(uid, chat_id)
        context = chat_page_context(chat_id, doctor, 'doctor')
    return render_template('patient_chat.html', user_name=user_name, user_initials=user_initials,
                           doctor=doctor, chat_unread_total=chat_unread_total(uid), **context)


def doctor_chat_list(doctor_uid):
    """Every active patient assigned to the doctor, most recent conversation first."""
    patients, _ = load_active_patients(doctor_uid)
    rows = []
    for p in patients:
        # Create missing chat nodes so the page's live listeners are allowed to attach before the first message
        chat_id = ensure_chat(doctor_uid, p['uid'])
        chat = db.reference(f'chats/{chat_id}').get() or {}
        last = chat.get('last_message') if isinstance(chat.get('last_message'), dict) else None
        rows.append({
            'uid': p['uid'], 'name': p['name'], 'chat_id': chat_id,
            'initials': ''.join(w[0].upper() for w in p['name'].split()[:2]) or 'P',
            'risk_level': p.get('risk_level'), 'risk_label': p.get('risk_label'),
            'last_text': last.get('text', '') if last else '',
            'last_mine': bool(last) and last.get('sender_uid') == doctor_uid,
            'last_ts': last.get('timestamp') if last and isinstance(last.get('timestamp'), (int, float)) else 0,
            'last_when': format_chat_timestamp(last.get('timestamp')) if last else '',
            'unread': unread_count(doctor_uid, chat_id, chat.get('messages') or {}),
        })
    rows.sort(key=lambda r: (-r['last_ts'], r['name'].lower()))
    return rows


@app.route('/doctor/messages')
@app.route('/doctor/messages/<patient_uid>')
@login_required
@role_required('doctor')
def doctor_messages(patient_uid=None):
    doctor_uid = session['user']['uid']
    user_name, user_initials = doctor_identity()
    context, selected = {}, None
    if patient_uid is not None:
        patient = get_active_patient(patient_uid)
        if not patient or not patient_assigned_to(patient, doctor_uid):
            flash("That patient isn't assigned to you.", 'error')
            return redirect(url_for('doctor_messages'))
        chat_id = ensure_chat(doctor_uid, patient_uid)
        mark_chat_read(doctor_uid, chat_id)
        selected = {'uid': patient_uid, 'name': patient.get('name') or 'Unnamed patient'}
        context = chat_page_context(chat_id, selected, 'patient')
    conversations = doctor_chat_list(doctor_uid)
    if selected:
        selected['initials'] = ''.join(w[0].upper() for w in selected['name'].split()[:2]) or 'P'
    return render_template('doctor_messages.html', user_name=user_name, user_initials=user_initials,
                           conversations=conversations, selected=selected,
                           chat_unread_total=chat_unread_total(doctor_uid), **context)


@app.route('/api/chat/send', methods=['POST'])
@login_required
@role_required('patient', 'doctor')
def chat_send():
    data = request.get_json(silent=True) or {}
    chat_id = str(data.get('chat_id') or '')
    text = data.get('text')
    if not isinstance(text, str) or not text.strip():
        return chat_json_error('Message cannot be empty.', 400)
    text = text.strip()
    if len(text) > CHAT_MESSAGE_MAX:
        return chat_json_error(f'Message is too long (max {CHAT_MESSAGE_MAX} characters).', 400)

    me = session['user']
    pair = chat_partner(me['uid'], chat_id)
    if not pair:
        return chat_json_error("You don't have access to this conversation.", 403)
    doctor_uid, patient_uid = pair
    ensure_chat(doctor_uid, patient_uid)
    recipient = patient_uid if me['uid'] == doctor_uid else doctor_uid

    sender_name = strip_dr(me.get('name')) if me['role'] == 'doctor' else (me.get('name') or 'Patient')
    msg_ref = db.reference(f'chats/{chat_id}/messages').push({
        'text': text,
        'sender_uid': me['uid'],
        'sender_name': sender_name,
        'sender_role': me['role'],
        'timestamp': {'.sv': 'timestamp'},
        'read_by': {me['uid']: {'.sv': 'timestamp'}},
    })
    saved = msg_ref.get() or {}
    db.reference(f'chats/{chat_id}/last_message').set({
        'text': text[:CHAT_PREVIEW_MAX], 'sender_uid': me['uid'], 'timestamp': saved.get('timestamp') or {'.sv': 'timestamp'}})
    db.reference(f'chat_unread/{recipient}/{chat_id}').transaction(lambda current: (current or 0) + 1)
    return jsonify({'success': True, 'message': chat_message_json(msg_ref.key, saved)})


@app.route('/api/chat/mark-read', methods=['POST'])
@login_required
@role_required('patient', 'doctor')
def chat_mark_read():
    data = request.get_json(silent=True) or {}
    chat_id = str(data.get('chat_id') or '')
    if not can_access_chat(session['user']['uid'], chat_id):
        return chat_json_error("You don't have access to this conversation.", 403)
    ids = data.get('message_ids')
    if ids is not None and (not isinstance(ids, list) or not all(isinstance(i, str) and is_safe_key(i) for i in ids)):
        return chat_json_error('message_ids must be a list of message ids.', 400)
    marked, remaining = mark_chat_read(session['user']['uid'], chat_id, ids)
    return jsonify({'success': True, 'marked': marked, 'unread': remaining,
                    'unread_total': chat_unread_total(session['user']['uid'])})


@app.route('/api/chat/messages')
@login_required
@role_required('patient', 'doctor')
def chat_messages():
    """Fallback when the browser can't read Firebase directly (database rules not deployed yet):
    messages after `after` (a message id), oldest first."""
    chat_id = request.args.get('chat_id', '')
    if not can_access_chat(session['user']['uid'], chat_id):
        return chat_json_error("You don't have access to this conversation.", 403)
    after = request.args.get('after', '')
    messages = db.reference(f'chats/{chat_id}/messages').get() or {}
    items = [chat_message_json(k, v) for k, v in sorted(messages.items()) if isinstance(v, dict) and (not after or k > after)]
    return jsonify({'success': True, 'messages': items,
                    'unread_total': chat_unread_total(session['user']['uid'])})


@app.route('/api/chat/unread')
@login_required
@role_required('patient', 'doctor')
def chat_unread():
    """Sidebar badge fallback when the browser can't listen to Firebase directly."""
    return jsonify({'success': True, 'unread_total': chat_unread_total(session['user']['uid'])})


@app.route('/admin/chats')
@admin_required
def admin_chats():
    # Placeholder: chat oversight for administrators is not built yet
    return jsonify({'success': False, 'error': 'Chat oversight for administrators is not available yet.'}), 501


# ========== ADMIN ==========
# Admins see every account and patient; doctor-patient assignment filtering doesn't apply here
ADMIN_PAGE_SIZE = 20
ADMIN_ACCOUNT_STATUSES = ['active', 'pending', 'approved', 'rejected', 'suspended']


def admin_load_users():
    users = db.reference('users').get() or {}
    return users if isinstance(users, dict) else {}


def admin_date(ms):
    return format_timestamp(ms, '%d %b %Y') if isinstance(ms, (int, float)) and ms else 'Date unavailable'


def admin_account_row(uid, user):
    created_at = user.get('created_at')
    return {
        'uid': uid,
        'name': user.get('name') or 'Unnamed user',
        'email': user.get('email', ''),
        'phone': user.get('phone', ''),
        'role': user.get('role', 'patient'),
        'status': user.get('status', 'active'),
        'created_at': created_at if isinstance(created_at, (int, float)) else 0,
        'created_date': admin_date(created_at)
    }


def admin_order_rows(users):
    raw_orders = db.reference('wristband_orders').get() or {}
    orders = []
    if isinstance(raw_orders, dict):
        for uid, order in raw_orders.items():
            if not isinstance(order, dict):
                continue
            profile = users.get(uid) if isinstance(users.get(uid), dict) else {}
            status = (order.get('status') or 'pending').lower()
            created_at = order.get('created_at')
            orders.append({
                'uid': uid,
                'name': order.get('fullname') or profile.get('name') or 'Unnamed patient',
                'email': profile.get('email', ''),
                'phone': order.get('phone', ''),
                'address': order.get('address', ''),
                'city': order.get('city', ''),
                'size': order.get('size', ''),
                'notes': order.get('notes', ''),
                'status': status if status in WRISTBAND_STATUSES else 'pending',
                'created_at': created_at if isinstance(created_at, (int, float)) else 0,
                'created_date': admin_date(created_at),
                'price': order_price(order), 'ref': order_reference(uid, order),
                'price_assumed': not isinstance(order.get('price'), int),   # placed before prices were stored
            })
    orders.sort(key=lambda order: order['created_at'], reverse=True)
    return orders


def admin_page(template, active, users, **context):
    """Renders an admin page with the shared sidebar/header context."""
    user_name = session['user'].get('name') or 'Administrator'
    pending_count = sum(1 for u in users.values()
                        if isinstance(u, dict) and u.get('role') == 'doctor' and u.get('status') == 'pending')
    return render_template(
        template,
        active_page=active,
        user_name=user_name,
        user_initials=''.join(word[0].upper() for word in user_name.split()[:2]) or 'A',
        csrf_token=session.setdefault('_admin_csrf_token', secrets.token_urlsafe(32)),
        pending_count=pending_count,
        **context
    )


def admin_back(endpoint, *also):
    """Redirects to an admin page after a POST, keeping its filters/page when the form sent them back.
    `also` lists other pages the form may return to (e.g. the dashboard); anything else goes to `endpoint`."""
    back = request.form.get('next', '')
    for allowed in (endpoint, *also):
        target = url_for(allowed)
        if back == target or back.startswith(target + '?'):
            return redirect(back)
    return redirect(url_for(endpoint))


def admin_dashboard_data(users):
    """Everything the admin dashboard and its CSV report show, from one read of users/ (plus
    alerts/ for acknowledgements and presence/ for active sessions). Dates use the server's local day."""
    from datetime import datetime, timedelta
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_ms = today.timestamp() * 1000
    yesterday_ms = (today - timedelta(days=1)).timestamp() * 1000
    month_ms = today.replace(day=1).timestamp() * 1000
    now_ms = time.time() * 1000

    acks = db.reference('alerts').get() or {}
    acks = acks if isinstance(acks, dict) else {}
    presence = db.reference('presence').get() or {}
    presence = presence if isinstance(presence, dict) else {}

    stats = dict.fromkeys(('patients', 'patients_this_month', 'doctors_active', 'doctors_pending',
                           'critical_today', 'critical_unacknowledged', 'assessments_today',
                           'assessments_yesterday', 'assessments_total'), 0)
    risk_counts = {level: 0 for level in RISK_LEVELS}
    pending_doctors, recent_assessments = [], []

    for uid, user in users.items():
        if not isinstance(user, dict):
            continue
        role, created = user.get('role'), user.get('created_at')
        if role == 'doctor':
            if user.get('status') == 'approved':
                stats['doctors_active'] += 1
            elif user.get('status') == 'pending':
                stats['doctors_pending'] += 1
                name = user.get('name') or 'Unnamed doctor'
                profile = user.get('profile') if isinstance(user.get('profile'), dict) else {}
                pending_doctors.append({
                    'uid': uid, 'name': name,
                    'initials': ''.join(w[0].upper() for w in strip_dr(name).split()[:2]) or 'D',
                    'specialization': profile.get('specialization') or 'Specialization not given',
                    'applied': admin_date(created), 'created_at': created if isinstance(created, (int, float)) else 0})
            continue
        if role != 'patient':
            continue
        stats['patients'] += 1
        if isinstance(created, (int, float)) and created >= month_ms:
            stats['patients_this_month'] += 1
        assessments = user.get('assessments') if isinstance(user.get('assessments'), dict) else {}
        latest = assessments.get(user.get('latest_assessment_id'))
        if isinstance(latest, dict):
            risk_counts[risk_band(compute_composite(latest))[0]] += 1
        patient_acks = acks.get(uid) if isinstance(acks.get(uid), dict) else {}
        for aid, a in assessments.items():
            if not isinstance(a, dict):
                continue
            stats['assessments_total'] += 1
            at = a.get('created_at')
            if not isinstance(at, (int, float)):
                continue
            level, label, color = risk_band(compute_composite(a))
            if at >= today_ms:
                stats['assessments_today'] += 1
                if level == 'critical':
                    stats['critical_today'] += 1
                    ack = patient_acks.get(aid)
                    if not (isinstance(ack, dict) and ack.get('acknowledged_at')):
                        stats['critical_unacknowledged'] += 1
            elif at >= yesterday_ms:
                stats['assessments_yesterday'] += 1
            recent_assessments.append({'name': user.get('name') or 'Unnamed patient', 'risk_label': label,
                                       'risk_color': color, 'created_at': at, 'date': admin_date(at)})

    stats['active_sessions'] = sum(1 for uid, seen in presence.items()
                                   if uid in users and isinstance(seen, (int, float)) and now_ms - seen <= ACTIVE_SESSION_MS)
    stats['assessments_change'] = stats['assessments_today'] - stats['assessments_yesterday']

    risk_distribution = [{'level': r['level'], 'label': r['label'], 'range': r['range'], 'color': r['color'],
                          'count': risk_counts[r['level']]} for r in risk_ranges()]
    pending_doctors.sort(key=lambda d: d['created_at'])   # oldest application first
    recent_assessments.sort(key=lambda a: a['created_at'], reverse=True)
    return {'stats': stats, 'risk_distribution': risk_distribution, 'risk_assessed': sum(risk_counts.values()),
            'critical_count': risk_counts['critical'], 'pending_doctors': pending_doctors,
            'recent_assessments': recent_assessments[:5],
            'today_label': f'{now:%A}, {now.day} {now:%B %Y}', 'today_iso': f'{now:%Y-%m-%d}'}


@app.route('/admin')
@admin_required
def admin_dashboard():
    users = admin_load_users()
    accounts = [admin_account_row(uid, u) for uid, u in users.items() if isinstance(u, dict)]
    accounts.sort(key=lambda account: account['created_at'], reverse=True)
    orders = admin_order_rows(users)
    data = admin_dashboard_data(users)

    return admin_page(
        'admin_dashboard.html', 'dashboard', users,
        **data,
        critical_dismissed=session.get('_critical_dismissed') == data['critical_count'],
        pending_orders=[o for o in orders if o['status'] == 'pending'],
        recent_users=accounts[:5],
        recent_orders=orders[:3],
        login_just_happened=session.pop('_login_toast', False)
    )


@app.route('/admin/dismiss-critical', methods=['POST'])
@admin_required
def admin_dismiss_critical():
    """Hides the critical-alerts banner for this admin session, until the critical count changes."""
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400
    count = request.form.get('count', '')
    if count.isdigit():
        session['_critical_dismissed'] = int(count)
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/report')
@admin_required
def admin_report():
    """CSV of the dashboard figures. Aggregate counts only, no individual results."""
    import csv
    import io
    data = admin_dashboard_data(admin_load_users())
    s = data['stats']
    rows = [
        ('Platform report', data['today_label']),
        ('Generated at', format_timestamp(time.time() * 1000, '%d %b %Y %H:%M')),
        (),
        ('Metric', 'Value'),
        ('Total patients', s['patients']),
        ('New patients this month', s['patients_this_month']),
        ('Active doctors', s['doctors_active']),
        ('Doctors pending approval', s['doctors_pending']),
        ('Assessments today', s['assessments_today']),
        ('Assessments yesterday', s['assessments_yesterday']),
        ('Assessments, all time', s['assessments_total']),
        ('Critical assessments today', s['critical_today']),
        ('Critical today, unacknowledged', s['critical_unacknowledged']),
        ('Active sessions (last 5 min)', s['active_sessions']),
        (),
        ('Risk band (latest assessment per patient)', 'Score range', 'Patients'),
        *[(r['label'], r['range'], r['count']) for r in data['risk_distribution']],
        ('Patients not yet assessed', '', s['patients'] - data['risk_assessed']),
    ]
    out = io.StringIO()
    csv.writer(out).writerows(rows)
    return app.response_class('﻿' + out.getvalue(), mimetype='text/csv', headers={
        'Content-Disposition': f"attachment; filename=platform_report_{data['today_iso']}.csv"})


# ========== ADMIN ANALYTICS ==========
ANALYTICS_RANGES = {'7d': 7, '30d': 30, '90d': 90}
RESPONSE_TARGET_MIN = 10   # alert response target, minutes


def analytics_range(args):
    """The selected period as local-midnight boundaries, plus the equal-length period before it."""
    from datetime import date, datetime, timedelta
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    key, error, custom_open = (args.get('range') or '30d').lower(), '', False
    start = end = None
    if key == 'custom':
        custom_open = True   # show the date picker
        raw_from, raw_to = args.get('from', ''), args.get('to', '')
        if raw_from or raw_to:
            try:
                first, last = date.fromisoformat(raw_from), date.fromisoformat(raw_to)
                first, last = min(first, last), max(first, last)
                last = min(last, today.date())
                first = min(first, last)
                start = datetime.combine(first, datetime.min.time())
                end = datetime.combine(last, datetime.min.time()) + timedelta(days=1)
            except ValueError:
                error = 'Choose both a start and an end date. Showing the last 30 days instead.'
    if start is None:
        key = key if key in ANALYTICS_RANGES else '30d'   # also covers an unfinished custom range
        days = ANALYTICS_RANGES[key]
        start, end = today - timedelta(days=days - 1), today + timedelta(days=1)
    days = (end - start).days
    if key in ANALYTICS_RANGES:
        label, compare = f'Last {days} days', f'previous {days} days'
    else:
        last_day = end - timedelta(days=1)
        label = f'{start.day} {start:%b %Y}' if days == 1 else f'{start.day} {start:%b %Y} – {last_day.day} {last_day:%b %Y}'
        compare = f'previous {days} day{"s" if days != 1 else ""}'
    prev_start = start - (end - start)
    ms = lambda dt: dt.timestamp() * 1000
    return {'key': key, 'label': label, 'compare': compare, 'days': days, 'error': error, 'custom_open': custom_open,
            'start': ms(start), 'end': ms(end), 'prev_start': ms(prev_start), 'prev_end': ms(start),
            'from_iso': f'{start:%Y-%m-%d}', 'to_iso': f'{end - timedelta(days=1):%Y-%m-%d}',
            'today': today, 'args': {k: v for k, v in (('range', key), ('from', f'{start:%Y-%m-%d}' if key == 'custom' else ''),
                                                        ('to', f'{end - timedelta(days=1):%Y-%m-%d}' if key == 'custom' else '')) if v}}


def load_model_metadata():
    """Models/model_metadata.json as a dict ({} if missing or unreadable)."""
    import json
    for d in ('Models', 'models'):
        try:
            with open(os.path.join(APP_DIR, d, 'model_metadata.json'), encoding='utf-8') as f:
                meta = json.load(f)
            return meta if isinstance(meta, dict) else {}
        except (OSError, ValueError):
            continue
    return {}


def model_accuracy():
    """(percent text, version) from the model metadata, so the figure follows the trained model."""
    meta = load_model_metadata()
    accuracy = meta.get('test_accuracy')
    text = f'{accuracy * 100:.1f}%' if isinstance(accuracy, (int, float)) else '—'
    return text, meta.get('version', '—')


def model_architecture(path):
    """Layer counts read from the .keras file's own config (no TensorFlow needed)."""
    import json
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            config = json.loads(z.read('config.json'))
        layers = [l.get('class_name') for l in config['config']['layers']]
        return {'conv': layers.count('Conv2D'), 'dense': layers.count('Dense'), 'pool': layers.count('MaxPooling2D')}
    except Exception:
        return None


def ml_model_info():
    """What the ML Models page shows about the facial emotion model: metadata + the model file itself."""
    meta = load_model_metadata()
    pct = lambda v: f'{v * 100:.2f}%' if isinstance(v, (int, float)) else '—'
    info = {
        'found': bool(meta), 'version': meta.get('version', ''),
        'name': f"Custom CNN {meta.get('version', '')}".strip(),
        'test_accuracy': pct(meta.get('test_accuracy')), 'val_accuracy': pct(meta.get('val_accuracy')),
        'epochs': meta.get('trained_epochs'), 'params': meta.get('total_params'),
        'input': f"{meta['input_size']}×{meta['input_size']} {meta.get('color_mode', '')}".strip() if meta.get('input_size') else '—',
        'classes': meta.get('classes') or [], 'improvements': meta.get('improvements') or [],
        'dataset': meta.get('dataset', '—'), 'preprocessing': meta.get('preprocessing', ''),
        'loaded': FER_MODEL is not None, 'file': None, 'architecture': None,
    }
    if FER_MODEL_PATH:
        stat = os.stat(FER_MODEL_PATH)
        info['file'] = {'name': os.path.relpath(FER_MODEL_PATH, APP_DIR).replace('\\', '/'),
                        'size': f'{stat.st_size / 1_048_576:.1f} MB', 'modified': format_timestamp(stat.st_mtime * 1000, '%d %b %Y')}
        info['architecture'] = model_architecture(FER_MODEL_PATH)
    # Notebook link: KAGGLE_NOTEBOOK_URL in .env, or "kaggle_url" in the metadata. Only kaggle.com links are used.
    url = (os.environ.get('KAGGLE_NOTEBOOK_URL') or meta.get('kaggle_url') or '').strip()
    info['kaggle_url'] = url if url.startswith(('https://www.kaggle.com/', 'https://kaggle.com/')) else ''
    return info


def ml_page(**extra):
    users = admin_load_users()
    settings = scoring_settings(refresh=True)
    return admin_page('admin_ml_models.html', 'ml_models', users, model=ml_model_info(),
                      weights=settings['weights'], thresholds=settings['thresholds'],
                      default_weights=DEFAULT_FUSION_WEIGHTS, default_thresholds=DEFAULT_RISK_THRESHOLDS, **extra)


@app.route('/admin/ml-models')
@admin_required
def admin_ml_models():
    return ml_page()


@app.route('/admin/ml-models/weights', methods=['POST'])
@admin_required
def admin_save_fusion_weights():
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400
    weights = clean_fusion_weights({k: request.form.get(k) for k in FUSION_KEYS})
    if weights is None:
        flash('Fusion weights must each be between 0 and 1 and add up to exactly 1.00.', 'error')
        return redirect(url_for('admin_ml_models') + '#fusion')
    old = fusion_weights()
    db.reference('system/fusion_weights').set({**weights, 'updated_at': {'.sv': 'timestamp'},
                                                'updated_by': session['user']['uid']})
    fmt = lambda w: ' / '.join(f'{k.upper()} {w[k]:.2f}' for k in FUSION_KEYS)
    activity_log('fusion_weights_changed', 'Changed fusion weights', details={'from': fmt(old), 'to': fmt(weights)})
    scoring_settings(refresh=True)
    flash('Fusion weights saved. They apply to assessments completed from now on.', 'success')
    return redirect(url_for('admin_ml_models') + '#fusion')


@app.route('/admin/ml-models/thresholds', methods=['POST'])
@admin_required
def admin_save_risk_thresholds():
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400
    thresholds = clean_risk_thresholds({k: request.form.get(k) for k in DEFAULT_RISK_THRESHOLDS})
    if thresholds is None:
        flash('Thresholds are out of range: Low 10-40, Moderate at least Low + 5 (max 70), '
              'High at least Moderate + 5 (max 90).', 'error')
        return redirect(url_for('admin_ml_models') + '#thresholds')
    old = risk_thresholds()
    db.reference('system/risk_thresholds').set({**thresholds, 'updated_at': {'.sv': 'timestamp'},
                                                 'updated_by': session['user']['uid']})
    fmt = lambda t: f"Low ≤{t['low']}, Moderate ≤{t['moderate']}, High ≤{t['high']}"
    activity_log('risk_thresholds_changed', 'Changed risk thresholds', details={'from': fmt(old), 'to': fmt(thresholds)})
    scoring_settings(refresh=True)
    flash('Risk thresholds saved. All assessments are now classified with the new bands.', 'success')
    return redirect(url_for('admin_ml_models') + '#thresholds')


def month_starts(today, count):
    from datetime import datetime
    out = []
    for back in range(count - 1, -1, -1):
        year, month = today.year, today.month - back
        while month <= 0:
            month, year = month + 12, year - 1
        out.append(datetime(year, month, 1))
    return out


def admin_analytics_data(args):
    """Stats, chart series and per-assessment rows for /admin/analytics and its exports.
    One read each of users/, alerts/, chats/ and login_counts/."""
    from datetime import datetime, timedelta
    rng = analytics_range(args)
    users = admin_load_users()
    acks = db.reference('alerts').get() or {}
    acks = acks if isinstance(acks, dict) else {}
    chats = db.reference('chats').get() or {}
    chats = chats if isinstance(chats, dict) else {}
    logins = db.reference('login_counts').get() or {}
    logins = logins if isinstance(logins, dict) else {}
    today = rng['today']
    month_start = today.replace(day=1)
    month_ms = month_start.timestamp() * 1000
    in_range = lambda t: rng['start'] <= t < rng['end']
    in_prev = lambda t: rng['prev_start'] <= t < rng['prev_end']

    rows, responses = [], []   # responses: (alert created ms, minutes to acknowledge)
    patients = paired = 0
    patients_before_month = paired_before_month = 0
    created_times = []
    for uid, user in users.items():
        if not isinstance(user, dict) or user.get('role') != 'patient':
            continue
        patients += 1
        created = user.get('created_at') if isinstance(user.get('created_at'), (int, float)) else None
        created_times.append(created)
        wb = user.get('wristband') if isinstance(user.get('wristband'), dict) else {}
        is_paired = bool(wb.get('device_id'))
        paired += is_paired
        if created is None or created < month_ms:
            patients_before_month += 1
            paired_at = wb.get('paired_at')
            paired_before_month += is_paired and isinstance(paired_at, (int, float)) and paired_at < month_ms
        name = user.get('name') or 'Unnamed patient'
        patient_acks = acks.get(uid) if isinstance(acks.get(uid), dict) else {}
        for aid, a in (user.get('assessments') or {}).items() if isinstance(user.get('assessments'), dict) else []:
            if not isinstance(a, dict) or not isinstance(a.get('created_at'), (int, float)):
                continue
            score = compute_composite(a)
            level, label, color = risk_band(score)
            rows.append({'uid': uid, 'name': name, 'created_at': a['created_at'],
                         'date': format_timestamp(a['created_at'], '%Y-%m-%d'), 'time': format_timestamp(a['created_at'], '%H:%M'),
                         'phq': a.get('phq_score'), 'gad': a.get('gad_score'), 'pss': a.get('pss_score'),
                         'facial': facial_mood(a, 'Skipped', ''), 'composite': score,
                         'level': level, 'risk_label': label, 'risk_color': color})
            ack = patient_acks.get(aid)
            if alert_severity(score) and isinstance(ack, dict) and isinstance(ack.get('acknowledged_at'), (int, float)):
                responses.append((a['created_at'], max(0, ack['acknowledged_at'] - a['created_at']) / 60000))
    rows.sort(key=lambda r: r['created_at'], reverse=True)

    cur = [r for r in rows if in_range(r['created_at'])]
    prev = [r for r in rows if in_prev(r['created_at'])]
    avg = lambda xs: sum(xs) / len(xs) if xs else None

    def fmt_minutes(m):
        if m is None:
            return '—'
        if m < 60:
            return f'{m:.1f} min'
        return f'{m / 60:.1f} h' if m < 1440 else f'{m / 1440:.1f} days'

    def pct_change(now, before):
        if not before:
            return None
        return round((now - before) / before * 100)

    # Activities: logins (counted from when tracking started) + assessments + chat messages
    def logins_between(lo, hi):
        total = 0
        for day, n in logins.items():
            try:
                t = datetime.strptime(day, '%Y-%m-%d').timestamp() * 1000
            except ValueError:
                continue
            if lo <= t < hi and isinstance(n, int):
                total += n
        return total
    message_times = [m['timestamp'] for chat in chats.values() if isinstance(chat, dict)
                     for m in (chat.get('messages') or {}).values()
                     if isinstance(m, dict) and isinstance(m.get('timestamp'), (int, float))]
    act_cur = logins_between(rng['start'], rng['end']) + len(cur) + sum(1 for t in message_times if in_range(t))
    act_prev = logins_between(rng['prev_start'], rng['prev_end']) + len(prev) + sum(1 for t in message_times if in_prev(t))

    avg_cur, avg_prev = avg([r['composite'] for r in cur]), avg([r['composite'] for r in prev])
    resp_cur = avg([m for t, m in responses if in_range(t)])
    usage_now = paired / patients * 100 if patients else 0
    usage_before = paired_before_month / patients_before_month * 100 if patients_before_month else 0
    accuracy, model_version = model_accuracy()
    stats = {
        'total': len(cur), 'total_change': len(cur) - len(prev),
        'avg_score': round(avg_cur, 1) if avg_cur is not None else None,
        'avg_band': risk_band(avg_cur) if avg_cur is not None else None,
        'avg_change': round(avg_cur - avg_prev, 1) if avg_cur is not None and avg_prev is not None else None,
        'accuracy': accuracy, 'model_version': model_version,
        'response': fmt_minutes(resp_cur), 'response_min': resp_cur,
        'response_ok': resp_cur is not None and resp_cur < RESPONSE_TARGET_MIN,
        'usage': round(usage_now), 'usage_change': round(usage_now - usage_before), 'paired': paired, 'patients': patients,
        'activities': act_cur, 'activities_change': pct_change(act_cur, act_prev),
    }

    # Chart 1 + 2: last 12 months
    months = month_starts(today, 12)
    bounds = [m.timestamp() * 1000 for m in months] + [month_starts(today + timedelta(days=32), 1)[0].timestamp() * 1000]
    month_labels = [f'{m:%b %y}' for m in months]
    registrations = [sum(1 for c in created_times if c is None or c < bounds[i + 1]) for i in range(12)]
    risk_series = []
    labels = {r['level']: r['label'] for r in risk_ranges()}
    for level in reversed(RISK_LEVELS):   # Critical first, as listed in the legend
        risk_series.append({'level': level, 'label': labels[level],
                            'color': {'low': '#059669', 'moderate': '#F59E0B', 'high': '#EA580C', 'critical': '#DC2626'}[level],
                            'values': [sum(1 for r in rows if r['level'] == level and bounds[i] <= r['created_at'] < bounds[i + 1])
                                       for i in range(12)]})

    # Chart 3 + 4: last 14 days
    days = [today - timedelta(days=13 - i) for i in range(14)]
    day_bounds = [d.timestamp() * 1000 for d in days] + [(today + timedelta(days=1)).timestamp() * 1000]
    day_labels = [f'{d.day} {d:%b}' for d in days]
    daily = [sum(1 for r in rows if day_bounds[i] <= r['created_at'] < day_bounds[i + 1]) for i in range(14)]
    response_daily = []
    for i in range(14):
        mins = [m for t, m in responses if day_bounds[i] <= t < day_bounds[i + 1]]
        response_daily.append(round(avg(mins), 1) if mins else None)

    charts = {
        'registrations': {'labels': month_labels, 'values': registrations, 'has_data': any(registrations)},
        'risk': {'labels': month_labels, 'series': risk_series, 'has_data': any(any(s['values']) for s in risk_series)},
        'daily': {'labels': day_labels, 'values': daily, 'has_data': any(daily)},
        'response': {'labels': day_labels, 'values': response_daily, 'target': RESPONSE_TARGET_MIN,
                     'has_data': any(v is not None for v in response_daily)},
    }
    distribution = [{'label': s['label'], 'color': s['color'], 'count': sum(1 for r in cur if r['level'] == s['level'])}
                    for s in reversed(risk_series)]
    return {'range': rng, 'stats': stats, 'charts': charts, 'rows': cur, 'distribution': distribution,
            'users': users, 'generated': format_timestamp(time.time() * 1000, '%d %b %Y, %H:%M')}


@app.route('/admin/analytics')
@admin_required
def admin_analytics():
    data = admin_analytics_data(request.args)
    users, period = data.pop('users'), data.pop('range')   # 'range' would shadow Jinja's range()
    return admin_page('admin_analytics.html', 'analytics', users, period=period, **data,
                      range_options=[('7d', '7D'), ('30d', '30D'), ('90d', '90D')])


def csv_cell(value):
    """Stops spreadsheet apps treating a cell (e.g. a patient name) as a formula."""
    text = '' if value is None else str(value)
    return "'" + text if text[:1] in ('=', '+', '-', '@', '\t', '\r') else text


@app.route('/admin/analytics/export.csv')
@admin_required
def admin_analytics_csv():
    import csv
    import io
    data = admin_analytics_data(request.args)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['Patient', 'Date', 'Time', 'PHQ-9', 'GAD-7', 'PSS', 'Facial scan', 'Composite', 'Risk level'])
    for r in data['rows']:
        writer.writerow([csv_cell(r['name']), r['date'], r['time'], csv_cell(r['phq']), csv_cell(r['gad']),
                         csv_cell(r['pss']), csv_cell(r['facial'] or '-'), r['composite'], r['risk_label']])
    rng = data['range']
    return app.response_class('﻿' + out.getvalue(), mimetype='text/csv', headers={
        'Content-Disposition': f"attachment; filename=assessments_{rng['from_iso']}_to_{rng['to_iso']}.csv"})


@app.route('/admin/analytics/export.pdf')
@admin_required
def admin_analytics_pdf():
    import io
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.legends import Legend
    from reportlab.graphics.widgets.markers import makeMarker
    import math

    data = admin_analytics_data(request.args)
    rng, s, charts = data['range'], data['stats'], data['charts']
    green, ink, muted, line = colors.HexColor('#0F7A5F'), colors.HexColor('#0F172A'), colors.HexColor('#64748B'), colors.HexColor('#E2E8F0')
    title = ParagraphStyle('t', fontName='Helvetica-Bold', fontSize=20, textColor=ink, leading=24)
    sub = ParagraphStyle('s', fontName='Helvetica', fontSize=10, textColor=muted, leading=14)
    h2 = ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=12, textColor=ink, leading=16, spaceBefore=10, spaceAfter=4)
    note = ParagraphStyle('n', fontName='Helvetica', fontSize=8, textColor=muted, leading=11)
    width = A4[0] - 36 * mm

    def sign(n, unit=''):
        return f'{n:+g}{unit}' if n else f'0{unit}'

    band = s['avg_band']
    stat_rows = [
        ['Total assessments', str(s['total']), f"{sign(s['total_change'])} vs {rng['compare']}"],
        ['Average risk score', '—' if s['avg_score'] is None else f"{s['avg_score']} / 100",
         '' if band is None else f"{band[1]} range" + ('' if s['avg_change'] is None else f" · {sign(s['avg_change'])} vs {rng['compare']}")],
        ['Detection rate', s['accuracy'], f"Model accuracy {s['model_version']} (FER-2013)"],
        ['Avg response time', s['response'], f"Target: under {RESPONSE_TARGET_MIN} min" + ('' if s['response_min'] is None else (' (met)' if s['response_ok'] else ' (missed)'))],
        ['Wristband usage', f"{s['usage']}%", f"{s['paired']} of {s['patients']} patients paired · {sign(s['usage_change'], ' pts')} this month"],
        ['Activities completed', str(s['activities']), 'Logins, assessments and chat messages' + ('' if s['activities_change'] is None else f" · {sign(s['activities_change'], '%')} vs {rng['compare']}")],
    ]
    stat_table = Table([['Metric', 'Value', 'Detail']] + stat_rows, colWidths=[45 * mm, 32 * mm, width - 77 * mm])
    stat_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9), ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
        ('FONT', (1, 1), (1, -1), 'Helvetica-Bold', 10), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 0), (-1, 0), green), ('TEXTCOLOR', (2, 1), (2, -1), muted),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, line), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    def empty(drawing, text='No data available for this period'):
        drawing.add(String(drawing.width / 2, drawing.height / 2, text, textAnchor='middle', fontName='Helvetica', fontSize=9, fillColor=muted))
        return drawing

    def style_axes(chart, labels, step):
        chart.categoryAxis.categoryNames = [l if i % step == 0 else '' for i, l in enumerate(labels)]
        chart.categoryAxis.labels.fontName = chart.valueAxis.labels.fontName = 'Helvetica'
        chart.categoryAxis.labels.fontSize = chart.valueAxis.labels.fontSize = 7
        chart.categoryAxis.labels.fillColor = chart.valueAxis.labels.fillColor = muted
        chart.categoryAxis.strokeColor = chart.valueAxis.strokeColor = line
        chart.valueAxis.visibleGrid = True
        chart.valueAxis.gridStrokeColor = line
        chart.valueAxis.valueMin = 0

    def count_axis(chart, series):
        top = max((v for row in series for v in row if v is not None), default=0)
        chart.valueAxis.valueStep = max(1, math.ceil(top / 5))   # counts: no 0.5 steps

    def line_chart(series, labels, colours, height=62 * mm, value_max=None, counts=False, markers=False):
        d = Drawing(width / 2 - 4 * mm, height)
        c = HorizontalLineChart()
        c.x, c.y, c.width, c.height = 26, 22, d.width - 34, height - 34
        c.data = series
        style_axes(c, labels, 2)
        if value_max:
            c.valueAxis.valueMax = value_max
        if counts:
            count_axis(c, series)
        for i, col in enumerate(colours):
            c.lines[i].strokeColor = colors.HexColor(col)
            c.lines[i].strokeWidth = 2
            if markers and i == 0:
                c.lines[i].symbol = makeMarker('FilledCircle', size=4, fillColor=colors.HexColor(col), strokeColor=None)
        d.add(c)
        return d

    def bar_chart(series, labels, colours, stacked=False, height=62 * mm, step=2):
        d = Drawing(width / 2 - 4 * mm, height)
        c = VerticalBarChart()
        c.x, c.y, c.width, c.height = 26, 22, d.width - 34, height - 34
        c.data = series
        style_axes(c, labels, step)
        if stacked:
            c.categoryAxis.style = 'stacked'
            count_axis(c, [[sum(col) for col in zip(*series)]])
        else:
            count_axis(c, series)
        c.barSpacing, c.groupSpacing = 0, 4
        for i, col in enumerate(colours):
            c.bars[i].fillColor = colors.HexColor(col)
            c.bars[i].strokeColor = None
        d.add(c)
        return d

    reg, risk, daily, resp = charts['registrations'], charts['risk'], charts['daily'], charts['response']
    blank = lambda: Drawing(width / 2 - 4 * mm, 62 * mm)
    c1 = line_chart([reg['values']], reg['labels'], ['#0F7A5F'], counts=True) if reg['has_data'] else empty(blank())
    if risk['has_data']:
        c2 = bar_chart([sr['values'] for sr in risk['series']], risk['labels'], [sr['color'] for sr in risk['series']], stacked=True)
        legend = Legend()
        legend.x, legend.y, legend.alignment = 30, c2.height - 4, 'right'
        legend.columnMaximum, legend.fontName, legend.fontSize = 1, 'Helvetica', 7
        legend.deltax, legend.dxTextSpace, legend.dx, legend.dy = 56, 4, 6, 6
        legend.colorNamePairs = [(colors.HexColor(sr['color']), sr['label']) for sr in risk['series']]
        c2.add(legend)
    else:
        c2 = empty(blank())
    c3 = bar_chart([daily['values']], daily['labels'], ['#0F7A5F']) if daily['has_data'] else empty(blank())
    if resp['has_data']:
        # Days with no acknowledged alerts have no average, so they're left as gaps (not drawn as 0 minutes)
        values = resp['values']
        top = max(v for v in values if v is not None)
        c4 = line_chart([values, [resp['target']] * len(values)], resp['labels'], ['#2563EB', '#DC2626'],
                        value_max=max(top, resp['target']) * 1.15, markers=True)
    else:
        c4 = empty(blank(), 'No acknowledged alerts in the last 14 days')

    def chart_cell(heading, drawing):
        return [Paragraph(heading, ParagraphStyle('ch', parent=h2, fontSize=10, spaceBefore=0)), drawing]

    grid = Table([[chart_cell('Patient registrations (cumulative, 12 months)', c1), chart_cell('Risk distribution by month', c2)],
                  [chart_cell('Daily assessment completions (14 days)', c3), chart_cell(f'Alert response time, minutes (target {RESPONSE_TARGET_MIN})', c4)]],
                 colWidths=[width / 2] * 2)
    grid.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0),
                              ('BOTTOMPADDING', (0, 0), (-1, -1), 10)]))

    dist = data['distribution']
    dist_table = Table([['Risk level', 'Assessments']] + [[d['label'], str(d['count'])] for d in dist],
                       colWidths=[45 * mm, 32 * mm], hAlign='LEFT')
    dist_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9), ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
        ('BACKGROUND', (0, 0), (-1, 0), green), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, line), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        *[('TEXTCOLOR', (0, i + 1), (0, i + 1), colors.HexColor(d['color'])) for i, d in enumerate(dist)],
    ]))

    story = [Paragraph('Analytics &amp; Reports', title),
             Paragraph(f"Mental Disorder Detection · {rng['label']} · generated {data['generated']}", sub),
             Spacer(1, 8 * mm), Paragraph('Key figures', h2), stat_table]
    if not data['rows']:
        story += [Spacer(1, 3 * mm), Paragraph('No assessments were completed in this period.', note)]
    story += [Paragraph(f"Assessments by risk level ({rng['label'].lower() if rng['key'] != 'custom' else rng['label']})", h2), dist_table,
              Spacer(1, 4 * mm), KeepTogether([Paragraph('Trends', h2), grid]),
              Paragraph('Activities count logins from when login tracking began. Detection rate is the facial '
                        'emotion model\'s test accuracy on FER-2013, not a clinical detection rate.', note)]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title='Analytics & Reports', author='Mental Disorder Detection')
    doc.build(story)
    return app.response_class(buf.getvalue(), mimetype='application/pdf', headers={
        'Content-Disposition': f"attachment; filename=analytics_report_{rng['from_iso']}_to_{rng['to_iso']}.pdf"})


# ========== SYSTEM SETTINGS (admin) ==========
# Stored at system/settings/{alerts,questionnaire,wristband}, plus system/settings/_meta/{section} for
# "last saved" info. Risk thresholds are NOT here: they belong to the ML Models page.
# field: (type, default, min, max, choices)
SETTINGS_SCHEMA = {
    'alerts': {
        'hr_max': ('int', 120, 30, 220, None),
        'gsr_max': ('float', 0.8, 0.1, 5.0, None),
        'temp_max': ('float', 37.5, 35.0, 42.0, None),
        'escalation_min': ('int', 30, 5, 180, None),
    },
    'questionnaire': {
        'cooldown_hours': ('int', 24, 0, 168, None),
        'mandatory_on_login': ('bool', False, None, None, None),
    },
    'wristband': {
        'interval_s': ('choice', 5, None, None, (1, 5, 10, 30, 60)),
        'buffer_min': ('int', 5, 1, 60, None),
        'auto_pair': ('bool', False, None, None, None),
    },
}
SETTINGS_SECTIONS = tuple(SETTINGS_SCHEMA)
SETTINGS_DEFAULTS = {s: {k: spec[1] for k, spec in fields.items()} for s, fields in SETTINGS_SCHEMA.items()}
_settings_cache = {'at': 0.0, 'values': SETTINGS_DEFAULTS, 'meta': {}, 'stale': False}


def parse_setting(spec, raw):
    """(value, error) for one submitted field; raw is a form string (or None for an unchecked checkbox)."""
    kind, _default, lo, hi, choices = spec
    if kind == 'bool':
        return raw in ('1', 'on', 'true', True), None
    text = '' if raw is None else str(raw).strip()
    if not text:
        return None, 'Required.'
    try:
        number = float(text)
    except ValueError:
        return None, 'Enter a number.'
    if number != number or number in (float('inf'), float('-inf')):
        return None, 'Enter a number.'
    if kind == 'choice':
        return (int(number), None) if number in choices else (None, 'Choose one of the listed options.')
    if kind == 'int':
        if number != int(number):
            return None, 'Enter a whole number.'
        number = int(number)
    else:
        number = round(number, 1)
    if not lo <= number <= hi:
        return None, f'Must be between {lo:g} and {hi:g}.'
    return number, None


def clean_settings_section(section, raw):
    """Stored values for one section, falling back to defaults field by field."""
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for key, spec in SETTINGS_SCHEMA[section].items():
        value = raw.get(key)
        if spec[0] == 'bool':
            out[key] = value if isinstance(value, bool) else spec[1]
            continue
        parsed, error = parse_setting(spec, value if isinstance(value, (int, float)) and not isinstance(value, bool) else None)
        out[key] = spec[1] if error else parsed
    return out


def system_settings(refresh=False):
    """{'values', 'meta', 'stale'}: cached for SETTINGS_TTL seconds; keeps the last good copy if Firebase fails."""
    if refresh or time.time() - _settings_cache['at'] > SETTINGS_TTL:
        try:
            raw = db.reference('system/settings').get() or {}
            raw = raw if isinstance(raw, dict) else {}
            _settings_cache['values'] = {s: clean_settings_section(s, raw.get(s)) for s in SETTINGS_SECTIONS}
            meta = raw.get('_meta') if isinstance(raw.get('_meta'), dict) else {}
            _settings_cache['meta'] = {s: m for s, m in meta.items() if s in SETTINGS_SECTIONS and isinstance(m, dict)}
            _settings_cache['stale'] = False
        except Exception:
            _settings_cache['stale'] = True   # show the last known settings with a warning
        _settings_cache['at'] = time.time()
    return _settings_cache


def setting(section, key):
    return system_settings()['values'][section][key]


def settings_page(values=None, errors=None, status=200):
    settings = system_settings(refresh=values is None)
    meta = {s: {'ago': time_ago(m.get('at')), 'full': format_timestamp(m.get('at'), '%d %b %Y, %H:%M'),
                'by': m.get('by_name') or 'an admin'}
            for s, m in settings['meta'].items() if isinstance(m.get('at'), (int, float))}
    t = risk_thresholds()
    page = admin_page('admin_settings.html', 'settings', admin_load_users(),
                      values=values or settings['values'], errors=errors or {}, meta=meta,
                      stale=settings['stale'], defaults=SETTINGS_DEFAULTS, schema=SETTINGS_SCHEMA,
                      critical_min=t['high'] + 1, high_max=t['high'])
    return page, status


@app.route('/admin/settings')
@admin_required
def admin_settings():
    return settings_page()


@app.route('/admin/settings/save', methods=['POST'])
@admin_required
def admin_save_settings():
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400
    target = request.form.get('section', 'all')
    sections = SETTINGS_SECTIONS if target == 'all' else (target,) if target in SETTINGS_SECTIONS else ()
    if not sections:
        flash('Unknown settings section.', 'error')
        return redirect(url_for('admin_settings'))

    current = system_settings(refresh=True)['values']
    values = {s: dict(current[s]) for s in SETTINGS_SECTIONS}   # sections not being saved keep their stored values
    errors = {}
    for section in sections:
        for key, spec in SETTINGS_SCHEMA[section].items():
            value, error = parse_setting(spec, request.form.get(f'{section}.{key}'))
            if error:
                errors[f'{section}.{key}'] = error
                # Re-show what the admin typed so they can correct it
                values[section][key] = request.form.get(f'{section}.{key}', '')
            else:
                values[section][key] = value
    if errors:
        flash(f"Nothing was saved: {len(errors)} field{'s need' if len(errors) != 1 else ' needs'} correcting.", 'error')
        return settings_page(values=values, errors=errors, status=400)

    me = session['user']
    updates = {}
    for section in sections:
        updates[f'system/settings/{section}'] = values[section]
        updates[f'system/settings/_meta/{section}'] = {'at': {'.sv': 'timestamp'}, 'by_uid': me['uid'],
                                                      'by_name': me.get('name') or 'Admin'}
    try:
        db.reference().update(updates)
    except Exception:
        flash('Settings could not be saved: Firebase is unreachable. Please try again.', 'error')
        return settings_page(values=values, status=503)
    changed = [f'{s}.{k}: {current[s][k]} → {values[s][k]}' for s in sections for k in SETTINGS_SCHEMA[s]
               if current[s][k] != values[s][k]]
    activity_log('settings_saved', 'Saved system settings' + ('' if target == 'all' else f' ({target})'),
                 details={'sections': ', '.join(sections), 'changes': '; '.join(changed) or 'no changes'})
    system_settings(refresh=True)
    flash('Settings saved successfully.', 'success')
    return redirect(url_for('admin_settings') + ('' if target == 'all' else f'#{target}'))


# ========== ADMIN SECURITY ==========
# Every check below is verified live where it can be (deployed rules, user records, app config);
# features that don't exist score 0 and are listed as recommendations.
DOCTOR_PATIENT_LIMIT = 25
_rules_cache = {'at': 0.0, 'value': None}


def deployed_firebase_rules():
    """{'ok', 'rules', 'matches_local', 'error'} for the live database rules, read with the service
    account (cached 5 minutes)."""
    import json
    import re
    import urllib.request
    if time.time() - _rules_cache['at'] < 300 and _rules_cache['value']:
        return _rules_cache['value']
    result = {'ok': False, 'rules': None, 'matches_local': False, 'error': ''}
    try:
        token = firebase_admin.get_app().credential.get_access_token().access_token
        url = FIREBASE_WEB_CONFIG['databaseURL'].rstrip('/') + '/.settings/rules.json'
        req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token})
        body = urllib.request.urlopen(req, timeout=10).read().decode()
        deployed = json.loads(re.sub(r'^\s*//.*$', '', body, flags=re.M))
        with open(os.path.join(APP_DIR, 'database.rules.json'), encoding='utf-8') as f:
            local = json.load(f)
        result.update(ok=True, rules=deployed, matches_local=deployed == local)
    except Exception as e:
        result['error'] = type(e).__name__
    _rules_cache.update(at=time.time(), value=result)
    return result


def count_rule_expressions(node):
    """(rule expressions, paths that have one) in a rules tree."""
    exprs, paths = 0, 0
    if isinstance(node, dict):
        here = [k for k in node if k.startswith('.')]
        exprs += len(here)
        paths += 1 if here else 0
        for k, v in node.items():
            if not k.startswith('.'):
                e, p = count_rule_expressions(v)
                exprs, paths = exprs + e, paths + p
    return exprs, paths


def csrf_coverage():
    """How each state-changing (POST) route is protected, read from the view functions themselves."""
    import inspect
    import re
    out = {'token': 0, 'json': 0, 'device': 0, 'none': []}
    for rule in app.url_map.iter_rules():
        if 'POST' not in rule.methods:
            continue
        src = inspect.getsource(inspect.unwrap(app.view_functions[rule.endpoint]))
        if 'admin_csrf_valid' in src or re.search(r"compare_digest\(request\.form\.get\('csrf_token'", src):
            out['token'] += 1           # per-session form token (admin pages, and the 2FA sign-in step)
        elif rule.endpoint == 'wristband_data':
            out['device'] += 1          # device API key, no browser session to forge
        elif re.search(r'is_json|get_json\((?![^)]*force=True)', src):
            # JSON-only: get_json() (even silent) only parses a JSON Content-Type, which a
            # cross-site HTML form can't send, so a forged form post arrives with no data
            out['json'] += 1
        else:
            out['none'].append(rule.rule)
    return out


def mask_uid(uid):
    return f'{uid[:4]}••••••••{uid[-4:]}' if uid and len(uid) > 8 else '••••'


def security_data():
    users = admin_load_users()
    now_ms = time.time() * 1000
    host = (request.host or '').split(':')[0]
    is_local = host in ('localhost', '127.0.0.1', '::1') or host.endswith('.local')
    is_https = request.is_secure or request.headers.get('X-Forwarded-Proto', '') == 'https'

    rules = deployed_firebase_rules()
    root_rules = (rules['rules'] or {}).get('rules', {}) if rules['ok'] else {}
    default_deny = root_rules.get('.read') is False and root_rules.get('.write') is False
    admins = [uid for uid, u in users.items() if isinstance(u, dict) and u.get('role') == 'admin']
    admin_configured = ADMIN_UID not in ('', 'SET_IN_ENV_FILE')
    csrf = csrf_coverage()
    # POST routes that change nothing (a redirect and a 501 placeholder) aren't CSRF risks
    csrf['none'] = [r for r in csrf['none'] if r not in ('/admin/login', '/api/admin/assign-patient-to-doctor')]
    secret_weak = not SECRET_KEY_FROM_ENV or len(app.secret_key) < 32   # temporary fallback key, or too short
    limiter_active = bool(app.extensions.get('limiter')) and limiter.enabled and app.config.get('RATELIMIT_ENABLED', True)
    admin_profile = users.get(ADMIN_UID) if isinstance(users.get(ADMIN_UID), dict) else {}
    admin_mfa = admin_profile.get('mfa') if mfa_enrolled_profile(admin_profile) else None

    def item(key, label, points, status, detail):
        return {'key': key, 'label': label, 'points': points, 'status': status, 'detail': detail,
                'earned': points if status == 'pass' else points // 2 if status == 'partial' else 0}

    checks = [
        item('rules', 'Firebase rules deployed', 20,
             'pass' if rules['ok'] and default_deny and rules['matches_local'] else 'partial' if rules['ok'] and default_deny else 'fail',
             ('Live rules match database.rules.json and deny by default.' if rules['matches_local'] else
              'Rules are live and deny by default, but differ from database.rules.json: redeploy with firebase deploy --only database.')
             if rules['ok'] and default_deny else
             ("Couldn't read the deployed rules" + (f" ({rules['error']})" if rules['error'] else '') + '.' if not rules['ok'] else
              'Deployed rules do not deny access by default.')),
        item('https', 'HTTPS in production', 15, 'pass' if is_https else 'na' if is_local else 'fail',
             'Served over HTTPS.' if is_https else
             'Not applicable on the local development server. Scores once the app is served over HTTPS.' if is_local else
             'This page was served over plain HTTP.'),
        item('admin', 'Single admin account (by UID)', 15,
             'pass' if admin_configured and admins in ([ADMIN_UID], []) and len(admins) <= 1 else 'fail',
             (f'ADMIN_UID is set in .env and {len(admins)} account has the admin role.' if admin_configured and len(admins) <= 1 else
              'ADMIN_UID is not set in .env.' if not admin_configured else
              f'{len(admins)} accounts have the admin role; only ADMIN_UID should.')),
        item('approval', 'Doctor approval workflow', 10, 'pass',
             'New doctors start as pending with sign-in disabled; /api/verify-token blocks anyone not approved.'),
        item('csrf', 'CSRF protection', 10, 'pass' if not csrf['none'] else 'partial',
             f"{csrf['token']} admin form routes check a per-session token; {csrf['json']} API routes accept JSON only; "
             f"the wristband endpoint uses a device key." + (f" Unprotected: {', '.join(csrf['none'])}." if csrf['none'] else '')),
        item('password', 'Password minimum length', 5, 'pass',
             'Firebase Auth requires at least 6 characters; admin account forms enforce the same.'),
        item('ratelimit', 'Rate limiting on sign-in', 10, 'pass' if limiter_active else 'fail',
             f"Flask-Limiter is active: sign-in {RATE_LIMITS['verify_token']}, 2FA codes {RATE_LIMITS['admin_mfa_verify']}, "
             f"default 200/day and 50/hour per route per IP." if limiter_active else 'Not active: nothing limits repeated sign-in or API attempts.'),
        item('mfa', 'Two-factor authentication for admin', 15, 'pass' if admin_mfa else 'fail',
             f"Authenticator app (TOTP) enrolled {time_ago(admin_mfa.get('enrolled_at'))}; "
             f"{len(admin_mfa.get('backup_codes') or {})} of {MFA_BACKUP_COUNT} backup codes unused." if admin_mfa else
             'Not enrolled: the admin signs in with a password only. Set it up below.'),
    ]
    score = sum(c['earned'] for c in checks)
    na_points = sum(c['points'] for c in checks if c['status'] == 'na')

    # Recent activity log (sessions + failed sign-ins)
    raw = db.reference('activity_logs').order_by_key().limit_to_last(1000).get() or {}
    logs = sorted((v for v in raw.values() if isinstance(v, dict) and isinstance(v.get('timestamp'), (int, float))),
                  key=lambda v: v['timestamp'], reverse=True)
    last_login = {}
    for v in logs:
        if v.get('action') == 'login' and v.get('user_uid') not in last_login:
            last_login[v['user_uid']] = v

    presence = db.reference('presence').get() or {}
    sessions = []
    for uid, seen in (presence.items() if isinstance(presence, dict) else []):
        if not isinstance(seen, (int, float)) or now_ms - seen > ACTIVE_SESSION_MS:
            continue
        user = users.get(uid) if isinstance(users.get(uid), dict) else {}
        login = last_login.get(uid)
        sessions.append({'name': user.get('name') or 'Unknown user', 'role': user.get('role', '—'),
                         'last_active': time_ago(seen), 'last_active_full': format_timestamp(seen, '%d %b %Y, %H:%M:%S'),
                         'ip': display_ip(login.get('ip', '')) if login else '—', 'device': login.get('user_agent_summary') if login else '—',
                         'duration': format_duration(now_ms - login['timestamp']) if login else '—',
                         'since': format_timestamp(login['timestamp'], '%d %b, %H:%M') if login else ''})
    sessions.sort(key=lambda s: s['name'].lower())

    failed_all = [v for v in logs if v.get('action') == 'login_failed']
    day_ago = now_ms - 86_400_000
    ip_counts = {}
    for v in failed_all:
        if v['timestamp'] >= day_ago:
            ip_counts[v.get('ip', '')] = ip_counts.get(v.get('ip', ''), 0) + 1
    reasons = {'not_registered': 'No registered profile', 'doctor_pending': 'Doctor pending approval',
               'doctor_rejected': 'Doctor rejected', 'doctor_suspended': 'Doctor suspended'}
    def reason_label(v):
        code = str((v.get('details') or {}).get('reason') or '')
        return reasons.get(code) or ('Token rejected' if code.endswith('Error') or not code else code)
    failed = [{'when': format_log_time(v['timestamp']), 'full': format_timestamp(v['timestamp'], '%d %b %Y, %H:%M:%S'),
               'email': (v.get('details') or {}).get('email') or '—', 'reason': reason_label(v),
               'ip': display_ip(v.get('ip', '')), 'device': v.get('user_agent_summary') or '—',
               'suspicious': ip_counts.get(v.get('ip', ''), 0) >= 3 and v['timestamp'] >= day_ago}
              for v in failed_all[:20]]
    suspicious_ips = sorted(display_ip(ip) for ip, n in ip_counts.items() if n >= 3)

    # Access control audit
    patients = {uid: u for uid, u in users.items() if isinstance(u, dict) and u.get('role') == 'patient'}
    doctors = {uid: u for uid, u in users.items() if isinstance(u, dict) and u.get('role') == 'doctor'}
    doctor_status = {s: sum(1 for d in doctors.values() if d.get('status') == s) for s in ('approved', 'pending', 'rejected', 'suspended')}
    load = {}
    unassigned, bad_assignment = [], []
    for uid, p in patients.items():
        if p.get('status') != 'active':
            continue
        doc = p.get('assigned_doctor_uid')
        if not doc:
            unassigned.append(p.get('name') or uid)
        elif not isinstance(doctors.get(doc), dict) or doctors[doc].get('status') != 'approved':
            bad_assignment.append(p.get('name') or uid)
        else:
            load[doc] = load.get(doc, 0) + 1
    overloaded = [{'name': strip_dr(doctors[d].get('name')) or d, 'count': n} for d, n in load.items() if n > DOCTOR_PATIENT_LIMIT]
    access = {
        'patients': len(patients), 'patients_active': sum(1 for p in patients.values() if p.get('status') == 'active'),
        'patients_suspended': sum(1 for p in patients.values() if p.get('status') == 'suspended'),
        'doctors': len(doctors), 'doctor_status': doctor_status, 'admins': len(admins),
        'unassigned': unassigned, 'bad_assignment': bad_assignment, 'overloaded': overloaded,
        'load': sorted(({'name': strip_dr(doctors[d].get('name')) or d, 'count': n} for d, n in load.items()), key=lambda x: -x['count']),
    }

    # Data-at-rest checks for the recommendations list
    stored_passwords = sum(1 for u in users.values() if isinstance(u, dict) and any('password' in str(k).lower() for k in u))
    devices = db.reference('wristband_devices').get() or {}
    devices = devices if isinstance(devices, dict) else {}
    hashed_ok = all(isinstance(d, dict) and len(str(d.get('api_key_hash', ''))) == 64 and 'api_key' not in d for d in devices.values())
    exprs, rule_paths = count_rule_expressions(root_rules)
    passed = {c['key']: c['status'] == 'pass' for c in checks}
    recommendations = [
        (rules['ok'] and default_deny, 'Firebase security rules deployed', 'Deny-by-default rules are live.' if rules['ok'] and default_deny else 'Deploy database.rules.json.'),
        (stored_passwords == 0, 'Passwords never stored in plain text', 'Firebase Auth holds the password hashes; no user record has a password field.'
         if stored_passwords == 0 else f'{stored_passwords} user record(s) contain a password field.'),
        (hashed_ok, 'Wristband API keys stored as SHA-256 hashes', f'All {len(devices)} paired device(s) store only a 64-character hash.' if hashed_ok
         else 'A device record holds something other than a SHA-256 hash.'),
        (bool(logs), 'Activity audit logging enabled', f'{len(logs)} recent entries; latest {time_ago(logs[0]["timestamp"])}.' if logs else 'No entries logged yet.'),
        (not secret_weak, 'Flask secret key kept out of the source code',
         f'Loaded from .env (SECRET_KEY, {len(app.secret_key)} characters). The old hard-coded key is still in git history, '
         'so it must never be reused.' if not secret_weak else
         'SECRET_KEY is missing from .env (or shorter than 32 characters), so a temporary random key is in use and everyone '
         'is signed out whenever the server restarts.'),
        (not app.debug or is_local, 'Debug mode off in production', 'Debug mode is on (fine on this local machine; never enable it on a public server).'
         if app.debug else 'Debug mode is off.'),
        (passed['https'], 'Serve over HTTPS with Secure cookies', 'Set SESSION_COOKIE_SECURE once deployed on HTTPS.'),
        (passed['ratelimit'], 'Rate limiting on login' if passed['ratelimit'] else 'Implement rate limiting on login',
         'Active on sign-in, registration, 2FA, pairing and the device endpoint. Uses in-memory counters: use Redis with several workers.'
         if passed['ratelimit'] else 'e.g. Flask-Limiter on /api/verify-token and the device endpoint.'),
        (passed['mfa'], '2FA enabled for the admin account' if passed['mfa'] else 'Add 2FA for the admin account',
         'Authenticator-app codes are required after the password; backup codes are stored hashed.' if passed['mfa']
         else 'Set up an authenticator app from the 2FA panel on this page.'),
        (False, 'Enable Firebase App Check', 'Not configured: it lets Firebase reject requests that don\'t come from this app.'),
        (False, 'Set up automated security scanning', 'e.g. pip-audit or GitHub Dependabot for dependency vulnerabilities.'),
    ]
    policies = {
        'password_min': 6, 'admin_uid': mask_uid(ADMIN_UID) if admin_configured else 'Not set',
        'cookie_httponly': app.config.get('SESSION_COOKIE_HTTPONLY'), 'cookie_secure': app.config.get('SESSION_COOKIE_SECURE'),
        'cookie_samesite': app.config.get('SESSION_COOKIE_SAMESITE'), 'rules_exprs': exprs, 'rules_paths': rule_paths,
        'rules_ok': rules['ok'], 'rules_match': rules['matches_local'],
    }
    policies['rate_limits'] = [
        ('Sign-in (/api/verify-token)', RATE_LIMITS['verify_token']), ('Admin 2FA code (/admin/mfa-verify)', RATE_LIMITS['admin_mfa_verify']),
        ('Registration (/api/register-profile)', RATE_LIMITS['register_profile']), ('Wristband pairing', RATE_LIMITS['wristband_pair']),
        ('Wristband data (ESP32)', RATE_LIMITS['wristband_data']), ('Facial scan frames', RATE_LIMITS['analyze_emotion']),
        ('2FA setup / changes', f"{RATE_LIMITS['admin_mfa_enroll']} · {RATE_LIMITS['admin_mfa_manage']}"),
        ('Every other route', '200 per day, 50 per hour'),
    ] if limiter_active else []
    mfa_panel = {'enrolled': bool(admin_mfa), 'enrolled_ago': time_ago(admin_mfa.get('enrolled_at')) if admin_mfa else '',
                 'backup_left': len(admin_mfa.get('backup_codes') or {}) if admin_mfa else 0,
                 'is_admin_uid': (session.get('user') or {}).get('uid') == ADMIN_UID}
    return {'checks': checks, 'score': score, 'na_points': na_points, 'secret_weak': secret_weak, 'mfa_panel': mfa_panel,
            'sessions': sessions, 'failed': failed, 'failed_total': len(failed_all), 'suspicious_ips': suspicious_ips,
            'access': access, 'recommendations': recommendations, 'policies': policies, 'users': users,
            'failed_all': failed_all, 'reason_label': reason_label}


@app.route('/admin/security')
@admin_required
def admin_security():
    data = security_data()
    data.pop('failed_all'); data.pop('reason_label')
    return admin_page('admin_security.html', 'security', data.pop('users'), **data, doctor_limit=DOCTOR_PATIENT_LIMIT)


@app.route('/admin/security/failed-logins.csv')
@admin_required
def admin_failed_logins_csv():
    import csv
    import io
    data = security_data()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['Time', 'Email', 'Reason', 'Action', 'IP', 'Device', 'User role'])
    for v in data['failed_all']:
        details = v.get('details') or {}
        writer.writerow([format_timestamp(v['timestamp'], '%Y-%m-%d %H:%M:%S'), csv_cell(details.get('email', '')),
                         csv_cell(data['reason_label'](v)), csv_cell(v.get('action_label', '')),
                         csv_cell(display_ip(v.get('ip', ''))), csv_cell(v.get('user_agent_summary', '')), v.get('user_role', '')])
    return app.response_class('﻿' + out.getvalue(), mimetype='text/csv', headers={
        'Content-Disposition': f"attachment; filename=failed_logins_{time.strftime('%Y-%m-%d')}.csv"})


# ========== ADMIN TWO-FACTOR AUTHENTICATION (TOTP) ==========
# users/{ADMIN_UID}/mfa: totp_secret_enc (Fernet-encrypted with a key derived from SECRET_KEY),
#   enrolled_at, backup_codes/{pbkdf2 hash}: true (8 one-time codes), last_step (anti-replay).
# Admin only. If SECRET_KEY changes, the authenticator secret can't be decrypted any more: sign in
# with a backup code, then disable and set up 2FA again.
MFA_ISSUER = 'Mental Disorder Detection'
MFA_PENDING_SECONDS = 300
MFA_BACKUP_COUNT = 8
MFA_CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'   # no 0/O or 1/I confusion
RATE_LIMITS.update({'admin_mfa_enroll': '10 per 15 minutes', 'admin_mfa_manage': '5 per 15 minutes'})


def _mfa_fernet():
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=b'mdd-admin-mfa', info=b'totp-secret').derive(app.secret_key.encode())
    return Fernet(base64.urlsafe_b64encode(key))


def mfa_encrypt(secret):
    return _mfa_fernet().encrypt(secret.encode()).decode()


def mfa_decrypt(token):
    """The TOTP secret, or None if it can't be decrypted (e.g. SECRET_KEY changed)."""
    try:
        return _mfa_fernet().decrypt(token.encode()).decode()
    except Exception:
        return None


def mfa_record(uid=None):
    node = db.reference(f'users/{uid or ADMIN_UID}/mfa').get()
    return node if isinstance(node, dict) and node.get('totp_secret_enc') else None


def mfa_enrolled_profile(profile):
    return isinstance(profile, dict) and isinstance(profile.get('mfa'), dict) and bool(profile['mfa'].get('totp_secret_enc'))


def backup_code_hash(code, uid):
    """Slow salted hash, so the stored values can't be brute-forced back into codes."""
    return hashlib.pbkdf2_hmac('sha256', code.encode(), f'{uid}:mfa-backup'.encode(), 120_000).hex()


def normalize_backup_code(text):
    return ''.join(ch for ch in str(text).upper() if ch in MFA_CODE_ALPHABET)


def new_backup_codes(uid):
    """(codes to show once, {hash: True} to store)."""
    codes = [''.join(secrets.choice(MFA_CODE_ALPHABET) for _ in range(10)) for _ in range(MFA_BACKUP_COUNT)]
    return [f'{c[:5]}-{c[5:]}' for c in codes], {backup_code_hash(c, uid): True for c in codes}


def totp_step(secret, code, last_step=0):
    """The 30-second step the code belongs to (now ±1 step for clock drift), or None. Steps at or
    before last_step are rejected so a code can't be replayed."""
    import pyotp
    code = ''.join(ch for ch in str(code) if ch.isdigit())
    if len(code) != 6 or not secret:
        return None
    totp = pyotp.TOTP(secret)
    now = time.time()
    for offset in (0, -1, 1):
        when = now + offset * 30
        if hmac.compare_digest(totp.at(when), code):
            step = int(when // 30)
            return step if step > (last_step or 0) else None
    return None


def mfa_check_code(uid, record, code):
    """('totp' | 'backup' | None, detail). Consumes a backup code or records the TOTP step."""
    text = str(code or '').strip()
    if len(''.join(ch for ch in text if ch.isdigit())) == 6 and not any(ch.isalpha() for ch in text):
        secret = mfa_decrypt(record.get('totp_secret_enc', ''))
        if secret is None:
            return None, 'Authenticator codes can\'t be checked because the server key changed. Use a backup code.'
        step = totp_step(secret, text, record.get('last_step', 0))
        if step is None:
            return None, 'That code is wrong or has already been used.'
        db.reference(f'users/{uid}/mfa/last_step').set(step)
        return 'totp', ''
    normalized = normalize_backup_code(text)
    if len(normalized) != 10:
        return None, 'Enter the 6-digit code from your authenticator app, or a 10-character backup code.'
    digest = backup_code_hash(normalized, uid)
    ref = db.reference(f'users/{uid}/mfa/backup_codes/{digest}')
    if ref.get() is not True:
        return None, 'That backup code is wrong or has already been used.'
    ref.delete()   # one use only
    return 'backup', ''


def mfa_admin_only(view):
    """The configured admin account only (not just any session with the admin role)."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        user = session.get('user') or {}
        if user.get('role') != 'admin' or user.get('uid') != ADMIN_UID:
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapper


def mfa_qr_svg(uri):
    import qrcode
    import qrcode.image.svg
    img = qrcode.make(uri, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
    return img.to_string(encoding='unicode')


def mfa_page(mode, **context):
    users = admin_load_users()
    record = context.pop('record', None) or mfa_record()
    return admin_page('admin_mfa.html', 'security', users, mode=mode, record=record,
                      backup_left=len((record or {}).get('backup_codes') or {}),
                      enrolled_ago=time_ago((record or {}).get('enrolled_at')), key_from_env=SECRET_KEY_FROM_ENV, **context)


@app.route('/admin/security/mfa-setup')
@admin_required
@mfa_admin_only
def admin_mfa_setup():
    import pyotp
    if mfa_record():
        return mfa_page('manage')
    if not SECRET_KEY_FROM_ENV:
        return mfa_page('nokey')
    secret = session.get('_mfa_setup_secret') or pyotp.random_base32()
    session['_mfa_setup_secret'] = secret   # kept until enrolment succeeds (signed cookie, admin's own browser)
    email = session['user'].get('email') or 'admin'
    uri = pyotp.TOTP(secret).provisioning_uri(name=email, issuer_name=MFA_ISSUER)
    return mfa_page('setup', qr_svg=mfa_qr_svg(uri), manual=' '.join(secret[i:i + 4] for i in range(0, len(secret), 4)))


@app.route('/admin/security/mfa-enroll', methods=['POST'])
@limiter.limit(RATE_LIMITS['admin_mfa_enroll'])
@admin_required
@mfa_admin_only
def admin_mfa_enroll():
    import pyotp
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400
    if mfa_record():
        return redirect(url_for('admin_mfa_setup'))
    secret = session.get('_mfa_setup_secret')
    if not secret or not SECRET_KEY_FROM_ENV:
        return redirect(url_for('admin_mfa_setup'))
    if totp_step(secret, request.form.get('code', '')) is None:
        uri = pyotp.TOTP(secret).provisioning_uri(name=session['user'].get('email') or 'admin', issuer_name=MFA_ISSUER)
        return mfa_page('setup', qr_svg=mfa_qr_svg(uri), manual=' '.join(secret[i:i + 4] for i in range(0, len(secret), 4)),
                        error='That code didn\'t match. Check the time on your phone is set automatically, then try the newest code.'), 400
    uid = session['user']['uid']
    codes, hashed = new_backup_codes(uid)
    record = {'totp_secret_enc': mfa_encrypt(secret), 'enrolled_at': {'.sv': 'timestamp'}, 'backup_codes': hashed,
              'last_step': int(time.time() // 30)}
    db.reference(f'users/{uid}/mfa').set(record)
    session.pop('_mfa_setup_secret', None)
    activity_log('mfa_enrolled', 'Enabled two-factor authentication', 'success', details={'method': 'TOTP authenticator', 'backup_codes': MFA_BACKUP_COUNT})
    return mfa_page('codes', codes=codes, record=mfa_record(uid), first_time=True)


@app.route('/admin/security/mfa-backup-codes', methods=['POST'])
@limiter.limit(RATE_LIMITS['admin_mfa_manage'])
@admin_required
@mfa_admin_only
def admin_mfa_regenerate():
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400
    uid = session['user']['uid']
    record = mfa_record(uid)
    if not record:
        return redirect(url_for('admin_mfa_setup'))
    method, error = mfa_check_code(uid, record, request.form.get('code', ''))
    if not method:
        activity_log('mfa_failed', 'Wrong 2FA code when regenerating backup codes', 'failed')
        return mfa_page('manage', error=error, error_form='regenerate'), 400
    codes, hashed = new_backup_codes(uid)
    db.reference(f'users/{uid}/mfa/backup_codes').set(hashed)
    activity_log('mfa_backup_codes_regenerated', 'Regenerated 2FA backup codes', 'success', details={'verified_with': method})
    return mfa_page('codes', codes=codes)


@app.route('/admin/security/mfa-disable', methods=['POST'])
@limiter.limit(RATE_LIMITS['admin_mfa_manage'])
@admin_required
@mfa_admin_only
def admin_mfa_disable():
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400
    uid = session['user']['uid']
    record = mfa_record(uid)
    if not record:
        return redirect(url_for('admin_security'))
    method, error = mfa_check_code(uid, record, request.form.get('code', ''))
    if not method:
        activity_log('mfa_failed', 'Wrong 2FA code when trying to disable 2FA', 'failed')
        return mfa_page('manage', error=error, error_form='disable'), 400
    db.reference(f'users/{uid}/mfa').delete()
    activity_log('mfa_disabled', 'Disabled two-factor authentication', 'alert', details={'verified_with': method})
    flash('Two-factor authentication is off. Your next sign-in needs only a password.', 'success')
    return redirect(url_for('admin_security'))


@app.route('/admin/mfa-verify', methods=['GET', 'POST'])
@limiter.limit(RATE_LIMITS['admin_mfa_verify'], methods=['POST'])
def admin_mfa_verify():
    """Second sign-in step for the admin. session['user'] is only created once this passes."""
    pending = session.get('_mfa_pending')
    if not isinstance(pending, dict) or time.time() - pending.get('at', 0) > MFA_PENDING_SECONDS:
        session.pop('_mfa_pending', None)
        if pending:
            flash('The sign-in took too long. Please sign in again.', 'error')
        return redirect(url_for('login'))
    user = pending['user']
    csrf_token = session.setdefault('_mfa_csrf', secrets.token_urlsafe(32))
    if request.method == 'GET':
        return render_template('admin_mfa_verify.html', csrf_token=csrf_token, email=user.get('email', ''))

    if not hmac.compare_digest(request.form.get('csrf_token', ''), csrf_token):
        return 'Invalid or expired form token.', 400
    record = mfa_record(user['uid'])
    if not record:
        # 2FA was removed meanwhile (e.g. in the Firebase console): fall back to the password sign-in
        method, error = 'none', ''
    else:
        method, error = mfa_check_code(user['uid'], record, request.form.get('code', ''))
    if not method:
        activity_log('mfa_failed', 'Wrong 2FA code at sign-in', 'failed', details={'email': user.get('email', '')}, user=user)
        return render_template('admin_mfa_verify.html', csrf_token=csrf_token, email=user.get('email', ''), error=error), 400

    session.pop('_mfa_pending', None)
    session.pop('_mfa_csrf', None)
    session['user'] = user
    session['_login_toast'] = True
    try:
        db.reference(f"login_counts/{time.strftime('%Y-%m-%d')}").set({'.sv': {'increment': 1}})
    except Exception:
        pass
    left = len((mfa_record(user['uid']) or {}).get('backup_codes') or {}) if method == 'backup' else None
    activity_log('mfa_success', '2FA code accepted' + (' (backup code)' if method == 'backup' else ''),
                 details={'method': method, 'backup_codes_left': left})
    activity_log('login', 'Logged in', details={'email': user.get('email', ''), 'second_factor': method})
    if method == 'backup':
        flash(f'Signed in with a backup code. {left} backup code{"s" if left != 1 else ""} left.'
              + (' Generate new ones on the Security page.' if left <= 3 else ''), 'success' if left > 3 else 'error')
    return redirect(url_for('admin_dashboard'))


# ========== ADMIN DATABASE BROWSER (read-only) ==========
# Only Firebase reads happen here: no route in this section writes or deletes anything.
DB_CHILD_LIMIT = 100          # children shown per object
DB_RENDER_BUDGET = 2500       # total rows rendered in one view
DB_RENDER_DEPTH = 6           # nested levels shown inline before "open" links
DB_SEARCH_LIMIT = 50
DB_VALUE_PREVIEW = 500
DB_SECRET_KEY = None          # compiled lazily below


def db_secret_key(key):
    """Keys whose values are never shown: API key hashes, passwords, tokens, private keys, credentials."""
    import re
    global DB_SECRET_KEY
    if DB_SECRET_KEY is None:
        DB_SECRET_KEY = re.compile(r'api_?key|password|passwd|secret|token|private_?key|credential', re.I)
    return bool(DB_SECRET_KEY.search(str(key)))


# Patient clinical content, masked by path ('*' matches any one key). Scores and metadata stay visible.
DB_PRIVATE_PATHS = (
    (('chats', '*', 'messages', '*', 'text'), '***message hidden***'),
    (('chats', '*', 'last_message', 'text'), '***message hidden***'),   # copy of the newest message
    (('clinical_notes', '*', '*', 'body'), '***note hidden***'),        # the note text
    (('clinical_notes', '*', '*', 'title'), '***note hidden***'),
    (('users', '*', 'clinical_notes', '*', 'body'), '***note hidden***'),   # older notes, stored under the patient
    (('users', '*', 'clinical_notes', '*', 'title'), '***note hidden***'),
    (('alerts', '*', '*', 'note'), '***note hidden***'),                   # doctor's note when acknowledging an alert
    (('users', '*', 'mfa', 'backup_codes'), '***hidden***'),              # hashed 2FA backup codes (totp_secret_enc is masked by key)
    (('users', '*', 'gratitude', '*', 'text'), '***gratitude hidden***'),   # Gratitude Jar notes (patient private)
    (('users', '*', 'journal', '*', 'text'), '***journal hidden***'),       # journal entries (private unless shared)
    (('users', '*', 'assessments', '*', 'phq_answers'), '***answers hidden***'),
    (('users', '*', 'assessments', '*', 'gad_answers'), '***answers hidden***'),
    (('users', '*', 'assessments', '*', 'pss_answers'), '***answers hidden***'),
)


def db_mask_label(path):
    """The placeholder for a masked path ('a/b/c'), or None if it may be shown.
    Secret-looking keys anywhere, plus the clinical-content paths above."""
    parts = [p for p in str(path).split('/') if p]
    if parts and db_secret_key(parts[-1]):
        return '***hidden***'
    for pattern, label in DB_PRIVATE_PATHS:
        if len(parts) == len(pattern) and all(p == '*' or p == s for p, s in zip(pattern, parts)):
            return label
    return None


def db_mask(value, path=''):
    """Copy of `value` (stored at `path`) with every masked path replaced by its placeholder."""
    label = db_mask_label(path) if path else None
    if label:
        return label
    if isinstance(value, dict):
        return {k: db_mask(v, f'{path}/{k}') for k, v in value.items()}
    if isinstance(value, list):
        return [db_mask(v, f'{path}/{i}') for i, v in enumerate(value)]
    return value


def db_clean_path(raw):
    """('a/b/c' segments list, error). Firebase keys can't contain . $ # [ ] / or control characters."""
    import re
    text = (raw or '').strip().strip('/')
    if not text:
        return [], None
    parts = [p for p in text.split('/') if p]
    if len(parts) > 32 or any(not re.fullmatch(r'[^.$#\[\]/\x00-\x1f\x7f]{1,256}', p) for p in parts):
        return [], 'That isn\'t a valid database path. Keys can\'t contain . $ # [ ] or /.'
    return parts, None


def db_view(value, path, key='', depth=0, budget=None):
    """A render-ready tree for templates/admin_database.html, capped in size."""
    budget = budget if budget is not None else {'left': DB_RENDER_BUDGET}
    label = db_mask_label(path) if path else None
    if label:
        return {'type': 'masked', 'label': label}
    if isinstance(value, (dict, list)):
        items_src = list(value.items()) if isinstance(value, dict) else list(enumerate(value))
        items_src = [(k, v) for k, v in items_src if v is not None]   # Firebase arrays can have holes
        node = {'type': 'array' if isinstance(value, list) else 'object', 'count': len(items_src), 'path': path}
        if depth >= DB_RENDER_DEPTH or budget['left'] <= 0:
            node['stub'] = True
            return node
        node['items'] = []
        for k, v in items_src[:DB_CHILD_LIMIT]:
            if budget['left'] <= 0:
                break
            budget['left'] -= 1
            node['items'].append((str(k), db_view(v, f'{path}/{k}'.strip('/'), str(k), depth + 1, budget)))
        node['hidden'] = len(items_src) - len(node['items'])
        return node
    if isinstance(value, bool):
        return {'type': 'boolean', 'value': 'true' if value else 'false'}
    if isinstance(value, (int, float)):
        return {'type': 'number', 'value': value}
    if value is None:
        return {'type': 'null'}
    text = str(value)
    return {'type': 'string', 'value': text, 'long': len(text) > DB_VALUE_PREVIEW, 'preview': text[:DB_VALUE_PREVIEW]}


def db_search(value, pattern, path, results, key=''):
    """Depth-first key search (substring, or * / ? wildcards), up to DB_SEARCH_LIMIT matches."""
    import fnmatch
    if len(results) >= DB_SEARCH_LIMIT or not isinstance(value, (dict, list)):
        return
    items = value.items() if isinstance(value, dict) else enumerate(value)
    wildcard = any(ch in pattern for ch in '*?')
    for k, v in items:
        if len(results) >= DB_SEARCH_LIMIT:
            return
        if v is None:
            continue
        child = f'{path}/{k}'.strip('/')
        name = str(k).lower()
        label = db_mask_label(child)
        if (fnmatch.fnmatch(name, pattern) if wildcard else pattern in name):
            if label:
                preview = label
            elif isinstance(v, dict):
                preview = f'{{{len(v)} keys}}'
            elif isinstance(v, list):
                preview = f'[{len(v)} items]'
            else:
                preview = str(v)[:120]
            results.append({'path': child, 'key': str(k), 'preview': preview})
        if not label:   # never look inside masked content
            db_search(v, pattern, child, results)


def firebase_console_url(path=''):
    """Deep link to this path in the Firebase console's Realtime Database viewer."""
    from urllib.parse import quote, urlparse
    project = FIREBASE_WEB_CONFIG['projectId']
    instance = urlparse(FIREBASE_WEB_CONFIG['databaseURL']).hostname.split('.')[0]
    encoded = quote('/' + path, safe='').replace('%2F', '~2F')
    return f'https://console.firebase.google.com/project/{project}/database/{instance}/data/{encoded}'


@app.route('/admin/database')
@admin_required
def admin_database():
    import json
    from concurrent.futures import ThreadPoolExecutor
    parts, path_error = db_clean_path(request.args.get('path'))
    path = '/'.join(parts)
    query = (request.args.get('q') or '').strip()[:100]
    context = {'path': path, 'parts': parts, 'path_error': path_error, 'query': query,
               'project_id': FIREBASE_WEB_CONFIG['projectId'], 'console_url': firebase_console_url(path),
               'child_limit': DB_CHILD_LIMIT, 'unreachable': False}
    users = {}
    try:
        top = db.reference('/').get(shallow=True) or {}
        top = top if isinstance(top, dict) else {}

        def count(name):
            children = db.reference(name).get(shallow=True)
            return len(children) if isinstance(children, dict) else (0 if children is None else 1)
        with ThreadPoolExecutor(max_workers=8) as pool:
            counts = dict(zip(top, pool.map(count, top)))
        presence = db.reference('presence').get() or {}
        presence = presence if isinstance(presence, dict) else {}
        now_ms = time.time() * 1000
        context.update(
            tree=[{'name': name, 'count': counts.get(name, 0)} for name in sorted(top)],
            stats={'nodes': len(top), 'users': counts.get('users', 0), 'presence': len(presence),
                   'active': sum(1 for v in presence.values() if isinstance(v, (int, float)) and now_ms - v <= ACTIVE_SESSION_MS)})

        if not parts:
            # Root: one level only (whole-database reads are left to export / size / search)
            context['node'] = {'type': 'object', 'path': '', 'count': len(top), 'hidden': 0,
                               'items': [(name, {'type': 'object', 'count': counts.get(name, 0), 'path': name, 'stub': True})
                                         for name in sorted(top)]}
            context['total_children'] = len(top)
            data = None
        else:
            ref = db.reference(path)
            shallow = ref.get(shallow=True)
            total = len(shallow) if isinstance(shallow, dict) else None
            if total is not None and total > DB_CHILD_LIMIT:
                data = ref.order_by_key().limit_to_first(DB_CHILD_LIMIT).get()
            else:
                data = ref.get() if isinstance(shallow, dict) else shallow
            context['total_children'] = total
            context['exists'] = shallow is not None
            context['node'] = db_view(data, path, parts[-1]) if shallow is not None else None
            context['size_bytes'] = len(json.dumps(db_mask(data, path), ensure_ascii=False, default=str).encode()) if data is not None else 0
            context['size_partial'] = total is not None and total > DB_CHILD_LIMIT
        if query:
            # Search covers the whole current node, not just the first 100 children
            source = db.reference(path or '/').get() if (parts and context.get('size_partial')) or not parts else data
            results = []
            db_search(source, query.lower(), path, results)
            context['results'] = results
        users = admin_load_users()   # sidebar's pending-approvals count, like the other admin pages
    except Exception:
        context['unreachable'] = True
    return admin_page('admin_database.html', 'database', users, **context)


@app.route('/admin/database/export')
@admin_required
def admin_database_export():
    import json
    parts, error = db_clean_path(request.args.get('path'))
    if error:
        return error, 400
    path = '/'.join(parts)
    try:
        data = db.reference(path or '/').get()
    except Exception:
        return 'Database unreachable. Try again in a moment.', 503
    body = json.dumps(db_mask(data, path), indent=2, ensure_ascii=False, default=str)
    name = '_'.join(parts) or 'root'
    # Data leaving the system is audited (browsing isn't)
    activity_log('database_export', f"Exported database node /{path}", 'info',
                 details={'path': '/' + path, 'bytes': len(body.encode())})
    return app.response_class(body, mimetype='application/json', headers={
        'Content-Disposition': f'attachment; filename=db_{name}_{time.strftime("%Y-%m-%d")}.json'})


@app.route('/admin/database/size')
@admin_required
def admin_database_size():
    """Approximate size: the JSON length of the whole database (downloads everything, so on demand only)."""
    import json
    try:
        data = db.reference('/').get()
    except Exception:
        return jsonify({'success': False, 'error': 'Database unreachable.'}), 503
    size = len(json.dumps(data, ensure_ascii=False, default=str).encode())
    return jsonify({'success': True, 'bytes': size})


# ========== ADMIN ACTIVITY LOGS ==========
LOGS_PAGE_SIZE = 50
LOG_PERIODS = (('all', 'All time'), ('today', 'Today'), ('week', 'This week'), ('month', 'This month'))
PUSH_CHARS = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz'


def push_key_prefix(ms):
    """The 8-character time prefix of a Firebase push id created at `ms`, for key-range queries
    (push ids sort by creation time, so no database index is needed)."""
    ms, chars = int(max(ms, 0)), []
    for _ in range(8):
        ms, r = divmod(ms, 64)
        chars.append(PUSH_CHARS[r])
    return ''.join(reversed(chars))


def format_log_time(ms):
    """'10:00 AM' today, 'Yesterday 3:05 PM', otherwise '05 Nov 2026, 3:05 PM'."""
    from datetime import datetime, timedelta
    if not isinstance(ms, (int, float)):
        return '—'
    when, now = datetime.fromtimestamp(ms / 1000), datetime.now()
    clock = when.strftime('%I:%M %p').lstrip('0')
    if when.date() == now.date():
        return clock
    if when.date() == (now - timedelta(days=1)).date():
        return f'Yesterday {clock}'
    return f"{when:%d %b %Y}, {clock}"


def display_ip(ip):
    return 'localhost' if ip in ('127.0.0.1', '::1', 'localhost') else (ip or '—')


def admin_activity_data(args):
    from datetime import datetime, timedelta
    query = (args.get('q') or '').strip()[:100]
    role = (args.get('role') or 'all').lower()
    role = role if role in ('all',) + LOG_ROLES else 'all'
    status = (args.get('status') or 'all').lower()
    status = status if status in ('all',) + LOG_STATUSES else 'all'
    period = (args.get('period') or 'all').lower()
    period = period if period in dict(LOG_PERIODS) else 'all'

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start = {'today': today, 'week': today - timedelta(days=today.weekday()), 'month': today.replace(day=1)}.get(period)
    ref = db.reference('activity_logs')
    if start:
        # Read only from (start - 1h) by push-id time, then filter on the server timestamp exactly
        raw = ref.order_by_key().start_at(push_key_prefix(start.timestamp() * 1000 - 3_600_000)).get() or {}
    else:
        raw = ref.get() or {}
    raw = raw if isinstance(raw, dict) else {}
    lo = start.timestamp() * 1000 if start else None

    logs = []
    for key, e in raw.items():
        if not isinstance(e, dict) or not isinstance(e.get('timestamp'), (int, float)):
            continue
        if lo is not None and e['timestamp'] < lo:
            continue
        details = e.get('details') if isinstance(e.get('details'), dict) else {}
        detail_text = '; '.join(f"{k.replace('_', ' ')}: {v}" for k, v in details.items())
        logs.append({
            'id': key, 'timestamp': e['timestamp'], 'when': format_log_time(e['timestamp']),
            'full_time': format_timestamp(e['timestamp'], '%d %b %Y, %H:%M:%S'),
            'user_uid': e.get('user_uid', ''), 'user_name': e.get('user_name') or 'Unknown user',
            'role': e.get('user_role') if e.get('user_role') in LOG_ROLES else 'auto',
            'action': e.get('action', ''), 'label': e.get('action_label') or e.get('action', ''),
            'details': details, 'detail_text': detail_text,
            'ip': e.get('ip', ''), 'ip_display': display_ip(e.get('ip', '')),
            'device': e.get('user_agent_summary') or 'Unknown',
            'status': e.get('status') if e.get('status') in LOG_STATUSES else 'info',
        })
    logs.sort(key=lambda x: x['timestamp'], reverse=True)

    if query:
        q = query.lower()
        logs = [x for x in logs if q in x['user_name'].lower() or q in x['label'].lower() or q in x['action'].lower()
                or q in x['ip'].lower() or q in x['ip_display'].lower()]
    if role != 'all':
        logs = [x for x in logs if x['role'] == role]
    if status != 'all':
        logs = [x for x in logs if x['status'] == status]
    filters = {k: v for k, v in (('q', query), ('role', role), ('status', status), ('period', period)) if v and v != 'all'}
    return {'logs': logs, 'filters': filters, 'query': query, 'role': role, 'status': status, 'period': period,
            'period_label': dict(LOG_PERIODS)[period]}


def page_window(page, total, edge=1, around=2):
    """Page numbers to show, with None where a gap ('…') goes: 1 … 4 5 [6] 7 8 … 20."""
    shown = sorted({n for n in range(1, total + 1)
                    if n <= edge or n > total - edge or abs(n - page) <= around})
    out = []
    for n in shown:
        if out and n - out[-1] > 1:
            out.append(None)
        out.append(n)
    return out


@app.route('/admin/activity-logs')
@admin_required
def admin_activity_logs():
    data = admin_activity_data(request.args)
    logs = data.pop('logs')
    total_pages = max(1, -(-len(logs) // LOGS_PAGE_SIZE))
    try:
        page = min(max(int(request.args.get('page', 1)), 1), total_pages)
    except ValueError:
        page = 1
    start = (page - 1) * LOGS_PAGE_SIZE
    return admin_page('admin_activity_logs.html', 'activity', admin_load_users(), **data,
                      page_logs=logs[start:start + LOGS_PAGE_SIZE], match_total=len(logs), first_index=start + 1,
                      page=page, total_pages=total_pages, page_numbers=page_window(page, total_pages),
                      roles=('all',) + LOG_ROLES, statuses=('all',) + LOG_STATUSES, periods=LOG_PERIODS)


@app.route('/admin/activity-logs/export.csv')
@admin_required
def admin_activity_logs_csv():
    import csv
    import io
    data = admin_activity_data(request.args)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['Timestamp', 'User', 'User ID', 'Role', 'Action code', 'Action', 'Details', 'IP', 'Device', 'Status'])
    for x in data['logs']:
        writer.writerow([format_timestamp(x['timestamp'], '%Y-%m-%d %H:%M:%S'), csv_cell(x['user_name']), csv_cell(x['user_uid']),
                         x['role'], x['action'], csv_cell(x['label']), csv_cell(x['detail_text']), csv_cell(x['ip_display']),
                         csv_cell(x['device']), x['status']])
    return app.response_class('﻿' + out.getvalue(), mimetype='text/csv', headers={
        'Content-Disposition': f"attachment; filename=activity_logs_{time.strftime('%Y-%m-%d')}.csv"})


# ========== ADMIN ALERTS LOG ==========
# Every patient's alerts (patient_alerts(), the same source as /doctor/risk-alerts), across all doctors.
ADMIN_ALERTS_PAGE_SIZE = 25
ADMIN_ALERT_STATUSES = ('all', 'unread', 'acknowledged')
ADMIN_ALERT_SEVERITIES = ('all', 'critical', 'high')
ADMIN_ALERT_PERIODS = (('all', 'All time'), ('today', 'Today'), ('week', 'This week'), ('month', 'This month'))


def alert_display_id(patient_uid, assessment_id):
    """Short readable id: first 8 characters of the patient uid + the assessment time in base 36."""
    n, digits = int(assessment_id), ''
    while True:
        n, r = divmod(n, 36)
        digits = '0123456789abcdefghijklmnopqrstuvwxyz'[r] + digits
        if not n:
            break
    return f'AL-{patient_uid[:8]}-{digits}'.upper()


def admin_alerts_data(args):
    from datetime import datetime
    users = admin_load_users()
    acks = db.reference('alerts').get() or {}
    acks = acks if isinstance(acks, dict) else {}

    alerts = []
    for uid, user in users.items():
        if not isinstance(user, dict) or user.get('role') != 'patient' or not user.get('assessments'):
            continue
        doctor_uid = user.get('assigned_doctor_uid')
        doctor = users.get(doctor_uid) if isinstance(doctor_uid, str) else None
        doctor_name = f"Dr. {strip_dr(doctor.get('name')) or 'Unnamed'}" if isinstance(doctor, dict) else 'Unassigned'
        for a in patient_alerts(uid, user, acks.get(uid) if isinstance(acks.get(uid), dict) else {}):
            response_ms = a['ack_at'] - a['created_at'] if a['acknowledged'] and isinstance(a['ack_at'], (int, float)) else None
            a.update(display_id=alert_display_id(uid, a['assessment_id']), doctor_name=doctor_name,
                     patient_email=user.get('email', ''), ago=time_ago(a['created_at']),
                     response_ms=max(0, response_ms) if response_ms is not None else None,
                     response=format_duration(max(0, response_ms)) if response_ms is not None else '—',
                     ack_by_label=('' if not a['acknowledged'] else 'a doctor' if a['ack_by_name'] == 'a doctor'
                                   else f"Dr. {strip_dr(a['ack_by_name'])}"))
            alerts.append(a)
    alerts.sort(key=lambda a: a['created_at'], reverse=True)

    # Stats are system-wide, whatever filters are applied
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    midnight = today.timestamp() * 1000
    acknowledged = [a for a in alerts if a['acknowledged']]
    responses = [a['response_ms'] for a in acknowledged if a['response_ms'] is not None]
    avg_ms = sum(responses) / len(responses) if responses else None
    stats = {
        'total': len(alerts),
        'today': sum(1 for a in alerts if a['created_at'] >= midnight),
        'today_label': f'{today:%A}, {today.day} {today:%B %Y}',
        'critical_open': sum(1 for a in alerts if a['severity'] == 'critical' and not a['acknowledged']),
        'acknowledged': len(acknowledged),
        'ack_rate': round(len(acknowledged) / len(alerts) * 100) if alerts else 0,
        'avg_response': format_duration(avg_ms) if avg_ms is not None else None,
        'response_ok': avg_ms is not None and avg_ms < RESPONSE_TARGET_MIN * 60000,
    }

    query = (args.get('q') or '').strip()[:100]
    status = (args.get('status') or 'all').lower()
    status = status if status in ADMIN_ALERT_STATUSES else 'all'
    severity = (args.get('severity') or 'all').lower()
    severity = severity if severity in ADMIN_ALERT_SEVERITIES else 'all'
    period = (args.get('period') or 'all').lower()
    period = period if period in dict(ADMIN_ALERT_PERIODS) else 'all'

    shown = alerts_in_range(alerts, period, '', '')
    if query:
        # Patient name, or the alert id with or without "#" (the internal id works too)
        q = query.lower().lstrip('#')
        shown = [a for a in shown if q in a['patient_name'].lower() or q in a['display_id'].lower() or q in a['id'].lower()]
    if status != 'all':
        shown = [a for a in shown if a['acknowledged'] == (status == 'acknowledged')]
    if severity != 'all':
        shown = [a for a in shown if a['severity'] == severity]

    filters = {k: v for k, v in (('q', query), ('status', status), ('severity', severity), ('period', period))
               if v and v != 'all'}
    return {'alerts': shown, 'stats': stats, 'filters': filters, 'query': query, 'status': status,
            'severity': severity, 'period': period, 'users': users,
            'period_label': dict(ADMIN_ALERT_PERIODS)[period]}


@app.route('/admin/alerts')
@admin_required
def admin_alerts():
    data = admin_alerts_data(request.args)
    shown = data.pop('alerts')
    total_pages = max(1, -(-len(shown) // ADMIN_ALERTS_PAGE_SIZE))
    try:
        page = min(max(int(request.args.get('page', 1)), 1), total_pages)
    except ValueError:
        page = 1
    start = (page - 1) * ADMIN_ALERTS_PAGE_SIZE
    bands = risk_ranges()
    return admin_page('admin_alerts.html', 'alerts', data.pop('users'), **data,
                      page_alerts=shown[start:start + ADMIN_ALERTS_PAGE_SIZE], match_total=len(shown),
                      first_index=start + 1, page=page, total_pages=total_pages,
                      statuses=ADMIN_ALERT_STATUSES, severities=ADMIN_ALERT_SEVERITIES, periods=ADMIN_ALERT_PERIODS,
                      high_min=bands[2]['min'], critical_min=bands[3]['min'])


@app.route('/admin/alerts/export.csv')
@admin_required
def admin_alerts_csv():
    import csv
    import io
    data = admin_alerts_data(request.args)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['Alert ID', 'Patient', 'Patient email', 'Doctor', 'Type', 'Composite score', 'Risk', 'Created',
                     'Status', 'Acknowledged by', 'Acknowledged at', 'Response time', 'Response minutes', 'Note'])
    for a in data['alerts']:
        writer.writerow([
            a['display_id'], csv_cell(a['patient_name']), csv_cell(a['patient_email']), csv_cell(a['doctor_name']),
            'Composite Alert', a['score'], a['severity_label'], format_timestamp(a['created_at'], '%Y-%m-%d %H:%M'),
            'Acknowledged' if a['acknowledged'] else 'Unread', csv_cell(a['ack_by_label']),
            format_timestamp(a['ack_at'], '%Y-%m-%d %H:%M') if a['acknowledged'] else '',
            a['response'] if a['acknowledged'] else '',
            round(a['response_ms'] / 60000, 1) if a['response_ms'] is not None else '', csv_cell(a['note'])])
    return app.response_class('﻿' + out.getvalue(), mimetype='text/csv', headers={
        'Content-Disposition': f"attachment; filename=alerts_log_{time.strftime('%Y-%m-%d')}.csv"})


@app.route('/admin/alerts/export.pdf')
@admin_required
def admin_alerts_pdf():
    import io
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    data = admin_alerts_data(request.args)
    s, alerts = data['stats'], data['alerts']
    green, ink, muted, line = colors.HexColor('#0F7A5F'), colors.HexColor('#0F172A'), colors.HexColor('#64748B'), colors.HexColor('#E2E8F0')
    title = ParagraphStyle('t', fontName='Helvetica-Bold', fontSize=20, textColor=ink, leading=24)
    sub = ParagraphStyle('s', fontName='Helvetica', fontSize=10, textColor=muted, leading=14)
    h2 = ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=12, textColor=ink, leading=16, spaceBefore=10, spaceAfter=6)
    cell = ParagraphStyle('c', fontName='Helvetica', fontSize=8, textColor=ink, leading=10)
    width = landscape(A4)[0] - 30 * mm

    labels = {'q': 'Search', 'status': 'Status', 'severity': 'Severity', 'period': 'Date'}
    shown_filters = ', '.join(f"{labels[k]}: {data['period_label'] if k == 'period' else v}" for k, v in data['filters'].items())
    label = ParagraphStyle('l', fontName='Helvetica-Bold', fontSize=7.5, textColor=muted, leading=10)
    value = ParagraphStyle('v', fontName='Helvetica-Bold', fontSize=17, textColor=ink, leading=21)
    note = ParagraphStyle('n', fontName='Helvetica', fontSize=8, textColor=muted, leading=10)
    stat = lambda name, val, detail, colour='#0F172A': [Paragraph(name, label), Paragraph(f"<font color='{colour}'>{val}</font>", value),
                                                         Paragraph(detail, note)]
    cards = Table([[
        stat('TOTAL TODAY', s['today'], escape(s['today_label'])),
        stat('CRITICAL UNACKNOWLEDGED', s['critical_open'], 'Needs immediate attention' if s['critical_open'] else 'None waiting',
             '#DC2626' if s['critical_open'] else '#0F172A'),
        stat('ACKNOWLEDGED', s['acknowledged'], f"{s['ack_rate']}% acknowledgment rate ({s['total']} alerts)"),
        stat('AVG RESPONSE TIME', s['avg_response'] or '—', f'Target &lt;{RESPONSE_TARGET_MIN} min'),
    ]], colWidths=[width / 4] * 4)
    cards.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 0.5, line), ('INNERGRID', (0, 0), (-1, -1), 0.5, line),
                               ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
                               ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                               ('LEFTPADDING', (0, 0), (-1, -1), 10)]))

    sev_colour = {'critical': '#DC2626', 'high': '#EA580C'}
    rows = [['Alert ID', 'Patient', 'Doctor', 'Type', 'Trigger', 'Risk', 'Created', 'Status', 'Response']]
    for a in alerts:
        status = f"Ack by {escape(a['ack_by_label'])}" if a['acknowledged'] else "<font color='#DC2626'><b>Unread</b></font>"
        rows.append([a['display_id'], Paragraph(escape(a['patient_name']), cell), Paragraph(escape(a['doctor_name']), cell),
                     'Composite', f"{a['score']}/100",
                     Paragraph(f"<font color='{sev_colour[a['severity']]}'><b>{a['severity_label']}</b></font>", cell),
                     format_timestamp(a['created_at'], '%d %b %Y %H:%M'), Paragraph(status, cell), a['response']])
    table = Table(rows, repeatRows=1, colWidths=[w * mm for w in (44, 38, 37, 19, 18, 17, 30, 44, 19)])
    table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 8), ('FONT', (0, 1), (-1, -1), 'Helvetica', 8),
        ('FONT', (0, 1), (0, -1), 'Courier', 7.5),
        ('BACKGROUND', (0, 0), (-1, 0), green), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('LINEBELOW', (0, 0), (-1, -1), 0.4, line), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))

    story = [Paragraph('Alerts Log', title),
             Paragraph(f"Mental Disorder Detection · generated {format_timestamp(time.time() * 1000, '%d %b %Y, %H:%M')}", sub),
             Spacer(1, 6 * mm), cards,
             Paragraph(f"{len(alerts)} alert{'s' if len(alerts) != 1 else ''}" + (f" · {escape(shown_filters)}" if shown_filters else ' · all alerts'), h2)]
    story.append(table if alerts else Paragraph('No alerts match these filters.', sub))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                      bottomMargin=14 * mm, title='Alerts Log', author='Mental Disorder Detection').build(story)
    return app.response_class(buf.getvalue(), mimetype='application/pdf', headers={
        'Content-Disposition': f"attachment; filename=alerts_log_{time.strftime('%Y-%m-%d')}.pdf"})


@app.route('/admin/dashboard')
def admin_dashboard_legacy():
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/approvals')
@admin_required
def admin_approvals():
    users = admin_load_users()
    by_status = {'pending': [], 'approved': [], 'rejected': []}
    for uid, user in users.items():
        if not isinstance(user, dict) or user.get('role') != 'doctor' or user.get('status') not in by_status:
            continue
        row = admin_account_row(uid, user)
        profile = user.get('profile') if isinstance(user.get('profile'), dict) else {}
        reviewed_at = user.get('reviewed_at')
        row.update(specialization=profile.get('specialization', ''), affiliation=profile.get('affiliation', ''),
                   reviewed_at=reviewed_at if isinstance(reviewed_at, (int, float)) else 0,
                   reviewed_date=admin_date(reviewed_at))
        by_status[user['status']].append(row)

    by_status['pending'].sort(key=lambda d: d['created_at'], reverse=True)
    for status in ('approved', 'rejected'):
        # Doctors reviewed before reviewed_at was recorded fall back to their sign-up date
        by_status[status].sort(key=lambda d: d['reviewed_at'] or d['created_at'], reverse=True)

    return admin_page('admin_approvals.html', 'approvals', users,
                      pending_doctors=by_status['pending'],
                      approved_doctors=by_status['approved'][:10],
                      rejected_doctors=by_status['rejected'][:10])


@app.route('/admin/users')
@admin_required
def admin_accounts():
    users = admin_load_users()
    accounts = [admin_account_row(uid, u) for uid, u in users.items()
                if isinstance(u, dict) and u.get('role') in ('patient', 'doctor')]

    role_filter = request.args.get('role', 'all').lower()
    if role_filter not in ('all', 'patient', 'doctor'):
        role_filter = 'all'
    status_filter = request.args.get('status', 'all').lower()
    if status_filter not in ['all'] + ADMIN_ACCOUNT_STATUSES:
        status_filter = 'all'
    query = request.args.get('q', '').strip()
    if role_filter != 'all':
        accounts = [account for account in accounts if account['role'] == role_filter]
    if status_filter != 'all':
        accounts = [account for account in accounts if account['status'] == status_filter]
    if query:
        lowered_query = query.lower()
        accounts = [account for account in accounts
                    if lowered_query in account['name'].lower()
                    or lowered_query in account['email'].lower()]
    accounts.sort(key=lambda account: account['created_at'], reverse=True)

    total_pages = max(1, -(-len(accounts) // ADMIN_PAGE_SIZE))
    try:
        page = min(max(int(request.args.get('page', 1)), 1), total_pages)
    except ValueError:
        page = 1
    start = (page - 1) * ADMIN_PAGE_SIZE

    return admin_page('admin_accounts.html', 'accounts', users,
                      accounts=accounts[start:start + ADMIN_PAGE_SIZE],
                      account_total=len(accounts),
                      first_index=start + 1,
                      page=page,
                      total_pages=total_pages,
                      account_query=query,
                      role_filter=role_filter,
                      status_filter=status_filter,
                      account_statuses=ADMIN_ACCOUNT_STATUSES,
                      # Only non-default filters, so pagination links stay short
                      filter_args={k: v for k, v in (('q', query), ('role', role_filter), ('status', status_filter))
                                   if v and v != 'all'},
                      open_create=request.args.get('create') == '1',
                      create_role='doctor' if request.args.get('new_role') == 'doctor' else 'patient')


@app.route('/admin/orders', endpoint='admin_orders')
@admin_required
def admin_orders_page():
    users = admin_load_users()
    orders = admin_order_rows(users)
    status_counts = {status: sum(1 for o in orders if o['status'] == status) for status in WRISTBAND_STATUSES}
    status_filter = request.args.get('status', 'all').lower()
    if status_filter not in WRISTBAND_STATUSES:
        status_filter = 'all'
    ordered_uids = {o['uid'] for o in orders}
    # Orders can only be created for active patients without one (admin_create_order enforces this)
    available_patients = sorted(
        ({'uid': uid, 'name': u.get('name') or 'Unnamed patient'} for uid, u in users.items()
         if isinstance(u, dict) and u.get('role') == 'patient' and u.get('status') == 'active'
         and uid not in ordered_uids),
        key=lambda p: p['name'].lower())

    delivered = [o for o in orders if o['status'] == 'delivered']
    open_orders = [o for o in orders if o['status'] != 'delivered']
    return admin_page('admin_orders.html', 'orders', users,
                      orders=[o for o in orders if status_filter == 'all' or o['status'] == status_filter],
                      order_total=len(orders),
                      revenue=sum(o['price'] for o in delivered), delivered_count=len(delivered),
                      pipeline=sum(o['price'] for o in open_orders), open_count=len(open_orders),
                      unit_price=WRISTBAND_PRICE_PKR,
                      status_counts=status_counts,
                      status_filter=status_filter,
                      wristband_statuses=WRISTBAND_STATUSES,
                      available_patients=available_patients,
                      open_create=request.args.get('create') == '1')


@app.route('/admin/users/create', methods=['POST'])
@admin_required
def admin_create_user():
    import re

    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400

    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip().lower()
    phone = request.form.get('phone', '').strip()
    password = request.form.get('password', '')
    role = request.form.get('role', '').strip().lower()
    if (not name or len(name) > 100 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email)
            or len(phone) > 30 or len(password) < 6 or role not in ('patient', 'doctor')):
        flash('Enter a valid name, email, phone, password (at least 6 characters), and account role.', 'error')
        return admin_back('admin_accounts')

    status = 'pending' if role == 'doctor' else 'active'
    created_user = None
    try:
        created_user = auth.create_user(
            email=email,
            password=password,
            display_name=name,
            disabled=role == 'doctor'
        )
        db.reference(f'users/{created_user.uid}').set({
            'uid': created_user.uid,
            'name': name,
            'email': email,
            'phone': phone,
            'role': role,
            'status': status,
            'created_at': {'.sv': 'timestamp'}
        })
        activity_log('account_created', f'Created {role} account {name}',
                     details={'account_uid': created_user.uid, 'email': email, 'role': role})
        flash(f'{role.capitalize()} account created. ' +
              ('It is pending approval.' if role == 'doctor' else 'They can now sign in.'), 'success')
    except Exception:
        if created_user:
            try:
                auth.delete_user(created_user.uid)
            except Exception:
                pass
        activity_log('account_create_failed', f'Failed to create {role} account {name}', 'failed',
                     details={'email': email, 'role': role})
        flash('The account could not be created. Check that the email is unused and Firebase is available.', 'error')
    return admin_back('admin_accounts')


@app.route('/admin/users/<user_uid>/update', methods=['POST'])
@admin_required
def admin_update_user(user_uid):
    import re

    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400
    if not is_safe_key(user_uid) or user_uid == ADMIN_UID:
        flash('That account cannot be edited.', 'error')
        return admin_back('admin_accounts')

    user_ref = db.reference(f'users/{user_uid}')
    profile = user_ref.get()
    if not isinstance(profile, dict) or profile.get('role') not in ('patient', 'doctor'):
        flash('That account could not be found.', 'error')
        return admin_back('admin_accounts')

    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip().lower()
    phone = request.form.get('phone', '').strip()
    password = request.form.get('password', '')
    status = request.form.get('status', '').strip().lower()
    role = profile['role']
    allowed_statuses = {'active', 'suspended'} if role == 'patient' else {'pending', 'approved', 'rejected', 'suspended'}
    if (not name or len(name) > 100 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email)
            or len(phone) > 30 or status not in allowed_statuses or (password and len(password) < 6)):
        flash('Review the account fields and try again. Passwords must be at least 6 characters.', 'error')
        return admin_back('admin_accounts')

    auth_changes = {
        'email': email,
        'display_name': name,
        'disabled': status == 'suspended' or (role == 'doctor' and status != 'approved')
    }
    if password:
        auth_changes['password'] = password
    try:
        auth.update_user(user_uid, **auth_changes)
        changes = {'name': name, 'email': email, 'phone': phone, 'status': status}
        if role == 'doctor' and status in ('approved', 'rejected') and status != profile.get('status'):
            # Keeps the Approvals page's 'recently approved/rejected' lists accurate
            changes.update(reviewed_at={'.sv': 'timestamp'}, reviewed_by=session['user']['uid'])
        user_ref.update(changes)
        changed = [f for f in ('name', 'email', 'phone', 'status') if changes[f] != profile.get(f, '')]
        if password:
            changed.append('password')   # that it changed, never the value
        activity_log('account_updated', f"Updated {role} account {name}",
                     details={'account_uid': user_uid, 'changed': ', '.join(changed) or 'no changes',
                              'status': f"{profile.get('status')} → {status}" if 'status' in changed else None})
        flash('Account updated successfully.', 'success')
    except Exception:
        flash('The account could not be updated. Check that the email is unused and Firebase is available.', 'error')
    return admin_back('admin_accounts')


@app.route('/admin/users/<user_uid>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_uid):
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400
    if not is_safe_key(user_uid) or user_uid == ADMIN_UID or user_uid == session['user'].get('uid'):
        flash('That account cannot be deleted.', 'error')
        return admin_back('admin_accounts')

    user_ref = db.reference(f'users/{user_uid}')
    profile = user_ref.get()
    if not isinstance(profile, dict) or profile.get('role') not in ('patient', 'doctor'):
        flash('That account could not be found.', 'error')
        return admin_back('admin_accounts')

    try:
        auth.delete_user(user_uid)
        deletions = {
            f'users/{user_uid}': None,
            f'wristband_orders/{user_uid}': None
        }
        deletions[f'chat_unread/{user_uid}'] = None
        deletions[f'presence/{user_uid}'] = None
        if profile.get('role') == 'patient':
            device_id = (profile.get('wristband') or {}).get('device_id')
            if isinstance(device_id, str) and is_safe_key(device_id):
                deletions[f'wristband_devices/{device_id}'] = None
            deletions[f'clinical_notes/{user_uid}'] = None
            deletions[f'alerts/{user_uid}'] = None
            # Their conversations go with them (chat ids are "<uid>_<uid>", sorted)
            for chat_id in (db.reference('chats').get(shallow=True) or {}):
                if (chat_id.startswith(user_uid + '_') or chat_id.endswith('_' + user_uid)) \
                        and db.reference(f'chats/{chat_id}/patient_uid').get() == user_uid:
                    deletions[f'chats/{chat_id}'] = None
                    doctor_uid = db.reference(f'chats/{chat_id}/doctor_uid').get()
                    if doctor_uid and is_safe_key(doctor_uid):
                        deletions[f'chat_unread/{doctor_uid}/{chat_id}'] = None
        else:
            # Unassign the deleted doctor's patients so they're auto-assigned a new
            # doctor on their next dashboard visit instead of pointing at a missing account
            for patient_uid, patient in (db.reference('users').get() or {}).items():
                if patient_assigned_to(patient, user_uid):
                    for field in ('assigned_doctor_uid', 'assigned_at', 'assigned_by'):
                        deletions[f'users/{patient_uid}/{field}'] = None
        db.reference().update(deletions)
        activity_log('account_deleted', f"Deleted {profile.get('role')} account {profile.get('name') or user_uid}",
                     details={'account_uid': user_uid, 'email': profile.get('email', ''), 'role': profile.get('role')})
        flash('Account and associated app data deleted.', 'success')
    except Exception:
        activity_log('account_delete_failed', f"Failed to delete account {profile.get('name') or user_uid}", 'failed',
                     details={'account_uid': user_uid})
        flash('The account could not be fully deleted. Check Firebase Auth and Realtime Database.', 'error')
    return admin_back('admin_accounts')


@app.route('/admin/orders/<user_uid>/status', methods=['POST'])
@admin_required
def admin_update_order_status(user_uid):
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400
    if not is_safe_key(user_uid):
        flash('That order could not be found.', 'error')
        return admin_back('admin_orders')

    status = request.form.get('status', '').lower()
    if status not in WRISTBAND_STATUSES:
        flash('Choose a valid wristband order status.', 'error')
        return admin_back('admin_orders')

    order_ref = db.reference(f'wristband_orders/{user_uid}')
    order = order_ref.get()
    if not isinstance(order, dict):
        flash('That wristband order could not be found.', 'error')
    else:
        order_ref.update({'status': status})
        flash('Wristband order status updated.', 'success')
    return admin_back('admin_orders')


@app.route('/admin/orders/create', methods=['POST'])
@admin_required
def admin_create_order():
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400

    user_uid = request.form.get('user_uid', '').strip()
    if not is_safe_key(user_uid):
        flash('Choose a valid patient account.', 'error')
        return admin_back('admin_orders')

    patient = db.reference(f'users/{user_uid}').get()
    if not isinstance(patient, dict) or patient.get('role') != 'patient' or patient.get('status') != 'active':
        flash('Orders can only be created for active patient accounts.', 'error')
        return admin_back('admin_orders')

    order_ref = db.reference(f'wristband_orders/{user_uid}')
    if isinstance(order_ref.get(), dict):
        flash('This patient already has a wristband order.', 'error')
        return admin_back('admin_orders')

    fullname = request.form.get('fullname', '').strip() or patient.get('name', '')
    phone = request.form.get('phone', '').strip()
    address = request.form.get('address', '').strip()
    city = request.form.get('city', '').strip()
    size = request.form.get('size', 'M').strip().upper()
    notes = request.form.get('notes', '').strip()
    if (not fullname or len(fullname) > 100 or not phone or len(phone) > 30 or not address
            or len(address) > 300 or not city or len(city) > 100 or size not in ('S', 'M', 'L')
            or len(notes) > 500):
        flash('Complete the delivery details and choose a valid wrist size.', 'error')
        return admin_back('admin_orders')

    order_ref.set(new_order_record(user_uid, {'fullname': fullname, 'phone': phone, 'address': address,
                                              'city': city, 'size': size, 'notes': notes}))
    flash(f'Wristband order created ({format_pkr(WRISTBAND_PRICE_PKR)}, cash on delivery).', 'success')
    return admin_back('admin_orders')


@app.route('/admin/orders/<user_uid>/delete', methods=['POST'])
@admin_required
def admin_delete_order(user_uid):
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400
    if not is_safe_key(user_uid):
        flash('That order could not be found.', 'error')
        return admin_back('admin_orders')

    order_ref = db.reference(f'wristband_orders/{user_uid}')
    if not isinstance(order_ref.get(), dict):
        flash('That wristband order could not be found.', 'error')
    else:
        order_ref.delete()
        flash('Wristband order deleted.', 'success')
    return admin_back('admin_orders')


@app.route('/admin/doctors/<doctor_uid>/<decision>', methods=['POST'])
@admin_required
def admin_review_doctor(doctor_uid, decision):
    if not admin_csrf_valid():
        return 'Invalid or expired form token.', 400

    if not is_safe_key(doctor_uid):
        flash('That doctor account could not be found.', 'error')
        return admin_back('admin_approvals', 'admin_dashboard')

    if decision not in ('approve', 'reject'):
        flash('That doctor review action is not valid.', 'error')
        return admin_back('admin_approvals', 'admin_dashboard')

    doctor_ref = db.reference(f'users/{doctor_uid}')
    doctor = doctor_ref.get()
    if not isinstance(doctor, dict) or doctor.get('role') != 'doctor' or doctor.get('status') != 'pending':
        flash('That doctor account is no longer pending approval.', 'error')
        return admin_back('admin_approvals', 'admin_dashboard')

    # Approval only grants access; existing patients are not reassigned to the new doctor
    status = 'approved' if decision == 'approve' else 'rejected'
    try:
        auth.update_user(doctor_uid, disabled=decision != 'approve')
    except Exception:
        flash('Firebase Auth could not apply this doctor review.', 'error')
        return admin_back('admin_approvals', 'admin_dashboard')
    doctor_ref.update({'status': status, 'reviewed_at': {'.sv': 'timestamp'},
                       'reviewed_by': session['user']['uid']})
    action = 'approved' if decision == 'approve' else 'rejected'
    activity_log(f'doctor_{action}', f"{action.capitalize()} doctor {strip_dr(doctor.get('name')) or doctor_uid}",
                 details={'doctor_uid': doctor_uid, 'email': doctor.get('email', '')})
    flash(f"{doctor.get('name') or 'Doctor'} was {action}.", 'success')
    return admin_back('admin_approvals', 'admin_dashboard')


@app.route('/api/admin/assign-patient-to-doctor', methods=['POST'])
@login_required
@role_required('admin')
def admin_assign_patient_to_doctor():
    # Placeholder for the admin module. Expected body: {patient_uid, doctor_uid}. The
    # implementation should validate both, then call
    # assign_doctor(patient_uid, doctor_uid, f"admin:{session['user']['uid']}")
    return jsonify({'success': False, 'error': 'Pending admin module'}), 501


# ========== LOGOUT ==========
@app.route('/logout')
def logout():
    uid = (session.get('user') or {}).get('uid')
    if session.get('user'):
        activity_log('logout', 'Logged out', 'info')
    if is_safe_key(uid):
        try:
            db.reference(f'presence/{uid}').delete()   # stop counting as an active session
        except Exception:
            pass
    session.clear()  # also drops the admin CSRF token
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)