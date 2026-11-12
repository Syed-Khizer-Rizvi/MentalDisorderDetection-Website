from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
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
app.secret_key = 'mental-disorder-detection-secret-key-change-in-production'
# Placeholder never matches a real Firebase UID, so no one is admin until .env sets it
ADMIN_UID = os.environ.get('ADMIN_UID', 'SET_IN_ENV_FILE').strip()

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
def admin_login():
    return redirect(url_for('login'))


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
            return jsonify({'success': False, 'needs_registration': True,
                            'error': 'User profile not found. Please create your account first.'}), 404

        if profile.get('role') == 'doctor' and profile.get('status') != 'approved':
            message = ('Your doctor account is pending approval by admin.'
                       if profile.get('status') == 'pending'
                       else 'Your doctor account was not approved. Please contact the administrator.')
            return jsonify({'success': False, 'error': message}), 403

        session['user'] = {
            'uid': uid,
            'email': email,
            'name': profile.get('name', ''),
            'role': profile.get('role', 'patient'),
            'status': profile.get('status', 'active')
        }
        session['_login_toast'] = True   # one-time: popped by the next dashboard load
        try:
            # Daily login total for the admin analytics "Activities Completed" figure (server-local day)
            db.reference(f"login_counts/{time.strftime('%Y-%m-%d')}").set({'.sv': {'increment': 1}})
        except Exception:
            pass   # analytics only; never block a login on it

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
                           order=order, statuses=WRISTBAND_STATUSES, wb=load_wristband_view(uid, 1))


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
    return jsonify({'success': True, 'device_id': device_id, 'api_key': api_key,
                    'endpoint': url_for('wristband_data', _external=True)})


@app.route('/api/wristband/data', methods=['POST'])
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
            return jsonify({'success': True, 'duplicate': True})
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
    return jsonify({'success': True})


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
                'created_date': admin_date(created_at)
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
    db.reference('system/fusion_weights').set({**weights, 'updated_at': {'.sv': 'timestamp'},
                                                'updated_by': session['user']['uid']})
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
    db.reference('system/risk_thresholds').set({**thresholds, 'updated_at': {'.sv': 'timestamp'},
                                                 'updated_by': session['user']['uid']})
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

    return admin_page('admin_orders.html', 'orders', users,
                      orders=[o for o in orders if status_filter == 'all' or o['status'] == status_filter],
                      order_total=len(orders),
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
        flash(f'{role.capitalize()} account created. ' +
              ('It is pending approval.' if role == 'doctor' else 'They can now sign in.'), 'success')
    except Exception:
        if created_user:
            try:
                auth.delete_user(created_user.uid)
            except Exception:
                pass
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
        flash('Account and associated app data deleted.', 'success')
    except Exception:
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

    order_ref.set({
        'uid': user_uid,
        'fullname': fullname,
        'phone': phone,
        'address': address,
        'city': city,
        'size': size,
        'notes': notes,
        'status': 'pending',
        'created_at': {'.sv': 'timestamp'}
    })
    flash('Wristband order created.', 'success')
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
    if is_safe_key(uid):
        try:
            db.reference(f'presence/{uid}').delete()   # stop counting as an active session
        except Exception:
            pass
    session.clear()  # also drops the admin CSRF token
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)