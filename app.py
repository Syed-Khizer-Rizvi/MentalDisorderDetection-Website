from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import firebase_admin
from firebase_admin import credentials, auth, db
import os
from functools import wraps

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
    """Called from frontend after Firebase sign-in. Verifies ID token and creates session."""
    try:
        data = request.get_json()
        id_token = data.get('idToken')

        # Verify the token with Firebase Admin
        decoded = auth.verify_id_token(id_token)
        uid = decoded['uid']
        email = decoded.get('email', '')

        # Fetch user profile from Realtime Database
        user_ref = db.reference(f'users/{uid}')
        profile = user_ref.get()

        if not profile:
            return jsonify({'success': False, 'error': 'User profile not found. Please contact support.'}), 404

        # Check if doctor account is approved
        if profile.get('role') == 'doctor' and profile.get('status') != 'approved':
            return jsonify({'success': False, 'error': 'Your doctor account is pending approval by admin.'}), 403

        # Save session
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
    """Called from frontend AFTER Firebase account is created. Saves profile to Realtime DB."""
    try:
        data = request.get_json()
        id_token = data.get('idToken')
        name = data.get('name', '').strip()
        phone = data.get('phone', '').strip()
        role = data.get('role', 'patient').strip().lower()

        # Verify the token
        decoded = auth.verify_id_token(id_token)
        uid = decoded['uid']
        email = decoded.get('email', '')

        # ADMIN role is NEVER allowed through public registration
        if role == 'admin':
            # Delete the just-created Firebase account for security
            try:
                auth.delete_user(uid)
            except:
                pass
            return jsonify({'success': False, 'error': 'Admin accounts cannot be created through public registration.'}), 403

        # Doctor accounts need approval
        status = 'pending' if role == 'doctor' else 'active'

        # Save profile to Realtime Database
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
    """Route to role-specific dashboard."""
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
    return f"<h1>Patient Dashboard</h1><p>Welcome, {session['user']['name']}!</p><p>Role: {session['user']['role']}</p><a href='/logout'>Logout</a>"


@app.route('/doctor/dashboard')
@login_required
@role_required('doctor')
def doctor_dashboard():
    return f"<h1>Doctor Dashboard</h1><p>Welcome, Dr. {session['user']['name']}!</p><p>Role: {session['user']['role']}</p><a href='/logout'>Logout</a>"


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