// Shared by the login and register pages (ES module).

// Friendly text for Firebase Auth error codes. Returns '' when the user simply
// cancelled (closed the Google popup), so callers can stay quiet.
export function authErrorMessage(err) {
  switch (err && err.code) {
    case 'auth/popup-closed-by-user':
    case 'auth/cancelled-popup-request':
      return '';
    case 'auth/invalid-credential':
    case 'auth/wrong-password':
    case 'auth/user-not-found':
      return 'Incorrect email or password.';
    case 'auth/invalid-email':
      return 'Enter a valid email address.';
    case 'auth/email-already-in-use':
      return 'An account with this email already exists. Sign in instead.';
    case 'auth/weak-password':
      return 'Password must be at least 6 characters.';
    case 'auth/too-many-requests':
      return 'Too many attempts. Try again later.';
    case 'auth/user-disabled':
      return 'This account has been disabled. Please contact the administrator.';
    case 'auth/popup-blocked':
      return 'Your browser blocked the Google sign-in popup. Allow popups for this site and try again.';
    case 'auth/operation-not-allowed':
      return 'Google sign-in is not enabled yet. Please use email and password.';
    case 'auth/unauthorized-domain':
      return 'Google sign-in is not allowed on this website address yet.';
    case 'auth/account-exists-with-different-credential':
      return 'An account already exists with this email. Sign in with your email and password.';
    case 'auth/network-request-failed':
      return 'Network error. Check your connection and try again.';
    default:
      return (err && err.message) || 'Something went wrong. Please try again.';
  }
}

export function googleProvider(GoogleAuthProvider) {
  const provider = new GoogleAuthProvider();
  provider.setCustomParameters({ prompt: 'select_account' });  // let people pick which Google account
  return provider;
}

export async function verifyToken(user) {
  const res = await fetch('/api/verify-token', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ idToken: await user.getIdToken() })
  });
  return res.json();
}

// Shows a compact success pill above the Sign in / Create account tabs, fades the form
// (still visible, but no longer clickable) and redirects after a short pause.
export function showAuthSuccess(message, href, delay = 1500) {
  const pill = document.createElement('div');
  pill.className = 'auth-success';
  pill.setAttribute('role', 'status');
  pill.innerHTML = '<span class="auth-success-icon" aria-hidden="true">✓</span><span></span>';
  pill.lastChild.textContent = message;
  const tabs = document.querySelector('.auth-tabs');
  tabs.parentNode.insertBefore(pill, tabs);
  pill.scrollIntoView({ block: 'nearest', behavior: 'smooth' });   // register page: submit button is far below

  document.querySelector('.auth-form-box').classList.add('is-redirecting');
  setTimeout(() => { window.location.href = href; }, delay);
}
