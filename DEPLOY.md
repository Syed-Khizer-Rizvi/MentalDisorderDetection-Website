# Deployment notes

Settings that live outside the code and have to be set up by hand when the app is deployed.

## Firebase: email verification template

New accounts that sign up with email and password must click a verification link before they can
sign in (Google sign-ups are already verified by Google). The app asks Firebase Authentication to
send that email, so its wording comes from the Firebase Console, not from this repository.

1. Open the [Firebase Console](https://console.firebase.google.com/) and select the
   `mentaldisorderdetection-31d60` project.
2. Go to **Authentication → Templates → Email address verification**.
3. Click the pencil icon to edit the template:
   - **Sender name:** `PsyMonitor Support`
   - **Subject:** `Verify your PsyMonitor account`
   - **Message:** leave Firebase's default, or reword it. Keep the `%LINK%` placeholder.
4. **Action URL:** leave it as the default Firebase handler
   (`https://mentaldisorderdetection-31d60.firebaseapp.com/__/auth/action`). It marks the address as
   verified and shows a confirmation page. The user then returns to the app and clicks
   **I've verified my email** on `/verify-email` (or simply signs in again).
5. Save.

Check that the deployed domain is listed under **Authentication → Settings → Authorized domains**,
otherwise Firebase rejects the request.

### How the app uses it

- Registration (`/api/register-profile`) stores `users/{uid}/email_verified: false` and requests the
  email from Firebase. The user lands on `/verify-email`.
- `/verify-email` can resend the email at most **once per minute per account**. The server enforces
  this limit (stored at `email_verification/{uid}/last_sent_at`), and the browser can't skip it.
- Sign-in (`/api/verify-token`) is refused for `email_verified: false` until Firebase reports the
  address verified. The flag is then set to `true` automatically.
- Doctors need **both** a verified email and admin approval.
- Admins can mark an account as verified on **Admin → Accounts** (for test accounts). Accounts created
  by an admin are verified automatically.
- Existing accounts were grandfathered once at startup: every `active` patient and `approved` doctor
  got `email_verified: true`. The run is recorded at `system/migrations/email_verification_v1` and
  never repeats. To run it again (for example on a restored database), delete that node and restart
  the app.

Activity log events: `email_verification_sent`, `email_verification_completed`,
`email_verification_bypassed`, `login_blocked_unverified`. None of them include email content.

## Email notifications (SMTP)

The five notification emails (doctor approved, consultation and emergency requests, risk alerts,
treatment plan updates) are sent by the app itself through Gmail SMTP and are unrelated to the
verification email above. Set `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD` and
`SMTP_FROM_NAME` in `.env` (never committed). Admins can switch them off under
**Admin → System Settings → Email Notifications**.

## Files that are never committed

`firebase_key.json` (Firebase Admin service account), `.env` (secrets) and `kaggle.json` stay on the
server only.
