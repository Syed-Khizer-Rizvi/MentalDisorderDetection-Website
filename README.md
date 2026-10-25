# Mental Disorder Detection

An AI-assisted mental health screening and monitoring web app. Patients complete validated
questionnaires (PHQ-9, GAD-7, PSS-10) and a webcam facial emotion scan. The app combines them
into a composite risk score, then tracks it over time. Doctors see a live overview of their
patients' risk levels.

Built with Flask, Firebase (Authentication + Realtime Database) and DeepFace.

## Prerequisites

- **Python 3.11 or newer** (tested on 3.14)
- **Git**
- A **webcam** for the facial scan. A phone works too with a webcam app such as Iriun Webcam.
- An **internet connection** on first run: DeepFace downloads its emotion model (about 6 MB)
- The project's **`firebase_key.json`** service account key (see below)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Syed-Khizer-Rizvi/MentalDisorderDetection-Website.git
cd MentalDisorderDetection-Website
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Firebase key

The app authenticates to Firebase with a service account key that is **not** in the repository.

1. Get `firebase_key.json` from the project owner. Alternatively, generate one in the Firebase console:
   **Project settings → Service accounts → Generate new private key**.
2. Place it in the project root, next to `app.py`:

```
MentalDisorderDetection-Website/
├── app.py
├── firebase_key.json   ← here
├── requirements.txt
├── static/
└── templates/
```

> **Never commit `firebase_key.json`.** It grants full admin access to the Firebase project.
> It is listed in `.gitignore`. If it is ever pushed by mistake, revoke the key in the
> Firebase console straight away and generate a new one.

### 5. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

The app runs in Flask debug mode, which is intended for local development only.

## Demo flow

### Patient

1. **Register** at `/register` as a *Patient*, then **log in**.
2. **Dashboard** (`/patient/dashboard`): starts empty until the first assessment.
3. **New Assessment**: answer the PHQ-9 (depression), GAD-7 (anxiety) and PSS-10 (stress)
   questionnaires. PSS-10 items 4, 5, 7 and 8 are reverse-scored.
4. **Facial Scan**: allow camera access. DeepFace analyses facial emotion from webcam frames
   and records the dominant and peak emotion.
5. **Results**: shows the composite score (0–100) and risk level (Low / Moderate / High / Critical).
   It also shows recommendations based on that level. Critical results prompt a wristband order.
6. Back on the **Dashboard**:
   - risk, last assessment, facial mood and wristband status cards
   - a **Risk Trend** chart of the last 7 assessments
7. **History**: every past assessment, newest first. Click one to reopen its full results.
8. **Wristband**: shows the PsyMonitor Band features, the order form, and order status tracking
   (pending → confirmed → shipped → delivered).
9. **Activities**: mindfulness games and guided meditations (placeholders for now) and a
   **Daily Journal** that saves entries to Firebase.
10. **Profile**: edit personal details and an emergency contact.

### Doctor

1. Register at `/register` as a *Doctor*. Doctor accounts start as **pending**.
2. Approve the account. There is no admin screen yet, so in the Firebase console open
   **Realtime Database → users → {uid}** and set `status` to `approved`.
3. Log in to the **Doctor Dashboard** (`/doctor/dashboard`):
   - total patients, critical alerts and average risk score
   - the risk distribution across patients
   - the highest-risk patients
   - the most recent assessments

## How the composite score works

```
composite = ((PHQ-9 / 27 + GAD-7 / 21 + PSS-10 / 40) × 100 + negative facial emotion %) / 4
```

Negative facial emotion % is the sum of *sad*, *angry*, *fear* and *disgust*.

| Composite | Risk level |
|-----------|------------|
| 0–25      | Low        |
| 26–50     | Moderate   |
| 51–70     | High       |
| 71–100    | Critical   |

## Technical note: TensorFlow → ONNX Runtime

The FYP-1 proposal specified **TensorFlow** for CNN model inference. The final implementation
uses **ONNX Runtime** instead, through **DeepFace 0.0.103**.

- **Same model.** DeepFace 0.0.103 loads `facial_expression_model_weights.onnx`. This is an
  ONNX export of the same pretrained facial-expression CNN (48×48 grayscale input, 7 emotion
  classes) that earlier DeepFace releases ran on TensorFlow/Keras. The model and its outputs
  are unchanged; only the inference engine differs.
- **Why ONNX Runtime.** ONNX Runtime is built only for running trained models, while
  TensorFlow also includes everything needed to train them. That makes it a much smaller
  install, and it generally uses less memory and starts faster. This app only needs to run a
  pretrained model, so the extra parts of TensorFlow aren't needed. Exporting trained models
  to ONNX is also common practice for production deployment.
- **Simpler setup.** TensorFlow is not a dependency at all, which avoids its large download
  and its frequent Python-version compatibility issues.

Note: no formal memory or speed benchmark was run for this project. The benefits above are
general properties of ONNX Runtime compared with TensorFlow, not measured results.

## Project structure

```
app.py                  Flask routes, Firebase access, scoring logic
requirements.txt        Python dependencies
static/css/             Page styles (dashboard, auth, assessment, results)
static/js/sidebar.js    Shared sidebar / mobile menu behaviour
templates/              Jinja2 page templates
```
