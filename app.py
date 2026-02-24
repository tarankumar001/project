"""
Motor Fault Detection Web App (Flask) + ML Admin Dashboard
===========================================================

Architecture:
  - Authentication (login / register / logout)  →  SQLite users table
  - Model Training                              →  RandomForestClassifier on uploaded CSV
  - Diagnostic Plots                            →  ROC Curve, Confusion Matrix, Feature Importance
  - Training Logs                               →  SQLite training_logs table
  - Live Monitor                                →  Arduino serial → ML prediction

Run with:
    python app.py
"""

# ──────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────
import datetime
import io
import os
import sqlite3
import time
from functools import wraps

# ──────────────────────────────────────────────
# Third-party
# ──────────────────────────────────────────────
import joblib
import matplotlib
matplotlib.use("Agg")          # non-GUI backend; must be before plt import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from flask import (
    Flask,
    Response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from werkzeug.security import check_password_hash, generate_password_hash


# ══════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════

SERIAL_PORT = "COM3"
BAUD_RATE   = 9600
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "motor_model.pkl")
PLOTS_DIR   = os.path.join("static", "plots")
DATABASE    = os.path.join(os.path.dirname(__file__), "users.db")

# ──────────────────────────────────────────────
# Flask application
# ──────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-to-a-long-random-secret-in-production"

# Global state (simple in-process storage)
model                = None   # Loaded/trained ML model
ser                  = None   # Serial connection to Arduino
last_training_results = None  # Last training metrics dict


# ══════════════════════════════════════════════
#  DATABASE HELPERS
# ══════════════════════════════════════════════

def get_db_connection():
    """Open a new SQLite connection using Row factory for dict-like access."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create database tables if they do not exist yet."""
    conn = get_db_connection()
    cur  = conn.cursor()

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL
        )
    """)

    # Training logs table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT    NOT NULL,
            username     TEXT    NOT NULL,
            accuracy     REAL,
            n_samples    INTEGER,
            n_train      INTEGER,
            n_test       INTEGER,
            anomaly_rate REAL,
            roc_auc      REAL
        )
    """)

    conn.commit()

    # ── Seed default admin account (only if it doesn't already exist) ──
    cur.execute("SELECT id FROM users WHERE username = ?", ("admin",))
    if not cur.fetchone():
        admin_hash = generate_password_hash("password")
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ("admin", admin_hash),
        )
        conn.commit()
        print("[init_db] Default admin account created (username: admin / password: password)")
    else:
        print("[init_db] Admin account already exists.")

    conn.close()


def log_training_attempt(username, train_out):
    """
    Persist a single training run's metrics to the training_logs table.
    Silently swallows errors so a logging failure never breaks the UI.
    """
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        created_at = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        cur.execute("""
            INSERT INTO training_logs
                (created_at, username, accuracy, n_samples,
                 n_train, n_test, anomaly_rate, roc_auc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            created_at,
            username,
            float(train_out.get("accuracy", 0.0)),
            int(train_out.get("n_samples", 0)),
            int(train_out.get("n_train",   0)),
            int(train_out.get("n_test",    0)),
            float(train_out["anomaly_rate"]) if train_out.get("anomaly_rate") is not None else None,
            float(train_out["roc_auc"])      if train_out.get("roc_auc")      is not None else None,
        ))
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[log_training_attempt] Failed to write log: {exc}")


# ══════════════════════════════════════════════
#  AUTHENTICATION HELPERS
# ══════════════════════════════════════════════

def login_required(view_func):
    """Decorator: redirect to login page if the user is not authenticated."""
    @wraps(view_func)
    def wrapped(**kwargs):
        if "user_id" not in session:
            return redirect(url_for("login", next=request.path))
        return view_func(**kwargs)
    return wrapped


@app.context_processor
def inject_user():
    """Inject authentication state into every template context."""
    return {
        "logged_in":    "user_id" in session,
        "current_user": session.get("username"),
    }


# ══════════════════════════════════════════════
#  AUTHENTICATION ROUTES
# ══════════════════════════════════════════════

@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration — hashes password with werkzeug before storing."""
    error   = None
    message = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm  = request.form.get("confirm_password", "").strip()

        if not username or not password or not confirm:
            error = "Please fill in all fields."
        elif password != confirm:
            error = "Passwords do not match."
        elif len(password) < 4:
            error = "Password must be at least 4 characters."
        else:
            try:
                conn = get_db_connection()
                cur  = conn.cursor()
                cur.execute("SELECT id FROM users WHERE username = ?", (username,))
                if cur.fetchone():
                    error = "Username already taken. Please choose another."
                else:
                    pw_hash = generate_password_hash(password)
                    cur.execute(
                        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                        (username, pw_hash),
                    )
                    conn.commit()
                    message = "Registration successful. You can now sign in."
                conn.close()
            except Exception as exc:
                error = f"Registration failed: {exc}"

    return render_template("register.html", error=error, message=message)


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login — validates credentials and creates a session."""
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            error = "Please enter both username and password."
        else:
            try:
                conn = get_db_connection()
                cur  = conn.cursor()
                cur.execute(
                    "SELECT id, username, password_hash FROM users WHERE username = ?",
                    (username,),
                )
                row = cur.fetchone()
                conn.close()

                if row and check_password_hash(row["password_hash"], password):
                    session["user_id"]  = row["id"]
                    session["username"] = row["username"]
                    next_page = request.args.get("next")
                    return redirect(next_page or url_for("dashboard_upload"))
                else:
                    error = "Invalid username or password."
            except Exception as exc:
                error = f"Login error: {exc}"

    return render_template("login.html", error=error)


@app.route("/logout")
@login_required
def logout():
    """Clear session and redirect to login."""
    session.clear()
    return redirect(url_for("login"))


# ══════════════════════════════════════════════
#  ML — TRAINING LOGIC
# ══════════════════════════════════════════════

def train_model_from_dataframe(df):
    """
    Train a RandomForestClassifier on the supplied DataFrame.

    Expected columns: temperature, vibration, label

    Returns a dict with:
        model, accuracy, cm, classes, roc_auc, plots,
        n_samples, n_train, n_test, class_counts,
        anomaly_label, anomaly_rate
    """
    # ── Validation ──────────────────────────────────
    required_cols = {"temperature", "vibration", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

    if len(df) < 10:
        raise ValueError("Need at least 10 rows to train. Please provide more data.")

    # ── Feature engineering ─────────────────────────
    df["vib_rms"] = df["vibration"].astype(float)
    X = df[["temperature", "vib_rms"]].astype(float)
    y = df["label"]

    if y.nunique() < 2:
        raise ValueError("Need at least two distinct label values for classification.")

    # ── Train / test split ──────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y,
    )

    # ── Model training ──────────────────────────────
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # ── Predictions ─────────────────────────────────
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # ── Metrics ─────────────────────────────────────
    acc      = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    classes  = sorted(y.unique().tolist())

    n_samples    = len(df)
    n_train      = len(X_train)
    n_test       = len(X_test)
    class_counts = y.value_counts().to_dict()

    # ── ROC (binary only) ───────────────────────────
    roc_auc      = None
    fpr = tpr    = None
    anomaly_label = None
    anomaly_rate  = None

    if len(classes) == 2:
        anomaly_label   = classes[1]         # second class treated as "fault"
        y_test_binary   = (y_test == anomaly_label).astype(int)
        fpr, tpr, _     = roc_curve(y_test_binary, y_proba[:, 1])
        roc_auc         = roc_auc_score(y_test_binary, y_proba[:, 1])
        anomaly_count   = class_counts.get(anomaly_label, 0)
        anomaly_rate    = anomaly_count / float(n_samples) if n_samples > 0 else None

    # ── Plot generation ─────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.style.use("dark_background")        # dark plots to match dashboard theme
    plots = {}

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    fig.patch.set_facecolor("#0d1117")
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix", color="#e6edf3", fontsize=12, pad=10)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, color="#94a3b8")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, color="#94a3b8")
    ax.set_xlabel("Predicted", color="#94a3b8")
    ax.set_ylabel("Actual",    color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2.0 else "#94a3b8", fontsize=13)
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close()
    plots["confusion_matrix"] = "plots/confusion_matrix.png"

    # ROC curve
    if roc_auc is not None and fpr is not None and tpr is not None:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        fig.patch.set_facecolor("#0d1117")
        ax.plot(fpr, tpr, color="#4f9cf9", linewidth=2,
                label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate", color="#94a3b8")
        ax.set_ylabel("True Positive Rate",  color="#94a3b8")
        ax.set_title("ROC Curve", color="#e6edf3", fontsize=12, pad=10)
        ax.tick_params(colors="#94a3b8")
        ax.legend(loc="lower right", facecolor="#161b22", labelcolor="#e6edf3")
        ax.set_facecolor("#0d1117")
        roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=130, facecolor=fig.get_facecolor())
        plt.close()
        plots["roc_curve"] = "plots/roc_curve.png"

    # Feature importance
    importances   = clf.feature_importances_
    feature_names = ["temperature", "vib_rms"]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    fig.patch.set_facecolor("#0d1117")
    colors = ["#4f9cf9", "#a78bfa"]
    y_pos  = np.arange(len(feature_names))
    ax.barh(y_pos, importances, align="center", color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, color="#94a3b8")
    ax.set_xlabel("Importance", color="#94a3b8")
    ax.set_title("Feature Importance", color="#e6edf3", fontsize=12, pad=10)
    ax.tick_params(colors="#94a3b8")
    ax.set_facecolor("#0d1117")
    fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(fi_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close()
    plots["feature_importance"] = "plots/feature_importance.png"

    # ── Save model ──────────────────────────────────
    joblib.dump(clf, MODEL_PATH)
    print(f"[train_model] Model saved → {MODEL_PATH}")

    return {
        "model":        clf,
        "accuracy":     acc,
        "cm":           cm,
        "classes":      classes,
        "roc_auc":      roc_auc,
        "plots":        plots,
        "n_samples":    n_samples,
        "n_train":      n_train,
        "n_test":       n_test,
        "class_counts": class_counts,
        "anomaly_label": anomaly_label,
        "anomaly_rate":  anomaly_rate,
    }


# ══════════════════════════════════════════════
#  SERIAL / MODEL HELPERS  (Live Monitor)
# ══════════════════════════════════════════════

def load_model():
    """Load the trained ML model from disk. Returns model or None."""
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"[load_model] '{MODEL_PATH}' not found.")
            return None
        m = joblib.load(MODEL_PATH)
        print(f"[load_model] Model loaded from '{MODEL_PATH}'.")
        return m
    except Exception as exc:
        print(f"[load_model] Error: {exc}")
        return None


def get_serial_connection():
    """Open (or reuse) a serial connection to the Arduino."""
    global ser
    if ser is not None and ser.is_open:
        return ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)  # allow Arduino to reset
        print(f"[serial] Connected on {SERIAL_PORT} @ {BAUD_RATE} baud.")
        return ser
    except serial.SerialException as exc:
        print(f"[serial] SerialException: {exc}")
        return None
    except Exception as exc:
        print(f"[serial] Unexpected error: {exc}")
        return None


def read_sensor_data(ser_conn):
    """
    Read two lines from Arduino:
      Line 1 → temperature (float)
      Line 2 → vibration   (float)

    Returns (temperature, vibration, error_msg).
    """
    try:
        temp_line = ser_conn.readline().decode("utf-8").strip()
        if not temp_line:
            return None, None, "No temperature data received."

        vib_line = ser_conn.readline().decode("utf-8").strip()
        if not vib_line:
            return None, None, "No vibration data received."

        try:
            temperature = float(temp_line)
        except ValueError:
            return None, None, f"Cannot parse temperature: '{temp_line}'"

        try:
            vibration = float(vib_line)
        except ValueError:
            return None, None, f"Cannot parse vibration: '{vib_line}'"

        return temperature, vibration, None

    except serial.SerialTimeoutException:
        return None, None, "Serial read timeout."
    except UnicodeDecodeError:
        return None, None, "Received non-text data from serial port."
    except Exception as exc:
        return None, None, f"Serial read error: {exc}"


def predict_status(temperature, vibration):
    """Run the loaded model and return integer prediction (or None on error)."""
    if model is None:
        return None
    try:
        features = np.array([[temperature, vibration]])
        return int(model.predict(features)[0])
    except Exception as exc:
        print(f"[predict_status] {exc}")
        return None


# ══════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════

@app.route("/")
def home():
    """Redirect to upload page if logged in, otherwise to login."""
    if "user_id" in session:
        return redirect(url_for("dashboard_upload"))
    return redirect(url_for("login"))


# ── Dashboard: Upload & Train ────────────────

@app.route("/dashboard/upload", methods=["GET", "POST"])
@login_required
def dashboard_upload():
    """
    Upload CSV → train RandomForest → store results in-memory + log to DB.
    Renders upload.html (new template name).
    """
    global last_training_results

    message        = None
    error          = None

    if request.method == "POST":
        uploaded_file = request.files.get("csv_file")

        if not uploaded_file or uploaded_file.filename == "":
            error = "Please select a CSV file."
        elif not uploaded_file.filename.lower().endswith(".csv"):
            error = "Only .csv files are accepted."
        else:
            try:
                df       = pd.read_csv(uploaded_file)
                train_out = train_model_from_dataframe(df)

                # Persist globally so the results page can read it
                last_training_results = train_out

                # Reload the global model so live monitor uses the new weights
                global model
                model = train_out["model"]

                # Log to database
                log_training_attempt(session.get("username", "unknown"), train_out)

                message = (
                    f"Model trained successfully — Accuracy: "
                    f"{train_out['accuracy']*100:.2f}%  |  "
                    "Saved to motor_model.pkl."
                )
            except Exception as exc:
                error = f"Training failed: {exc}"

    return render_template(
        "upload.html",
        message=message,
        error=error,
    )


# ── Dashboard: Results ───────────────────────

@app.route("/dashboard")
@app.route("/dashboard/results")
@login_required
def dashboard():
    """Show last training metrics and plots."""
    global last_training_results

    results = None
    if last_training_results:
        # Wrap dict in a lightweight object so templates use dot notation
        results = type("Results", (), last_training_results)()

    cache_buster = str(int(time.time()))

    return render_template(
        "dashboard.html",
        results=results,
        cache_buster=cache_buster,
    )


# ── Training Logs ────────────────────────────

@app.route("/logs")
@login_required
def logs():
    """Fetch training history from the database (most recent first)."""
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT
            created_at, username, accuracy,
            n_samples,  n_train,  n_test,
            anomaly_rate, roc_auc
        FROM training_logs
        ORDER BY datetime(created_at) DESC
        LIMIT 50
    """)
    rows = cur.fetchall()
    conn.close()
    return render_template("logs.html", logs=rows)


# ── ML Implementation Info ───────────────────

@app.route("/ml-info")
@login_required
def ml_info():
    """Static page explaining the ML pipeline with code snippets."""
    return render_template("ml_info.html")


# ── Unlabeled Prediction ─────────────────────

# In-memory store for the last unlabeled prediction result (DataFrame + stats)
last_unlabeled_result = None   # dict: { df, total, normal, fault, fault_pct }


@app.route("/predict-unlabeled", methods=["GET", "POST"])
@login_required
def predict_unlabeled():
    """
    Unlabeled Prediction Mode.

    GET  → render the upload form.
    POST → accept a CSV with columns [temperature, vibration],
           run model.predict(), return summary stats + first 10 rows.
    """
    global last_unlabeled_result, model

    error   = None
    result  = None

    if request.method == "POST":
        uploaded_file = request.files.get("csv_file")

        if not uploaded_file or uploaded_file.filename == "":
            error = "Please select a CSV file."
        elif not uploaded_file.filename.lower().endswith(".csv"):
            error = "Only .csv files are accepted."
        else:
            try:
                df = pd.read_csv(uploaded_file)

                # ── Column validation ───────────────────────────
                required_cols = {"temperature", "vibration"}
                missing = required_cols - set(df.columns)
                if missing:
                    raise ValueError(
                        f"CSV is missing required column(s): {', '.join(sorted(missing))}. "
                        "Please include 'temperature' and 'vibration' columns."
                    )

                if len(df) == 0:
                    raise ValueError("The uploaded file contains no data rows.")

                # ── Load model if not in memory ─────────────────
                if model is None:
                    model = load_model()
                if model is None:
                    raise ValueError(
                        "No trained model found. Please upload a labelled CSV "
                        "and train the model first."
                    )

                # ── Prediction ──────────────────────────────────
                # The model was trained on ["temperature", "vib_rms"],
                # so we rename "vibration" → "vib_rms" to match.
                X = df[["temperature", "vibration"]].astype(float)
                X = X.rename(columns={"vibration": "vib_rms"})
                predictions = model.predict(X)
                df["prediction"] = predictions.astype(int)

                # ── Statistics ─────────────────────────────────
                total      = len(df)
                normal_cnt = int((df["prediction"] == 0).sum())
                fault_cnt  = int((df["prediction"] == 1).sum())
                fault_pct  = round(fault_cnt / total * 100, 2) if total > 0 else 0.0

                # Store globally for CSV download
                last_unlabeled_result = {
                    "df":        df,
                    "total":     total,
                    "normal":    normal_cnt,
                    "fault":     fault_cnt,
                    "fault_pct": fault_pct,
                }

                # Preview: first 10 FAULT rows (prediction == 1)
                fault_df = df[df["prediction"] == 1]
                preview_rows = fault_df.head(10).to_dict(orient="records")

                result = {
                    "total":       total,
                    "normal":      normal_cnt,
                    "fault":       fault_cnt,
                    "fault_pct":   fault_pct,
                    "preview":     preview_rows,
                    "is_critical": fault_pct >= 20,
                }

            except Exception as exc:
                error = str(exc)

    return render_template(
        "predict_unlabeled.html",
        result=result,
        error=error,
        has_download=(last_unlabeled_result is not None),
    )


@app.route("/predict-unlabeled/download")
@login_required
def predict_unlabeled_download():
    """Return the last unlabeled prediction result as a downloadable CSV."""
    global last_unlabeled_result

    if last_unlabeled_result is None:
        return redirect(url_for("predict_unlabeled"))

    df = last_unlabeled_result["df"]
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    return Response(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=predictions.csv",
            "Content-Length": str(len(csv_bytes)),
        },
    )


# ── Live Monitor ─────────────────────────────

@app.route("/monitor")
def index():
    """
    Live motor fault detection page.
    Reads sensor data from Arduino, runs prediction, renders monitor.html.
    The page does NOT use meta-refresh; instead app.js uses JS location.reload().
    """
    temperature  = None
    vibration    = None
    status_text  = "WAITING"
    status_color = "#fbbf24"     # yellow
    status_class = "status-waiting"
    status_emoji = "⏳"
    error_msg    = None

    if model is None:
        error_msg    = "Model not loaded. Run a training session first."
        status_text  = "NO MODEL"
        status_emoji = "🤖"
    else:
        ser_conn = get_serial_connection()
        if ser_conn is None:
            error_msg    = f"Could not open serial port {SERIAL_PORT}."
            status_text  = "NO SERIAL"
            status_emoji = "🔌"
            status_color = "#fb923c"
            status_class = "status-error"
        else:
            temperature, vibration, error_msg = read_sensor_data(ser_conn)

            if error_msg is None and temperature is not None and vibration is not None:
                prediction = predict_status(temperature, vibration)

                if prediction is None:
                    status_text  = "PREDICTION ERROR"
                    status_color = "#fb923c"
                    status_class = "status-error"
                    status_emoji = "⚠️"
                elif prediction == 0:
                    status_text  = "MOTOR NORMAL"
                    status_color = "#34d399"   # green
                    status_class = "status-normal"
                    status_emoji = "✅"
                elif prediction == 1:
                    status_text  = "MOTOR FAULT"
                    status_color = "#f87171"   # red
                    status_class = "status-fault"
                    status_emoji = "🚨"
                else:
                    status_text  = f"STATUS: {prediction}"
                    status_color = "#f87171"
                    status_class = "status-fault"
                    status_emoji = "⚠️"
            else:
                status_text  = "NO DATA"
                status_color = "#fbbf24"
                status_class = "status-waiting"
                status_emoji = "📡"

    return render_template(
        "monitor.html",
        temperature  = temperature,
        vibration    = vibration,
        status_text  = status_text,
        status_color = status_color,
        status_class = status_class,
        status_emoji = status_emoji,
        error_msg    = error_msg,
    )


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    # Initialise database tables
    init_db()

    # Try to load an existing model so live monitor works immediately
    model = load_model()

    print("=" * 58)
    print("  MotorAI Dashboard  →  http://127.0.0.1:5000")
    print("=" * 58)

    app.run(debug=True, host="0.0.0.0", port=5000)
