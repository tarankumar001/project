"""
Motor Fault Detection Web App (Flask) + ML Admin Dashboard

Part 1 (existing):
- Loads trained model from motor_model.csv using joblib
- Connects to Arduino on COM3 (9600 baud)
- Arduino sends two lines:
    1) temperature (float)
    2) vibration value (float)
- Uses temperature and vibration as features for prediction
- Shows MOTOR NORMAL (green) or MOTOR FAULT (red) on a web page
- Page auto-refreshes every 2 seconds

Part 2 (admin dashboard):
- Web UI to upload CSV file with columns: temperature, vibration, label
- Trains a RandomForestClassifier on uploaded data
- Shows Accuracy, Confusion Matrix, ROC Curve, Feature Importance
- Saves plots into static/plots and model into motor_model.csv
"""

from flask import (
    Flask,
    render_template,
    render_template_string,
    request,
    redirect,
    url_for,
    session,
)
import serial
import joblib
import numpy as np
import time
import os
import sqlite3
from functools import wraps
import datetime

from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

import matplotlib

# Use a non-GUI backend so plotting works on servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==== Configuration ====
SERIAL_PORT = "COM3"
BAUD_RATE = 9600
MODEL_PATH = "motor_model.pkl"
PLOTS_DIR = os.path.join("static", "plots")
DATABASE = os.path.join(os.path.dirname(__file__), "users.db")

# Global objects (reused to keep things simple)
model = None
ser = None
last_training_results = None  # store last training metrics for dashboard

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key"  # for sessions; replace in production


def log_training_attempt(username, train_out):
    """Store a single training run in the training_logs table."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        created_at = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        cur.execute(
            """
            INSERT INTO training_logs (
                created_at,
                username,
                accuracy,
                n_samples,
                n_train,
                n_test,
                anomaly_rate,
                roc_auc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                username,
                float(train_out.get("accuracy", 0.0)),
                int(train_out.get("n_samples", 0)),
                int(train_out.get("n_train", 0)),
                int(train_out.get("n_test", 0)),
                float(train_out.get("anomaly_rate") or 0.0)
                if train_out.get("anomaly_rate") is not None
                else None,
                float(train_out.get("roc_auc", 0.0))
                if train_out.get("roc_auc") is not None
                else None,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        # Log to console; we don't want logging errors to break the UI
        print(f"Failed to write training log: {exc}")


def get_db_connection():
    """Create a new connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create database tables if they do not exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    # Users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    # Training logs table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            username TEXT NOT NULL,
            accuracy REAL,
            n_samples INTEGER,
            n_train INTEGER,
            n_test INTEGER,
            anomaly_rate REAL,
            roc_auc REAL
        )
        """
    )
    conn.commit()
    conn.close()


def login_required(view_func):
    """Decorator to restrict access to logged-in users only."""
    @wraps(view_func)
    def wrapped_view(**kwargs):
        if "user_id" not in session:
            return redirect(url_for("login", next=request.path))
        return view_func(**kwargs)

    return wrapped_view


@app.context_processor
def inject_user():
    """Make logged-in state available in all templates."""
    return {
        "logged_in": "user_id" in session,
        "current_user": session.get("username"),
    }


# ========== AUTHENTICATION ROUTES ==========

@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration page."""
    error = None
    message = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm_password", "").strip()

        if not username or not password or not confirm:
            error = "Please fill in all fields."
        elif password != confirm:
            error = "Passwords do not match."
        elif len(password) < 4:
            error = "Password should be at least 4 characters long."
        else:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT id FROM users WHERE username = ?", (username,))
                existing = cur.fetchone()
                if existing:
                    error = "Username is already taken. Please choose another one."
                else:
                    password_hash = generate_password_hash(password)
                    cur.execute(
                        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                        (username, password_hash),
                    )
                    conn.commit()
                    message = "Registration successful. You can now log in."
                conn.close()
            except Exception as exc:
                error = f"Registration failed: {exc}"

    return render_template("register.html", error=error, message=message)


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login page."""
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            error = "Please enter both username and password."
        else:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, username, password_hash FROM users WHERE username = ?",
                    (username,),
                )
                row = cur.fetchone()
                conn.close()

                if row and check_password_hash(row["password_hash"], password):
                    # Login successful
                    session["user_id"] = row["id"]
                    session["username"] = row["username"]
                    next_page = request.args.get("next")
                    return redirect(next_page or url_for("dashboard"))
                else:
                    error = "Invalid username or password."
            except Exception as exc:
                error = f"Login failed: {exc}"

    return render_template("login.html", error=error)


@app.route("/logout")
@login_required
def logout():
    """Log the current user out and clear the session."""
    session.clear()
    return redirect(url_for("login"))


def load_model():
    """
    Load the trained machine learning model from disk.
    Returns the model object, or None if loading fails.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Model file '{MODEL_PATH}' not found. Please run training first.")
            return None

        loaded_model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from '{MODEL_PATH}'.")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_serial_connection():
    """
    Create (or reuse) a serial connection to the Arduino.
    Returns a serial object or None if connection fails.
    """
    global ser

    # If we already have an open connection, reuse it
    if ser is not None and ser.is_open:
        return ser

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        # Small delay so Arduino can reset and start sending data
        time.sleep(2)
        print(f"Connected to Arduino on {SERIAL_PORT} at {BAUD_RATE} baud.")
        return ser
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error opening serial port: {e}")
        return None


def read_sensor_data(ser_conn):
    """
    Read two lines from Arduino:
      1) temperature (float)
      2) vibration (float)

    Returns:
        (temperature, vibration, error_message)
        - temperature (float or None)
        - vibration (float or None)
        - error_message (str or None)
    """
    try:
        # Line 1: temperature
        temp_line = ser_conn.readline().decode("utf-8").strip()
        if not temp_line:
            return None, None, "No temperature data received"

        # Line 2: vibration
        vib_line = ser_conn.readline().decode("utf-8").strip()
        if not vib_line:
            return None, None, "No vibration data received"

        # Convert to floats
        try:
            temperature = float(temp_line)
        except ValueError:
            return None, None, f"Could not parse temperature from '{temp_line}'"

        try:
            vibration = float(vib_line)
        except ValueError:
            return None, None, f"Could not parse vibration from '{vib_line}'"

        return temperature, vibration, None

    except serial.SerialTimeoutException:
        return None, None, "Serial read timeout"
    except UnicodeDecodeError:
        return None, None, "Received non-text data from serial"
    except Exception as e:
        return None, None, f"Error reading from serial: {e}"


def predict_status(temperature, vibration):
    """
    Use the trained model to predict motor status.
    Returns:
        prediction (int or None)
    """
    if model is None:
        return None

    try:
        # Model expects a 2D array: [[temperature, vibration]]
        features = np.array([[temperature, vibration]])
        prediction = model.predict(features)[0]
        return int(prediction)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


@app.route("/monitor")
def index():
    """
    Live motor fault detection page:
    - Reads the latest sensor values
    - Runs prediction
    - Returns a minimal HTML page that auto-refreshes every 2 seconds
    """
    # Default display values
    temperature = None
    vibration = None
    status_text = "WAITING"
    status_color = "#7f8c8d"  # grey
    error_msg = None

    # Check model
    if model is None:
        error_msg = "Model not loaded. Please ensure 'motor_model.pkl' exists."
    else:
        # Get (or open) serial connection
        ser_conn = get_serial_connection()
        if ser_conn is None:
            error_msg = f"Could not open serial port {SERIAL_PORT}."
        else:
            # Read data from Arduino
            temperature, vibration, error_msg = read_sensor_data(ser_conn)

            if error_msg is None and temperature is not None and vibration is not None:
                # Make prediction
                prediction = predict_status(temperature, vibration)

                if prediction is None:
                    status_text = "PREDICTION ERROR"
                    status_color = "#e67e22"  # orange
                else:
                    if prediction == 0:
                        status_text = "MOTOR NORMAL"
                        status_color = "#2ecc71"  # green
                    elif prediction == 1:
                        status_text = "MOTOR FAULT"
                        status_color = "#e74c3c"  # red
                    else:
                        # Any other label is treated as fault-like
                        status_text = f"STATUS: {prediction}"
                        status_color = "#e74c3c"
            else:
                # We had a read error; keep "WAITING" text but set color to orange
                status_text = "NO DATA"
                status_color = "#f39c12"  # yellow/orange

    # Prepare optional error HTML (to keep the main f-string simple)
    error_html = f'<div class="error">{error_msg}</div>' if error_msg else ""

    # Build simple HTML with inline CSS
    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Motor Fault Detection</title>
    <!-- Auto-refresh every 2 seconds -->
    <meta http-equiv="refresh" content="2">
    <style>
      body {{
        margin: 0;
        padding: 0;
        font-family: Arial, sans-serif;
        background-color: #f4f4f4;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
      }}
      .card {{
        background-color: #ffffff;
        padding: 24px 32px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        min-width: 320px;
        text-align: center;
      }}
      h1 {{
        margin-top: 0;
        margin-bottom: 16px;
        font-size: 24px;
        color: #333333;
      }}
      .value-row {{
        margin: 8px 0;
        font-size: 18px;
      }}
      .label {{
        font-weight: bold;
        margin-right: 8px;
      }}
      .status {{
        margin-top: 18px;
        padding: 12px;
        border-radius: 6px;
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
        background-color: {status_color};
      }}
      .error {{
        margin-top: 10px;
        color: #e74c3c;
        font-size: 14px;
      }}
      .footer {{
        margin-top: 12px;
        font-size: 12px;
        color: #888888;
      }}
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Motor Fault Detection</h1>

      <div class="value-row">
        <span class="label">Temperature:</span>
        <span>{f"{temperature:.2f} °C" if temperature is not None else "N/A"}</span>
      </div>

      <div class="value-row">
        <span class="label">Vibration:</span>
        <span>{f"{vibration:.3f}" if vibration is not None else "N/A"}</span>
      </div>

      <div class="status">{status_text}</div>

      {error_html}

      <div class="footer">
        Page auto-refreshes every 2 seconds.
      </div>
    </div>
  </body>
  </html>
"""

    return html


def train_model_from_dataframe(df):
    """
    Train a RandomForest model from a pandas DataFrame.
    The DataFrame must contain: temperature, vibration, label.

    Returns a dictionary with:
      - model: trained model
      - accuracy: float
      - cm: confusion matrix (2D array)
      - classes: list of class labels
      - roc_auc: float or None
      - plots: dict of plot filenames
      - n_samples: total number of rows
      - n_train: number of training samples
      - n_test: number of test samples
      - class_counts: dict mapping label -> count
    """
    # Basic validation
    required_cols = {"temperature", "vibration", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    if len(df) < 10:
        raise ValueError("Not enough rows in CSV. Please provide at least 10 rows.")

    # "Compute" vibration RMS (here we assume vibration column already stores RMS)
    # For clarity we copy it into a new column.
    df["vib_rms"] = df["vibration"].astype(float)

    # Features and target
    X = df[["temperature", "vib_rms"]].astype(float)
    y = df["label"]

    if y.nunique() < 2:
        raise ValueError("Need at least two different label values for classification.")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classes = list(sorted(y.unique()))

    # Sample counts for dashboard display
    n_samples = len(df)
    n_train = len(X_train)
    n_test = len(X_test)
    class_counts = y.value_counts().to_dict()

    # ROC curve (only for binary classification)
    roc_auc = None
    fpr = tpr = None
    anomaly_label = None
    anomaly_rate = None
    if len(classes) == 2:
        anomaly_label = classes[1]  # treat the second class as "anomaly/fault"
        y_test_binary = (y_test == anomaly_label).astype(int)
        fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, 1])
        roc_auc = roc_auc_score(y_test_binary, y_proba[:, 1])

        # Approximate anomaly rate in full dataset
        anomaly_count = class_counts.get(anomaly_label, 0)
        anomaly_rate = anomaly_count / float(n_samples) if n_samples > 0 else None

    # Ensure plots directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plots = {}

    # Confusion matrix plot
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # Add numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2.0 else "black",
            )
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=120)
    plt.close()
    plots["confusion_matrix"] = "plots/confusion_matrix.png"

    # ROC curve plot (if available)
    if roc_auc is not None and fpr is not None and tpr is not None:
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=120)
        plt.close()
        plots["roc_curve"] = "plots/roc_curve.png"

    # Feature importance plot
    importances = clf.feature_importances_
    feature_names = ["temperature", "vib_rms"]

    plt.figure(figsize=(4, 4))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, importances, align="center", color="#3498db")
    plt.yticks(y_pos, feature_names)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(fi_path, dpi=120)
    plt.close()
    plots["feature_importance"] = "plots/feature_importance.png"

    # Save trained model
    joblib.dump(clf, MODEL_PATH)

    return {
        "model": clf,
        "accuracy": acc,
        "cm": cm,
        "classes": classes,
        "roc_auc": roc_auc,
        "plots": plots,
        "n_samples": n_samples,
        "n_train": n_train,
        "n_test": n_test,
        "class_counts": class_counts,
        "anomaly_label": anomaly_label,
        "anomaly_rate": anomaly_rate,
    }


# Simple Bootstrap-based admin dashboard template
ADMIN_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ML Admin Dashboard</title>
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
      rel="stylesheet"
    >
  </head>
  <body class="bg-light">
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">Motor ML Admin</a>
        <div class="d-flex">
          <a href="/" class="btn btn-outline-light btn-sm">Live Monitor</a>
        </div>
      </div>
    </nav>

    <div class="container">
      <div class="row">
        <div class="col-lg-4">
          <div class="card mb-4">
            <div class="card-header">
              <strong>Upload Training Data (CSV)</strong>
            </div>
            <div class="card-body">
              <form method="post" enctype="multipart/form-data">
                <div class="mb-3">
                  <label for="csvFile" class="form-label">CSV file</label>
                  <input
                    class="form-control"
                    type="file"
                    id="csvFile"
                    name="csv_file"
                    accept=".csv"
                    required
                  >
                </div>
                <p class="small text-muted mb-2">
                  CSV must contain columns:
                  <code>temperature</code>, <code>vibration</code>, <code>label</code>.
                </p>
                <button type="submit" class="btn btn-primary w-100">
                  Train Model
                </button>
              </form>

              {% if message %}
              <div class="alert alert-success mt-3" role="alert">
                {{ message }}
              </div>
              {% endif %}

              {% if error %}
              <div class="alert alert-danger mt-3" role="alert">
                {{ error }}
              </div>
              {% endif %}
            </div>
          </div>
        </div>

        <div class="col-lg-8">
          <div class="card mb-4">
            <div class="card-header">
              <strong>Training Results</strong>
            </div>
            <div class="card-body">
              {% if results %}
                <p><strong>Accuracy:</strong> {{ (results.accuracy * 100) | round(2) }}%</p>

                {% if results.roc_auc is not none %}
                  <p><strong>ROC AUC:</strong> {{ results.roc_auc | round(4) }}</p>
                {% endif %}

                <div class="row mt-3">
                  {% if 'confusion_matrix' in results.plots %}
                  <div class="col-md-6 mb-3">
                    <h6>Confusion Matrix</h6>
                    <img
                      src="{{ url_for('static', filename=results.plots.confusion_matrix) }}?v={{ cache_buster }}"
                      class="img-fluid img-thumbnail"
                      alt="Confusion Matrix"
                    >
                  </div>
                  {% endif %}

                  {% if 'roc_curve' in results.plots %}
                  <div class="col-md-6 mb-3">
                    <h6>ROC Curve</h6>
                    <img
                      src="{{ url_for('static', filename=results.plots.roc_curve) }}?v={{ cache_buster }}"
                      class="img-fluid img-thumbnail"
                      alt="ROC Curve"
                    >
                  </div>
                  {% endif %}

                  {% if 'feature_importance' in results.plots %}
                  <div class="col-md-6 mb-3">
                    <h6>Feature Importance</h6>
                    <img
                      src="{{ url_for('static', filename=results.plots.feature_importance) }}?v={{ cache_buster }}"
                      class="img-fluid img-thumbnail"
                      alt="Feature Importance"
                    >
                  </div>
                  {% endif %}
                </div>
              {% else %}
                <p class="text-muted mb-0">
                  No results yet. Upload a CSV file to train a model.
                </p>
              {% endif %}
            </div>
          </div>
        </div>
      </div>
    </div>
  </body>
</html>
"""


@app.route("/admin", methods=["GET", "POST"])
def admin_dashboard():
    """
    Simple ML admin dashboard:
    - Upload CSV
    - Train RandomForest
    - Show metrics and plots
    """
    results = None
    message = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("csv_file")
        if not uploaded_file or uploaded_file.filename == "":
            error = "Please choose a CSV file to upload."
        elif not uploaded_file.filename.lower().endswith(".csv"):
            error = "Only .csv files are supported."
        else:
            try:
                # Read CSV directly into pandas
                df = pd.read_csv(uploaded_file)

                train_out = train_model_from_dataframe(df)
                results = type("Results", (), train_out)  # simple object-like access
                message = "Model trained successfully and saved to 'motor_model.pkl'."
            except Exception as exc:
                error = f"Training failed: {exc}"

    cache_buster = str(int(time.time()))

    return render_template_string(
        ADMIN_TEMPLATE,
        results=results,
        message=message,
        error=error,
        cache_buster=cache_buster,
    )


@app.route("/")
def home():
    """Redirect users to upload page if logged in, otherwise to login."""
    if "user_id" in session:
        return redirect(url_for("dashboard_upload"))
    return redirect(url_for("login"))


@app.route("/dashboard/upload", methods=["GET", "POST"])
@login_required
def dashboard_upload():
    """
    Page to upload CSV and train the model.
    Metrics and plots will appear on the Results page.
    """
    global last_training_results

    message = None
    error = None
    show_animation = False

    if request.method == "POST":
        uploaded_file = request.files.get("csv_file")
        if not uploaded_file or uploaded_file.filename == "":
            error = "Please choose a CSV file to upload."
        elif not uploaded_file.filename.lower().endswith(".csv"):
            error = "Only .csv files are supported."
        else:
            try:
                df = pd.read_csv(uploaded_file)
                train_out = train_model_from_dataframe(df)
                last_training_results = train_out  # persist across pages
                # Log this attempt to the database
                log_training_attempt(session.get("username", "unknown"), train_out)

                message = "Model trained successfully and saved to 'motor_model.pkl'. Open the Results tab to view metrics."
                show_animation = True
            except Exception as exc:
                error = f"Training failed: {exc}"

    return render_template(
        "dashboard_upload.html",
        message=message,
        error=error,
        show_animation=show_animation,
    )


@app.route("/dashboard")
@app.route("/dashboard/results")
@login_required
def dashboard():
    """
    Results page: shows last training metrics and plots.
    """
    global last_training_results

    results = None
    if last_training_results:
        results = type("Results", (), last_training_results)

    cache_buster = str(int(time.time()))

    return render_template(
        "dashboard.html",
        results=results,
        cache_buster=cache_buster,
    )


@app.route("/logs")
@login_required
def logs():
    """Show history of training runs from the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            created_at,
            username,
            accuracy,
            n_samples,
            n_train,
            n_test,
            anomaly_rate,
            roc_auc
        FROM training_logs
        ORDER BY datetime(created_at) DESC
        LIMIT 50
        """
    )
    rows = cur.fetchall()
    conn.close()

    return render_template("logs.html", logs=rows)


if __name__ == "__main__":
    # Initialize user database (creates table if needed)
    init_db()

    # Optionally load existing model at startup (for live monitor route)
    model = load_model()

    print("Starting Flask development server on http://127.0.0.1:5000")
    # debug=True for easier development; set to False for production
    app.run(debug=True)

