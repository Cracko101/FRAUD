# app.py

import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import (Flask, request, render_template, flash,
                   redirect, url_for, session, g) # Added g for context
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import traceback
from functools import wraps
import math
from datetime import datetime, timezone # Import timezone for UTC

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
ARTIFACTS_DIR = 'artifacts'
PREPROCESSOR_FILE = 'upi_fraud_preprocessor.joblib'
MODEL_FILE = 'best_model.keras'
FEATURE_LIST_FILE = 'feature_list.json'
PREDICTION_THRESHOLD = 0.5
ROWS_PER_PAGE = 10

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'change_this_dev_secret_key_in_prod') # Use env var
# Optional: Configure session lifetime if needed
# from datetime import timedelta
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

# Hardcoded credentials (NOT SECURE FOR PRODUCTION)
DEFAULT_USERNAME = 'admin'
DEFAULT_PASSWORD = 'admin'

# --- Load Artifacts ---
preprocessor = None
model = None
feature_list = []
artifacts_loaded = False

try:
    # --- MARKED CHANGE: Consolidate path definitions ---
    artifacts_base = os.path.dirname(__file__) # Get directory of app.py
    preprocessor_path = os.path.join(artifacts_base, ARTIFACTS_DIR, PREPROCESSOR_FILE)
    model_path = os.path.join(artifacts_base, ARTIFACTS_DIR, MODEL_FILE)
    feature_list_path = os.path.join(artifacts_base, ARTIFACTS_DIR, FEATURE_LIST_FILE)

    if os.path.exists(preprocessor_path) and os.path.exists(model_path) and os.path.exists(feature_list_path):
        preprocessor = joblib.load(preprocessor_path)
        model = load_model(model_path)
        with open(feature_list_path, 'r') as f:
            feature_list = json.load(f)
        artifacts_loaded = True
        print("--- Artifacts Loaded Successfully ---")
    else:
         print("--- Warning: One or more artifact files not found. ---")
         if not os.path.exists(preprocessor_path): print(f"- {preprocessor_path} missing")
         if not os.path.exists(model_path): print(f"- {model_path} missing")
         if not os.path.exists(feature_list_path): print(f"- {feature_list_path} missing")

except Exception as e:
    artifacts_loaded = False
    print(f"--- Error loading artifacts: {e} ---")
    traceback.print_exc()

# --- Context Processor ---
@app.context_processor
def inject_global_vars():
    # --- MARKED CHANGE: Fix DeprecationWarning ---
    return dict(
        current_year=datetime.now(timezone.utc).year,  # Use timezone-aware UTC now
        min=min  # Add min function to template context
    )

# --- Authentication Decorator ---
def login_required(f):
    @wraps(f)
    def wrapped_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in to access this page.', 'info')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return wrapped_function

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df, required_features, preprocessor_obj, model_obj):
    """Prepares the dataframe for prediction."""
    # --- MARKED CHANGE: Add specific timestamp format if known ---
    # If your timestamps are consistently in 'YYYY-MM-DD HH:MM:SS' format, specifying it is better.
    # Otherwise, let Pandas infer (accept the UserWarning).
    KNOWN_TIMESTAMP_FORMAT = None # Example: '%Y-%m-%d %H:%M:%S' or None to infer

    print("--- Starting Preprocessing ---")
    df_processed = None
    num_processed_features = 0

    if 'timestamp' in df.columns:
        try:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce', format=KNOWN_TIMESTAMP_FORMAT) # Store as datetime
            if not df['timestamp_dt'].isnull().all():
                df['hour'] = df['timestamp_dt'].dt.hour
                df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
                df['hour'] = df['hour'].fillna(-1).astype(int)
                df['day_of_week'] = df['day_of_week'].fillna(-1).astype(int)
                print("Timestamp features engineered.")
            else:
                print("Warning: Timestamp values failed parsing.")
                if 'hour' not in df.columns: df['hour'] = -1
                if 'day_of_week' not in df.columns: df['day_of_week'] = -1
        except Exception as e:
            print(f"Warning: Timestamp processing error: {e}")
            traceback.print_exc()
            if 'hour' not in df.columns: df['hour'] = -1
            if 'day_of_week' not in df.columns: df['day_of_week'] = -1
    else:
         print("Timestamp column not found.")
         if 'hour' not in df.columns: df['hour'] = -1
         if 'day_of_week' not in df.columns: df['day_of_week'] = -1

    print(f"Required Features: {required_features}")
    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}.")

    try:
        df_selected = df[required_features]
        print(f"Selected {len(df_selected.columns)} features.")
    except KeyError as e:
         raise ValueError(f"Column Error during selection: {e}.")

    try:
        df_processed = preprocessor_obj.transform(df_selected)
        print(f"Data transformed. Shape: {df_processed.shape}")
        if hasattr(df_processed, "toarray"):
            df_processed = df_processed.toarray()
            print("Converted sparse matrix to dense.")
    except Exception as e:
        print(f"Preprocessor transform error: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Error applying preprocessor: {e}.")

    num_processed_features = df_processed.shape[1]
    if model_obj is not None and model_obj.input_shape[1] is not None and num_processed_features != model_obj.input_shape[1]:
         raise ValueError(f"Feature count mismatch: Preprocessor output {num_processed_features}, model expects {model_obj.input_shape[1]}.")

    df_reshaped = df_processed.reshape(df_processed.shape[0], num_processed_features, 1)
    print(f"Data reshaped. Shape: {df_reshaped.shape}")
    print("--- Preprocessing Finished ---")
    return df_reshaped

def run_prediction_and_filter(filepath):
    """Reads CSV, preprocesses, predicts, and returns original df with predictions and filtered fraud df."""
    # --- MARKED CHANGE: Encapsulated prediction logic ---
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed file not found: {filepath}")

    df_input = pd.read_csv(filepath)
    if df_input.empty:
        print("CSV file is empty during re-processing.")
        return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames

    df_original = df_input.copy() # Keep original untouched for filtering later

    # Run preprocessing
    data_ready_for_model = preprocess_data(df_input, feature_list, preprocessor, model) # Pass copy to preprocess

    # Run prediction
    predictions_proba = model.predict(data_ready_for_model)
    predictions = (predictions_proba > PREDICTION_THRESHOLD).astype(int).flatten()

    # Add predictions to the original data copy
    df_original['is_fraud_prediction'] = predictions
    # df_original['fraud_probability'] = predictions_proba.round(4).flatten() # Optional

    # Filter
    fraudulent_df = df_original[df_original['is_fraud_prediction'] == 1].copy()

    # Convert timestamp to datetime objects IN the final fraud df for template use
    if 'timestamp' in fraudulent_df.columns:
        # Specify format if known, otherwise let pandas infer
        fraudulent_df['timestamp_dt_obj'] = pd.to_datetime(fraudulent_df['timestamp'], errors='coerce', format=None) # Create new column

    return df_original, fraudulent_df


# --- Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if session.get('logged_in'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        if request.form.get('username') == DEFAULT_USERNAME and request.form.get('password') == DEFAULT_PASSWORD:
            session['logged_in'] = True
            session.permanent = True
            flash('Login successful!', 'success')
            next_url = request.args.get('next')
            return redirect(next_url or url_for('index'))
        else:
            flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Handles user logout."""
    # --- MARKED CHANGE: Clear result session data on logout ---
    session.pop('logged_in', None)
    session.pop('processed_filepath', None)
    session.pop('summary_results', None)
    session.pop('last_upload_total_fraud', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    """Handles file upload (POST) and displaying results/upload form (GET)."""
    page = request.args.get('page', 1, type=int)
    if page < 1: page = 1

    if not artifacts_loaded:
         flash('Prediction service unavailable: Model artifacts missing.', 'danger')
         return render_template('index.html', artifacts_loaded=artifacts_loaded)

    if request.method == 'POST':
        # --- MARKED CHANGE: Clear previous session results on new upload ---
        session.pop('processed_filepath', None)
        session.pop('summary_results', None)
        session.pop('last_upload_total_fraud', None)

        if 'file' not in request.files or request.files['file'].filename == '':
            flash('No file selected.', 'warning')
            return redirect(request.url)
        file = request.files['file']

        if file and allowed_file(file.filename):
            # --- MARKED CHANGE: Use secure filename and ensure uploads dir exists ---
            os.makedirs(os.path.join(os.path.dirname(__file__), app.config['UPLOAD_FOLDER']), exist_ok=True)
            filename = secure_filename(f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{file.filename}") # Add timestamp prefix
            filepath = os.path.join(os.path.dirname(__file__), app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved: {filepath}")

            try:
                # --- MARKED CHANGE: Run prediction and get both original+preds and filtered fraud ---
                df_original_with_preds, fraudulent_df = run_prediction_and_filter(filepath)
                total_transactions = len(df_original_with_preds)
                total_fraud = len(fraudulent_df)

                # Calculate Specific Summaries
                summary = {
                    "total_transactions": total_transactions,
                    "total_fraud": total_fraud,
                    "total_fraud_amount": None,
                    "merchant_distribution": None,
                    "fraud_rate_percent": round((total_fraud / total_transactions) * 100, 2) if total_transactions > 0 else 0,
                }
                if total_fraud > 0:
                    if 'amount' in fraudulent_df.columns:
                        numeric_amounts = pd.to_numeric(fraudulent_df['amount'], errors='coerce').dropna()
                        if not numeric_amounts.empty: summary["total_fraud_amount"] = numeric_amounts.sum()
                    if 'merchant_category_code' in fraudulent_df.columns:
                        summary["merchant_distribution"] = fraudulent_df['merchant_category_code'].value_counts().to_dict()

                print(f"Summary Stats calculated: {summary}")

                # Store info needed for GET requests in session
                session['processed_filepath'] = filepath
                session['summary_results'] = summary # Store the calculated summary
                session['last_upload_total_fraud'] = total_fraud # Store total count

                # Redirect to the GET request for page 1 of results
                return redirect(url_for('index', page=1))

            except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
                # Handle errors during initial processing
                error_type = type(e).__name__
                print(f"Error during POST processing ({error_type}): {e}")
                traceback.print_exc()
                flash(f"Error processing file: {e}", 'danger')
                # Clean up failed upload if file exists
                if os.path.exists(filepath):
                    try: os.remove(filepath)
                    except OSError as remove_error: print(f"Error removing file after error: {remove_error}")
                # Redirect back to the index (GET) to show the upload form with error
                return redirect(url_for('index'))
            # Note: No 'finally' block needed here for file removal, handled on error or assumes success leads to session storage

        else: # Invalid file type
            flash('Invalid file type. Please upload a CSV.', 'warning')
            return redirect(request.url)

    # --- GET Request Handling ---
    # Check session for results from a previous POST
    processed_filepath = session.get('processed_filepath')
    summary = session.get('summary_results')
    total_fraud = session.get('last_upload_total_fraud')

    if processed_filepath and summary is not None and total_fraud is not None:
        # Results exist in session, display the dashboard
        transactions_on_page = []
        total_pages = 0
        message = None # No specific message unless reprocessing fails

        try:
            if total_fraud > 0:
                 # --- MARKED CHANGE: Rerun prediction/filter only if fraud exists ---
                 _, fraudulent_df = run_prediction_and_filter(processed_filepath) # Rerun to get filtered df

                 total_pages = math.ceil(total_fraud / ROWS_PER_PAGE)
                 if page > total_pages: page = total_pages
                 elif page < 1: page = 1

                 start_index = (page - 1) * ROWS_PER_PAGE
                 end_index = start_index + ROWS_PER_PAGE

                 # Select ONLY needed columns and slice for pagination
                 columns_to_display = ['transaction_id', 'user_id', 'merchant_id',
                                       'merchant_category_code', 'amount', 'timestamp_dt_obj', # Use the datetime obj column
                                       'description']
                 display_df = pd.DataFrame()
                 for col in columns_to_display:
                      internal_col_name = col if col != 'timestamp_dt_obj' else 'timestamp' # Map back for selection if needed
                      source_col_name = 'timestamp_dt_obj' if col == 'timestamp_dt_obj' else col # Use correct source name

                      if source_col_name in fraudulent_df.columns:
                          display_df[col] = fraudulent_df[source_col_name]
                      else:
                          display_df[col] = None

                 paginated_df = display_df.iloc[start_index:end_index]
                 transactions_on_page = paginated_df.to_dict(orient='records')

                 # Rename 'timestamp_dt_obj' back to 'timestamp' for template consistency
                 for tx in transactions_on_page:
                    if 'timestamp_dt_obj' in tx:
                        tx['timestamp'] = tx.pop('timestamp_dt_obj') # Rename key
                    # Handle potential NaT from reprocessing
                    if 'timestamp' in tx and pd.isna(tx['timestamp']):
                       tx['timestamp'] = None


            # Render results with data (even if total_fraud is 0, show summary)
            return render_template('results.html',
                                   transactions_on_page=transactions_on_page,
                                   summary=summary,
                                   current_page=page,
                                   total_pages=total_pages,
                                   total_fraud=total_fraud,
                                   ROWS_PER_PAGE=ROWS_PER_PAGE)  # Add ROWS_PER_PAGE to template context

        except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
             # Handle errors during GET request reprocessing
             error_type = type(e).__name__
             print(f"Error during GET processing ({error_type}): {e}")
             traceback.print_exc()
             flash(f"Error retrieving results: {e}. Please try uploading again.", 'danger')
             # Clear potentially corrupt session data
             session.pop('processed_filepath', None)
             session.pop('summary_results', None)
             session.pop('last_upload_total_fraud', None)
             # Show the upload form again
             return render_template('index.html', artifacts_loaded=artifacts_loaded)

    else:
        # No results in session, show the upload form
        return render_template('index.html', artifacts_loaded=artifacts_loaded)


# --- Run the App ---
if __name__ == '__main__':
    # --- MARKED CHANGE: Use secure path creation ---
    upload_dir = os.path.join(os.path.dirname(__file__), app.config['UPLOAD_FOLDER'])
    os.makedirs(upload_dir, exist_ok=True)

    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    host = os.environ.get('FLASK_RUN_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_RUN_PORT', 5000))
    app.run(debug=debug, host=host, port=port)