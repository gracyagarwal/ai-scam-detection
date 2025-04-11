from flask import Blueprint, request, jsonify, render_template_string, current_app
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
import re
import ssl
import socket
import certifi
import hashlib
import validators
from datetime import datetime
from urllib.parse import urlparse
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from config import VIRUSTOTAL_API_KEY

qr_bp = Blueprint('qr', __name__, url_prefix='/qr')

# Set up paths and allowed extensions
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'qr_malware_model.keras')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the fallback CNN model (trained with augmentation and regularization)
model = tf.keras.models.load_model(MODEL_PATH)

# HTML template
QR_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>QR Code Malware Checker</title>
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: Arial, sans-serif;
            padding: 40px;
            text-align: center;
        }
        form { margin-bottom: 30px; }
        input[type="file"] {
            background-color: #1e1e1e;
            border: none;
            color: white;
            padding: 12px;
            font-size: 16px;
            border-radius: 8px;
        }
        button {
            background-color: #0d6efd;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 18px;
            border-radius: 8px;
            cursor: pointer;
            margin-left: 10px;
        }
        button:hover { background-color: #0b5ed7; }
        .preview-result-wrapper {
            display: flex;
            justify-content: center;
            gap: 40px;
            align-items: flex-start;
            margin-top: 30px;
            flex-wrap: wrap;
        }
        .preview { max-width: 250px; border-radius: 8px; }
        .result {
            padding: 25px;
            border-radius: 10px;
            font-size: 18px;
            min-width: 300px;
            text-align: left;
        }
        .benign { background-color: #1e4423; color: #c7f5d3; }
        .malicious { background-color: #5a1b1b; color: #ffd6d6; }
        a {
            color: #10a37f;
            display: inline-block;
            margin-top: 40px;
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>QR Code Malware Checker</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" id="fileInput" accept="image/*" required>
        <button type="submit">Check</button>
    </form>
    {% if result %}
    <div class="preview-result-wrapper">
        <img id="preview" class="preview" src="{{ image_url }}" alt="QR Preview">
        <div class="result {{ 'malicious' if is_malicious else 'benign' }}">
            <p><strong>Status:</strong> {{ result }}</p>
            <p><strong>QR Content:</strong> {{ qr_data }}</p>
        </div>
    </div>
    {% else %}
    <img id="preview" class="preview" src="#" alt="QR Preview" style="display: none;">
    {% endif %}
    <div style="margin-top: 30px; text-align: center;">
        <a href="{{ url_for('home') }}">
            <button style="padding: 8px 16px; background-color: #444; color: white; border: none; border-radius: 5px; font-family: 'Segoe UI', Tahoma, sans-serif;">
            ← Back to Home
            </button>
        </a>
    </div>
    <script>
        const input = document.getElementById("fileInput");
        const preview = document.getElementById("preview");
        input.onchange = evt => {
            const [file] = input.files;
            if (file) {
                preview.src = URL.createObjectURL(file);
                preview.style.display = 'block';
            }
        };
    </script>
</body>
</html>
"""

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    return np.expand_dims(image, axis=[0, -1])

def extract_qr_data(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(thresh)
    if retval and decoded_info:
        # Return the first non-empty string
        for item in decoded_info:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None

def extract_url(text):
    match = re.search(r'(https?://[^\s]+)', text)
    return match.group(1) if match else None

def is_url_safe(url):
    try:
        if not url.lower().startswith(('http://', 'https://')):
            return False
        response = requests.head(url, timeout=3, allow_redirects=True)
        is_https = url.lower().startswith("https")
        is_ok = response.status_code < 400
        content_type = response.headers.get("Content-Type", "")
        is_html = "text/html" in content_type
        return is_https and is_ok and is_html
    except Exception as e:
        print(f"[URL CHECK ERROR] {e}")
        return False

def check_url_reputation(url):
    try:
        url_id = hashlib.sha256(url.encode()).hexdigest()
        headers = {"accept": "application/json", "x-apikey": VIRUSTOTAL_API_KEY}
        r = requests.get(f"https://www.virustotal.com/api/v3/urls/{url_id}", headers=headers)
        return r.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
    except:
        return None

def check_ssl_cert(domain):
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with socket.create_connection((domain, 443), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname=domain):
                return True
    except:
        return False

def check_domain_age(domain):
    try:
        w = whois.whois(domain)
        creation = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        return (datetime.now() - creation).days > 30 if creation else False
    except:
        return False

def analyze_url_pattern(url):
    patterns = [r'bit\.ly', r'goo\.gl', r'tinyurl', r'login|signin|account|password', r'verify|confirm|update|secure']
    return any(re.search(p, url.lower()) for p in patterns)

# ==== Flask Route ==== 
@qr_bp.route('/', methods=['GET', 'POST'])
def qr_home():
    result = None
    is_malicious = False
    confidence = 0.0
    qr_data = ''
    image_url = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Convert image to base64 for preview display
            import base64
            with open(filepath, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_url = f"data:image/png;base64,{image_data}"

            try:
                raw_data = extract_qr_data(filepath)
                # Clean the QR data (if it contains extra text, extract the URL)
                if raw_data:
                    cleaned_data = extract_url(raw_data) or raw_data.split('\n')[0].strip()
                else:
                    cleaned_data = ''

                qr_data = cleaned_data

                # Primary: URL Analysis if a valid URL is found
                if cleaned_data and validators.url(cleaned_data):
                    # URL analysis using concurrent checks
                    domain = urlparse(cleaned_data).netloc
                    with ThreadPoolExecutor(max_workers=4) as ex:
                        ssl_valid = ex.submit(check_ssl_cert, domain).result()
                        age_ok = ex.submit(check_domain_age, domain).result()
                        vt_rep = ex.submit(check_url_reputation, cleaned_data).result()
                        suspicious = ex.submit(analyze_url_pattern, cleaned_data).result()

                    # Build a hybrid score (start from 0.5 and adjust based on URL checks)
                    final_score = 0.5
                    if ssl_valid:
                        final_score -= 0.1
                    if age_ok:
                        final_score -= 0.1
                    if vt_rep and vt_rep.get('malicious', 0) == 0:
                        final_score -= 0.1
                    if suspicious:
                        final_score += 0.2

                    confidence = (1 - final_score) * 100 if final_score > 0.5 else final_score * 100
                    is_malicious = final_score > 0.5
                    result = "Malicious" if is_malicious else "Benign"
                else:
                    # Fallback: Use CNN model if no valid URL found
                    result = "Using fallback CNN classifier"
                    image = preprocess_image(filepath)
                    pred = model.predict(image, verbose=0)[0][0]
                    is_malicious = pred > 0.5
                    confidence = pred * 100 if is_malicious else (1 - pred) * 100
                    result = "Malicious" if is_malicious else "Benign"
                    qr_data = cleaned_data if cleaned_data else "No URL decoded"

            except Exception as e:
                return jsonify({'error': str(e)})
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

    return render_template_string(
        QR_HTML,
        image_url=image_url,
        result=result,
        is_malicious=is_malicious,
        qr_data=qr_data
    )
