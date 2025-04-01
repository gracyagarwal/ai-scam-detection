from flask import Blueprint, request, jsonify, render_template_string, current_app
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
import re
from werkzeug.utils import secure_filename

qr_bp = Blueprint('qr', __name__, url_prefix='/qr')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'qr_malware_model.keras')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)

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
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(gray)
    if retval and decoded_info:
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

        form {
            margin-bottom: 30px;
        }

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

        button:hover {
            background-color: #0b5ed7;
        }

        .preview-result-wrapper {
            display: flex;
            justify-content: center;
            gap: 40px;
            align-items: flex-start;
            margin-top: 30px;
            flex-wrap: wrap;
        }

        .preview {
            max-width: 250px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(16, 163, 127, 0.3);
        }

        .result {
            padding: 25px;
            border-radius: 10px;
            font-size: 18px;
            min-width: 300px;
            text-align: left;
        }

        .benign {
            background-color: #1e4423;
            color: #c7f5d3;
        }

        .malicious {
            background-color: #5a1b1b;
            color: #ffd6d6;
        }

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
            <p><strong>Confidence:</strong> {{ confidence }}%</p>
            <p><strong>QR Content:</strong> {{ qr_data }}</p>
        </div>
    </div>
    {% else %}
    <img id="preview" class="preview" src="#" alt="QR Preview" style="display: none;">
    {% endif %}

    <div style="margin-top: 30px; text-align: center;">
        <a href="{{ url_for('home') }}">
            <button style="padding: 8px 16px; background-color: #444; color: white; border: none; border-radius: 5px;">
                ⬅ Back to Home
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
                qr_data = extract_qr_data(filepath)
                if isinstance(qr_data, (list, np.ndarray, pd.Series)):
                    qr_data = str(qr_data[0] if len(qr_data) > 0 else '')
                else:
                    qr_data = str(qr_data).strip()

                url_in_qr = extract_url(qr_data)
                display_qr_data = url_in_qr if url_in_qr else qr_data.split('\n')[0].strip()

                if url_in_qr and is_url_safe(url_in_qr):
                    result = 'Benign'
                    is_malicious = False
                    confidence = 100
                else:
                    image = preprocess_image(filepath)
                    pred = model.predict(image)[0][0]
                    threshold = 0.5
                    is_malicious = pred > threshold
                    confidence = pred * 100 if is_malicious else (1 - pred) * 100
                    qr_data = display_qr_data
                    result = "Malicious" if is_malicious else "Benign"

            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

    return render_template_string(
        QR_HTML,
        image_url=image_url,
        result=result,
        is_malicious=is_malicious,
        confidence=round(confidence, 2),
        qr_data=qr_data,
    )
