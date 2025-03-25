from flask import Flask, request, jsonify, render_template_string
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'qr_malware_model.keras'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
    """Extracts QR code content using OpenCV's QRCodeDetector."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(gray)
    if retval and decoded_info:
        for item in decoded_info:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None

def extract_url(text):
    """Extracts the first valid URL from a string."""
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

HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>QR Code Malware Detection</title>
    <style>
        body { font-family: Arial; background: #f5f5f5; padding: 20px; }
        .container { max-width: 600px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; }
        h1 { text-align: center; }
        .upload-box { border: 2px dashed #ccc; padding: 20px; text-align: center; border-radius: 8px; cursor: pointer; }
        img { max-width: 300px; display: block; margin: 20px auto; }
        .result { padding: 15px; margin-top: 20px; border-radius: 6px; }
        .malicious { background: #ffebee; border: 1px solid #ffcdd2; }
        .benign { background: #e8f5e9; border: 1px solid #c8e6c9; }
    </style>
</head>
<body>
    <div class="container">
        <h1>QR Code Malware Detection</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-box" onclick="document.getElementById('file').click()">
                <p>Click or drag a QR code image here</p>
                <input type="file" id="file" name="file" accept="image/*" style="display:none" required>
            </div>
            <img id="preview" style="display:none;">
            <br>
            <button type="submit">Analyze QR Code</button>
        </form>
        <div id="result" class="result" style="display:none;"></div>
    </div>

    <script>
        const fileInput = document.getElementById('file');
        const preview = document.getElementById('preview');
        const resultDiv = document.getElementById('result');

        fileInput.addEventListener('change', () => {
            const reader = new FileReader();
            reader.onload = e => {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(fileInput.files[0]);
        });

        document.getElementById('uploadForm').addEventListener('submit', async e => {
            e.preventDefault();
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            resultDiv.innerHTML = 'Analyzing...';
            resultDiv.style.display = 'block';
            resultDiv.className = 'result';

            const res = await fetch('/analyze', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.error) {
                resultDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
                return;
            }

            resultDiv.classList.add(data.is_malicious ? 'malicious' : 'benign');
            resultDiv.innerHTML = `
                <strong>Status:</strong> ${data.is_malicious ? '⚠ Malicious' : '✅ Benign'}<br>
                <strong>Confidence:</strong> ${data.confidence_level}<br>
                <strong>QR Content:</strong> <a href="${data.qr_data}" target="_blank">${data.qr_data}</a><br>
                <strong>Reason:</strong> ${data.reason || 'N/A'}
            `;
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HOME_PAGE)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        qr_data = extract_qr_data(file_path)

        # Handle pandas Series, list, or other formats
        if isinstance(qr_data, (list, np.ndarray, pd.Series)):
            qr_data = str(qr_data[0] if len(qr_data) > 0 else '')
        else:
            qr_data = str(qr_data).strip()

        # Clean the QR content: extract only the URL if present,
        # otherwise take the first line of the content.
        url_in_qr = extract_url(qr_data)
        display_qr_data = url_in_qr if url_in_qr else qr_data.split('\n')[0].strip()

        if not display_qr_data:
            return jsonify({'error': 'No QR code content found'}), 400

        # Check live URL safety if a URL was found
        if url_in_qr and is_url_safe(url_in_qr):
            print(f"[SAFE URL OVERRIDE] {url_in_qr}")
            return jsonify({
                'is_malicious': False,
                'confidence_level': "100% (Safe URL Check)",
                'malicious_probability': 0.0,
                'qr_data': display_qr_data,
                'reason': 'URL appears safe on live check'
            })

        # If no safe URL override, run model prediction
        image = preprocess_image(file_path)
        pred = model.predict(image)[0][0]
        threshold = 0.5
        is_malicious = pred > threshold
        confidence = pred * 100 if is_malicious else (1 - pred) * 100

        return jsonify({
            'is_malicious': bool(is_malicious),
            'malicious_probability': float(pred),
            'confidence_level': f"{confidence:.2f}%",
            'qr_data': display_qr_data,
            'threshold_used': threshold,
            'reason': 'Model used'
        })

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    print("✅ Model loaded. Visit http://127.0.0.1:5000 to use the web app.")
    app.run(debug=True)
