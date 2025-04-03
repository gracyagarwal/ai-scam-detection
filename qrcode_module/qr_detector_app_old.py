from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
import tensorflow as tf
import os
import glob
import atexit
import signal
from urllib.parse import urlparse
import requests
import whois
import ssl
import socket
import re
import certifi
import hashlib
import validators
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from werkzeug.utils import secure_filename
from config import VIRUSTOTAL_API_KEY

# Set up app and paths
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
QR_DATASET_PATH = r'C:\Users\Gracy\OneDrive\Desktop\vit\24-25winter\project-2\qrcode_module\archive\QR codes cleaned'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Config
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'qr_malware_model.keras'
VIRUSTOTAL_API_KEY = VIRUSTOTAL_API_KEY  

# Cleanup on exit
def cleanup_and_goodbye():
    print("🔒 Cleaning up uploads folder...")
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

atexit.register(cleanup_and_goodbye)
signal.signal(signal.SIGINT, lambda s, f: (cleanup_and_goodbye(), exit(0)))

# MODEL
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def create_model():
    base = EfficientNetB0(include_top=False, weights=None, input_shape=(64, 64, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

class QRDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, files, labels, batch_size=32, shuffle=True):
        self.files = files
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.files))
        if self.shuffle: np.random.shuffle(self.indexes)

    def __len__(self): return len(self.files) // self.batch_size

    def __getitem__(self, idx):
        ixs = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        paths = [self.files[i] for i in ixs]
        X = np.zeros((self.batch_size, 64, 64, 3))
        y = np.array([self.labels[i] for i in ixs])
        for i, path in enumerate(paths):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                X[i] = img / 255.0
        return X, y

def load_and_preprocess_dataset():
    benign = glob.glob(os.path.join(QR_DATASET_PATH, 'benign', '*'))
    malicious = glob.glob(os.path.join(QR_DATASET_PATH, 'malicious', '*'))
    benign = np.random.choice(benign, 10000, replace=False)
    malicious = np.random.choice(malicious, 10000, replace=False)
    all_files = list(benign) + list(malicious)
    all_labels = [0]*len(benign) + [1]*len(malicious)
    zipped = list(zip(all_files, all_labels))
    np.random.shuffle(zipped)
    all_files, all_labels = zip(*zipped)
    usable = (len(all_files) // 32) * 32
    split = int(0.8 * usable)
    return (
        QRDataGenerator(all_files[:split], all_labels[:split]),
        QRDataGenerator(all_files[split:usable], all_labels[split:usable], shuffle=False)
    )

# Image Preprocessing
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64))
    return np.expand_dims(img / 255.0, axis=0)

def extract_qr_data(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    detector = cv2.QRCodeDetector()
    retval, data, _, _ = detector.detectAndDecodeMulti(thresh)
    return data[0] if retval and data else None

# URL CHECKS
def check_url_reputation(url):
    url_id = hashlib.sha256(url.encode()).hexdigest()
    headers = {"accept": "application/json", "x-apikey": VIRUSTOTAL_API_KEY}
    try:
        r = requests.get(f"https://www.virustotal.com/api/v3/urls/{url_id}", headers=headers)
        return r.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
    except: return None

def check_ssl_cert(domain):
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with socket.create_connection((domain, 443), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                return True
    except: return False

def check_domain_age(domain):
    try:
        w = whois.whois(domain)
        creation = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        return (datetime.now() - creation).days > 30 if creation else False
    except: return False

def analyze_url_pattern(url):
    patterns = [r'bit\.ly', r'goo\.gl', r'tinyurl', r'login|signin|account|password',
                r'verify|confirm|update|secure', r'\d{12,}', r'[0-9a-zA-Z]{32,}']
    return any(re.search(p, url.lower()) for p in patterns)

# Load or Train model
print("🔄 Initializing...")
if os.path.exists(MODEL_PATH):
    print("📂 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("🧠 Training new model...")
    model = create_model()
    train_gen, val_gen = load_and_preprocess_dataset()
    model.fit(train_gen, validation_data=val_gen, epochs=15)
    model.save(MODEL_PATH)
    print("✅ Model trained and saved.")

# Web Interface
@app.route('/')
def home():
    return render_template_string("""
        <h2>QR Code Malware Detector</h2>
        <form method="post" enctype="multipart/form-data" action="/analyze">
            <input type="file" name="file" required><br><br>
            <input type="submit" value="Analyze QR Code">
        </form>
    """)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(path)

    try:
        qr_data = extract_qr_data(path)
        pred = model.predict(preprocess_image(path), verbose=0)
        ml_score = float(pred[0][0])
        security_score = ml_score
        checks = []

        if qr_data and validators.url(qr_data):
            domain = urlparse(qr_data).netloc
            with ThreadPoolExecutor(max_workers=4) as ex:
                ssl = ex.submit(check_ssl_cert, domain).result()
                age = ex.submit(check_domain_age, domain).result()
                rep = ex.submit(check_url_reputation, qr_data).result()
                suspicious = ex.submit(analyze_url_pattern, qr_data).result()

            if ssl: security_score *= 0.8; checks.append("✅ SSL valid")
            if age: security_score *= 0.9; checks.append("✅ Domain established")
            if rep and rep.get('malicious', 0) == 0: security_score *= 0.7; checks.append("✅ Good URL reputation")
            if suspicious: security_score *= 1.2; checks.append("⚠️ Suspicious pattern detected")

        return jsonify({
            "ml_score": round(ml_score, 4),
            "final_score": round(security_score, 4),
            "is_malicious": security_score > 0.75,
            "qr_data": qr_data,
            "checks": checks
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        os.remove(path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)