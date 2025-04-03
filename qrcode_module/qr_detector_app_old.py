from flask import Flask, request, jsonify, render_template_string
import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import atexit
import signal
import requests
import whois
import ssl
import socket
import re
import certifi
import hashlib
import validators
from datetime import datetime
from urllib.parse import urlparse
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from config import VIRUSTOTAL_API_KEY
import random
from tensorflow.keras import regularizers

# ==== Config ====
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
QR_DATASET_PATH = r'C:\Users\Gracy\OneDrive\Desktop\vit\24-25winter\project-2\qrcode_module\archive\QR codes cleaned'
MODEL_PATH = 'qr_malware_model.keras'
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 15
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==== Cleanup on exit ====
def cleanup_and_goodbye():
    print("\n🔒 Cleaning up uploads folder...")
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))
atexit.register(cleanup_and_goodbye)
signal.signal(signal.SIGINT, lambda s, f: (cleanup_and_goodbye(), exit(0)))

# ==== Data Augmentation Function ====
def augment_image(img):
    # Random rotation between -15 and 15 degrees
    angle = random.uniform(-15, 15)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Random horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    
    # Random brightness adjustment
    brightness = random.uniform(0.7, 1.3)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    
    # Occasionally apply Gaussian blur
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

# ==== CNN Model with L2 Regularization ====
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# ==== Data Generator with Augmentation Option ====
class QRDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=BATCH_SIZE, shuffle=True, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return len(self.file_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = [self.file_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        X = np.zeros((self.batch_size, IMG_SIZE, IMG_SIZE, 1))
        y = np.array(batch_labels)
        
        for i, path in enumerate(batch_paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            if self.augment:
                img = augment_image(img)
            img = img / 255.0
            X[i] = np.expand_dims(img, axis=-1)
        
        return X, y

# ==== Dataset Loader ====
def load_dataset(augment=False):
    benign_path = os.path.join(QR_DATASET_PATH, 'benign')
    malicious_path = os.path.join(QR_DATASET_PATH, 'malicious')
    
    benign_files = glob.glob(os.path.join(benign_path, '*'))
    malicious_files = glob.glob(os.path.join(malicious_path, '*'))
    
    all_files = benign_files + malicious_files
    all_labels = [0] * len(benign_files) + [1] * len(malicious_files)
    
    temp = list(zip(all_files, all_labels))
    np.random.shuffle(temp)
    all_files, all_labels = zip(*temp)
    
    usable = (len(all_files) // BATCH_SIZE) * BATCH_SIZE
    all_files = all_files[:usable]
    all_labels = all_labels[:usable]
    
    split = int(0.8 * usable)
    train_files, val_files = all_files[:split], all_files[split:]
    train_labels, val_labels = all_labels[:split], all_labels[split:]
    
    return (QRDataGenerator(train_files, train_labels, augment=augment),
            QRDataGenerator(val_files, val_labels, shuffle=False))

# ==== Training ==== 
def train_cnn_if_needed():
    if os.path.exists(MODEL_PATH):
        print("📂 Loading existing CNN fallback model...")
        return tf.keras.models.load_model(MODEL_PATH)
    
    print("🧠 Training fallback CNN model with augmentation...")
    model = create_model()
    train_gen, val_gen = load_dataset(augment=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
    ]
    
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    return model

# ==== Utility Functions ==== 
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=(0, -1))

def extract_qr_data(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    detector = cv2.QRCodeDetector()
    retval, data, _, _ = detector.detectAndDecodeMulti(thresh)
    return data[0] if retval and data else None

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==== Flask Web App ==== 
model = train_cnn_if_needed()

@app.route('/')
def home():
    return render_template_string("""
    <h2>QR Scam Detection Hybrid App</h2>
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
        return jsonify({'error': 'Invalid file type'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    try:
        qr_data = extract_qr_data(filepath)
        checks = []
        final_score = 0.0

        if qr_data and validators.url(qr_data):
            domain = urlparse(qr_data).netloc
            with ThreadPoolExecutor(max_workers=4) as ex:
                ssl_valid = ex.submit(check_ssl_cert, domain).result()
                age_ok = ex.submit(check_domain_age, domain).result()
                vt_rep = ex.submit(check_url_reputation, qr_data).result()
                suspicious = ex.submit(analyze_url_pattern, qr_data).result()

            final_score = 0.5
            if ssl_valid: final_score -= 0.1; checks.append("✅ SSL Valid")
            if age_ok: final_score -= 0.1; checks.append("✅ Domain is old")
            if vt_rep and vt_rep.get('malicious', 0) == 0: final_score -= 0.1; checks.append("✅ VirusTotal Clean")
            if suspicious: final_score += 0.2; checks.append("⚠️ Suspicious pattern detected")
        else:
            checks.append("⚠️ QR code unreadable or no valid URL. Using image classifier.")
            pred = model.predict(preprocess_image(filepath), verbose=0)[0][0]
            final_score = float(pred)
            checks.append(f"🧠 CNN fallback prediction: {round(pred, 3)}")

        return jsonify({
            'qr_data': qr_data,
            'final_score': round(final_score, 4),
            'is_malicious': final_score > 0.5,
            'checks': checks
        })

    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        os.remove(filepath)

if __name__ == '__main__':
    print("🌐 Running hybrid scam detector at http://127.0.0.1:5000")
    app.run(debug=True)
