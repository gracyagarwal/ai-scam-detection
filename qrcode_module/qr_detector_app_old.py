from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
import tensorflow as tf
import os
import glob
import atexit
import signal
from werkzeug.utils import secure_filename

# ==== Configuration ====
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Actual dataset path
QR_DATASET_PATH = 'C:\\Users\\Gracy\\OneDrive\\Desktop\\vit\\24-25winter\\project-2\\qrcode_module\\archive\\QR codes'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'qr_malware_model.keras'

# ==== Cleanup on Exit ====
def cleanup_and_goodbye():
    print("\nCleaning up resources...")
    try:
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("Thank you for using QR Code Malware Detection System!")
    except Exception as e:
        print(f"Cleanup error: {e}")

atexit.register(cleanup_and_goodbye)
signal.signal(signal.SIGINT, lambda s, f: (cleanup_and_goodbye(), exit(0)))

# ==== Model Creation ====
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 1)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# ==== Custom Data Generator ====
class QRDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=64, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.file_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = [self.file_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]

        X = np.zeros((self.batch_size, 64, 64, 1))
        y = np.array(batch_labels)

        for i, path in enumerate(batch_paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                img = img / 255.0
                X[i] = np.expand_dims(img, axis=-1)

        return X, y

# ==== Dataset Loader ====
def load_and_preprocess_dataset():
    print("\nLoading dataset...")
    benign_path = os.path.join(QR_DATASET_PATH, 'benign', 'benign')
    malicious_path = os.path.join(QR_DATASET_PATH, 'malicious', 'malicious')

    benign_files = glob.glob(os.path.join(benign_path, '*'))
    malicious_files = glob.glob(os.path.join(malicious_path, '*'))

    benign_files = [f for f in benign_files if f.lower().endswith(tuple(ALLOWED_EXTENSIONS))]
    malicious_files = [f for f in malicious_files if f.lower().endswith(tuple(ALLOWED_EXTENSIONS))]

    print(f"Benign files found: {len(benign_files)}")
    print(f"Malicious files found: {len(malicious_files)}")

    all_files = benign_files + malicious_files
    all_labels = [0] * len(benign_files) + [1] * len(malicious_files)

    if len(all_files) == 0:
        raise Exception("No QR code images found in dataset folders!")

    temp = list(zip(all_files, all_labels))
    np.random.shuffle(temp)
    all_files, all_labels = zip(*temp)

    batch_size = 64
    usable_size = (len(all_files) // batch_size) * batch_size
    all_files = all_files[:usable_size]
    all_labels = all_labels[:usable_size]

    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    train_labels = all_labels[:split_idx]
    val_labels = all_labels[split_idx:]

    print(f"Training: {len(train_files)}, Validation: {len(val_files)}")

    train_gen = QRDataGenerator(train_files, train_labels, batch_size)
    val_gen = QRDataGenerator(val_files, val_labels, batch_size, shuffle=False)

    return train_gen, val_gen, len(train_files), len(val_files)

# ==== Preprocessing ====
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return np.expand_dims(img, axis=(0, -1))

def extract_qr_data(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    detector = cv2.QRCodeDetector()
    retval, decoded_info, _, _ = detector.detectAndDecodeMulti(gray)
    return decoded_info[0] if retval and decoded_info else None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==== Model Init or Training ====
print("\nInitializing QR Code Malware Detection System...")
if os.path.exists(MODEL_PATH):
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Training new model...")
    model = create_model()
    train_gen, val_gen, train_size, val_size = load_and_preprocess_dataset()

    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
        ]
    )
    model = tf.keras.models.load_model('best_model.keras')
    model.save(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

# ==== Flask Web Interface ====
@app.route('/')
def home():
    return render_template_string("<h2>QR Code Scanner Running</h2>")

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        qr_data = extract_qr_data(filepath)
        if qr_data is None:
            return jsonify({'error': 'No QR code detected'}), 400

        img = preprocess_image(filepath)
        pred = model.predict(img)[0][0]
        threshold = 0.75
        is_malicious = pred > threshold
        confidence = round(float(pred if is_malicious else 1 - pred) * 100, 2)

        return jsonify({
            'is_malicious': is_malicious,
            'malicious_probability': float(pred),
            'confidence': confidence,
            'qr_data': qr_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(filepath)

if __name__ == '__main__':
    print("Starting server at http://127.0.0.1:5000")
    app.run(debug=True)
