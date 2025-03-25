import os
import tensorflow as tf
import cv2
import numpy as np

# Set model path (update if necessary)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'qr_malware_model.keras')

# Load the saved model (only if it exists)
if os.path.exists(MODEL_PATH):
    print("\n📂 Loading existing model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"\n❌ No model found at {MODEL_PATH}. Please train and save the model first.")

def preprocess_image(image_path):
    """Preprocess the QR code image for prediction."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    return np.expand_dims(image, axis=[0, -1])

def extract_qr_data(image_path):
    """Extract data encoded inside a QR code."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(gray)
    if retval and decoded_info:
        return decoded_info[0]
    return None
