import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Define the image size as used during training
IMG_SIZE = 64

# Load trained fallback CNN model
model = tf.keras.models.load_model("qr_malware_model.keras")

def preprocess_image(path):
    """Read an image from path, resize, normalize, and add channel dimension."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=-1)

# Define dataset directories (update if needed)
benign_dir = "archive/QR codes cleaned/benign"
malicious_dir = "archive/QR codes cleaned/malicious"

# Get list of image files (you can adjust the glob pattern if needed)
benign_files = glob.glob(os.path.join(benign_dir, "*"))
malicious_files = glob.glob(os.path.join(malicious_dir, "*"))

# Limit sample size for quick evaluation (optional)
benign_files = benign_files[:500]
malicious_files = malicious_files[:500]

X = []
y = []

# Process benign images
for path in benign_files:
    X.append(preprocess_image(path))
    y.append(0)

# Process malicious images
for path in malicious_files:
    X.append(preprocess_image(path))
    y.append(1)

# Convert lists to numpy arrays
X = np.array(X)  # Shape: (num_samples, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# Predict probabilities and classes
y_pred_probs = model.predict(X, batch_size=32)
y_pred = (y_pred_probs > 0.5).astype("int32")

# Print Classification Report
print("\n QR Code Scam Classification Report:")
print(classification_report(y, y_pred, target_names=["Benign", "Malicious"]))

# Compute and Plot Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xticks([0, 1], ['Benign', 'Malicious'])
plt.yticks([0, 1], ['Benign', 'Malicious'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='red', fontsize=16)
plt.colorbar()
plt.show()
