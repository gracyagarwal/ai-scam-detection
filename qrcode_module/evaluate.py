import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("qr_malware_model.keras")

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return np.expand_dims(img, axis=-1)

# Load image paths
benign_dir = "archive/QR codes cleaned/benign"
malicious_dir = "archive/QR codes cleaned/malicious"

benign_files = glob.glob(os.path.join(benign_dir, "*.png"))
malicious_files = glob.glob(os.path.join(malicious_dir, "*.png"))

# Limit for quick testing (optional)
benign_files = benign_files[:500]
malicious_files = malicious_files[:500]

X = []
y = []

for path in benign_files:
    X.append(preprocess_image(path))
    y.append(0)

for path in malicious_files:
    X.append(preprocess_image(path))
    y.append(1)

X = np.array(X)
y = np.array(y)

# Predict
y_pred_probs = model.predict(X, batch_size=32)
y_pred = (y_pred_probs > 0.5).astype("int32")

# Report
print("\n📊 Classification Report:")
print(classification_report(y, y_pred, target_names=["Benign", "Malicious"]))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xticks([0,1], ['Benign', 'Malicious'])
plt.yticks([0,1], ['Benign', 'Malicious'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center')
plt.colorbar()
plt.show()
