import json
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# ==== CONFIGURATION ====
# Change this if your JSON file is elsewhere
JSON_PATH = "C:/Users/Gracy/OneDrive/Desktop/vit/24-25winter/project-2/scam calls/merged_dataset.json"
MODEL_DIR = "new_scam_call_model"
BATCH_SIZE = 16  # Adjust based on system RAM

# ==== LOAD DATASET ====
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Map labels
label_map = {"normal": 0, "fraud": 1}
df["label"] = df["label"].map(label_map)

texts = df["transcript"].tolist()
labels = df["label"].tolist()

# ==== LOAD MODEL ====
print(" Loading model from:", MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

# ==== INFERENCE IN BATCHES ====
print(" Running inference...")
all_preds = []

model.eval()
with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).numpy()
        all_preds.extend(preds)

# ==== EVALUATION ====
print("\n Classification Report:")
print(classification_report(labels, all_preds, target_names=["Normal", "Fraud"]))

cm = confusion_matrix(labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
