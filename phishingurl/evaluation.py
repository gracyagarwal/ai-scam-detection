import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Path to phishing.csv (same folder)
CSV_PATH = os.path.join(os.path.dirname(__file__), 'phishing.csv')

def load_and_train_model():
    df = pd.read_csv(CSV_PATH)

    # Assuming: first column = ID, last column = target, features in the middle
    X = df.iloc[:, 1:-1]
    y = (df.iloc[:, -1] + 1) // 2  # Convert -1 (phish) to 0, and 1 stays 1

    # Save feature order for consistency
    feature_columns = X.columns

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)

    return model, X_test_scaled, y_test

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='red', fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    print(" Evaluating Phishing URL Classifier...\n")
    model, X_test, y_test = load_and_train_model()

    y_pred = model.predict(X_test)

    print(" Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Phishing", "Legitimate"]))

    cm = confusion_matrix(y_test, y_pred)
    print("\n Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm, labels=["Phishing", "Legitimate"])

if __name__ == "__main__":
    main()
