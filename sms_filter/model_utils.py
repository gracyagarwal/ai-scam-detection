import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model():
    # Get the absolute path to the spam.csv file
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "spam.csv")

    # Load dataset
    df = pd.read_csv(csv_path, encoding="latin1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"spam": 1, "ham": 0})

    # Train/test split
    X_train, _, y_train, _ = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    # Vectorize
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model, vectorizer

def classify_message(text, model, vectorizer):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return "spam" if prediction == 1 else "genuine"
