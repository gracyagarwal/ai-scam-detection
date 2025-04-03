import os
import re
import logging
import argparse

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------------ #
# TEXT PREPROCESSING
# ------------------------ #
def preprocess_text(text):
    """
    Preprocesses the input text by lowercasing, removing URLs, punctuation,
    and stopwords, and applying lemmatization.
    """
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation and numbers
        words = text.split()
        return ' '.join(lemmatizer.lemmatize(word) for word in words if word not in stop_words)
    return ""

# ------------------------ #
# LOAD AND CLEAN DATA
# ------------------------ #
def load_data(file_path="spam_ham_india.csv"):
    """
    Loads and cleans the dataset from a CSV file.
    The file is expected to have columns: 'Msg' and 'Label'.
    """
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} not found.")
        raise FileNotFoundError(f"{file_path} not found.")
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        logging.error("Error reading the CSV file: " + str(e))
        raise e

    df = df.dropna(subset=['Msg'])
    df.columns = ['message', 'label']
    df['label'] = df['label'].str.strip().str.lower()
    return df

# ------------------------ #
# TRAIN THE MODEL
# ------------------------ #
def train_model():
    """
    Trains a Logistic Regression model using a TF-IDF vectorizer.
    Returns the trained model and the vectorizer.
    """
    df = load_data()
    df['processed'] = df['message'].apply(preprocess_text)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_df=0.9,
        min_df=3,
        stop_words='english',
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df['processed'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    logging.info("\nModel Evaluation:\n" + "=" * 60)
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info("\n" + classification_report(y_test, y_pred))

    return model, vectorizer

# ------------------------ #
# CLASSIFY A SINGLE MESSAGE
# ------------------------ #
def classify_message(message, model, vectorizer):
    """
    Classifies a single message using the trained model.
    """
    processed = preprocess_text(message)
    vec = vectorizer.transform([processed])
    return model.predict(vec)[0]

# ------------------------ #
# MAIN FUNCTION
# ------------------------ #
def main():
    """
    Main function to train the model and classify sample messages.
    """
    logging.info("Training SMS spam detection model with enhanced NLP pipeline...")
    model, vectorizer = train_model()
    logging.info("Model training complete.")

    sample_msgs = [
        "Get FREE recharge now at http://freerecharge.com",
        "Hey, are we still on for dinner tonight?",
        "Congratulations! You've won a 5 lakh rupee lottery. Click now.",
        "New episodes of your favorite show are streaming.",
        "You’ve been selected for a luxury trip to Maldives!",
        "Refer a friend and earn ₹500 instantly!",
        "Your KYC is incomplete. Click now to avoid suspension."
    ]

    for msg in sample_msgs:
        result = classify_message(msg, model, vectorizer)
        label = "SPAM" if result == "spam" else "GENUINE"
        print(f"\nMessage: {msg}\n→ Classified as: {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SMS Spam Detection Model")
    parser.add_argument("--file", type=str, default="spam_ham_india.csv", help="Path to the CSV dataset file")
    args = parser.parse_args()
    
    main()
