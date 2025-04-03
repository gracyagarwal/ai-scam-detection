import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------------ #
# TEXT PREPROCESSING
# ------------------------ #
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation and numbers
        words = text.split()
        text = ' '.join(
            lemmatizer.lemmatize(word)
            for word in words if word not in stop_words
        )
        return text
    return ""

# ------------------------ #
# LOAD AND CLEAN DATA
# ------------------------ #
def load_data():
    df = pd.read_csv("spam_ham_india.csv", encoding='latin1')
    df = df.dropna(subset=['Msg'])  # remove rows with empty messages
    df.columns = ['message', 'label']
    df['label'] = df['label'].str.strip().str.lower()
    return df

# ------------------------ #
# TRAIN THE MODEL
# ------------------------ #
def train_model():
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
    print("\nModel Evaluation:\n" + "=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    return model, vectorizer

# ------------------------ #
# CLASSIFY A SINGLE MESSAGE
# ------------------------ #
def classify_message(message, model, vectorizer):
    processed = preprocess_text(message)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    return pred

# ------------------------ #
# MAIN FUNCTION
# ------------------------ #
def main():
    print("Training SMS spam detection model with enhanced NLP pipeline...")
    model, vectorizer = train_model()
    print("Model training complete.")

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
        print(f"\nMessage: {msg}\n→ Classified as: {'SPAM' if result == 'spam' else 'GENUINE'}")

if __name__ == "__main__":
    main()
