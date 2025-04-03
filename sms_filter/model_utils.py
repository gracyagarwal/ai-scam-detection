import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

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
    df = pd.read_csv(file_path, encoding='latin1')
    df = df.dropna(subset=['Msg'])
    df.columns = ['message', 'label']
    df['label'] = df['label'].str.strip().str.lower()
    return df

# ------------------------ #
# TRAIN MODEL
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
def classify_message(text, model_tuple):
    model, vectorizer = model_tuple
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    return model.predict(vec)[0]
