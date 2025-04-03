import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        return ' '.join(
            lemmatizer.lemmatize(word)
            for word in words if word not in stop_words
        )
    return ""

def train_model():
    df = pd.read_csv("spam_ham_india.csv", encoding="latin1")
    df = df.dropna(subset=['Msg'])
    df.columns = ['message', 'label']
    df['label'] = df['label'].str.strip().str.lower()
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

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    return model, vectorizer

def classify_message(text, model, vectorizer):
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    return pred
