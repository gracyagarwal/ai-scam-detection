import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, accuracy_score
from lime.lime_text import LimeTextExplainer  # For local explanations

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs from text during TF-IDF
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation and numbers
        words = text.split()
        return ' '.join(lemmatizer.lemmatize(word) for word in words if word not in stop_words)
    return ""


def load_data(file_path="sms_filter/spam_ham_india.csv"):
    df = pd.read_csv(file_path, encoding='latin1')
    df = df.dropna(subset=['Msg'])
    df.columns = ['message', 'label']
    df['label'] = df['label'].str.strip().str.lower()
    # Convert labels: 'spam' becomes 'scam'; everything else becomes 'genuine'
    df['label'] = df['label'].apply(lambda x: 'scam' if x == 'spam' else 'genuine')
    return df


from sklearn.base import BaseEstimator, TransformerMixin

class AdditionalFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer that extracts additional heuristic features from text:
      - digit_count: total number of digit characters.
      - suspicious_keyword_count: count of selected suspicious keywords.
    """
    def __init__(self):
        self.suspicious_keywords = ['win', 'free', 'congrat', 'dial', 'offer', 'urgent']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            # Count digit characters (could indicate phone numbers or codes)
            digit_count = sum(c.isdigit() for c in text)
            # Count occurrences of suspicious keywords
            text_lower = text.lower()
            suspicious_count = sum(text_lower.count(keyword) for keyword in self.suspicious_keywords)
            features.append([digit_count, suspicious_count])
        return np.array(features)


def train_model():
    df = load_data()
    # X will be the raw messages; preprocessing is handled in the pipeline.
    X = df['message']
    y = df['label']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Build a pipeline that combines TF-IDF features with additional heuristic features.
    # The additional features are scaled so they harmonize with the TF-IDF values.
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(
                preprocessor=preprocess_text,
                ngram_range=(1, 3),
                max_df=0.9,
                min_df=3,
                stop_words='english',
                sublinear_tf=True
            )),
            ('additional', Pipeline([
                ('extract', AdditionalFeaturesExtractor()),
                ('scaler', StandardScaler())
            ]))
        ])),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    # Hyperparameter tuning: grid search for logistic regression parameter C
    from sklearn.model_selection import GridSearchCV
    param_grid = {'clf__C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X_train, y_train)
    print("Best parameter:", grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    
    # Calibrate probabilities for better threshold decisions
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = calibrated_model.predict(X_test)
    print("\nModel Evaluation:\n" + "=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    return calibrated_model


def classify_message(text, model, threshold=0.5):
    """
    Classifies an SMS as scam or genuine using the calibrated model's probabilities.
    Args:
      text (str): The SMS message.
      model: The calibrated model pipeline.
      threshold (float): Probability threshold for 'scam'.
    Returns:
      'scam' or 'genuine'
    """
    probas = model.predict_proba([text])[0]
    try:
        scam_index = list(model.classes_).index('scam')
    except ValueError:
        scam_index = 0  # Fallback if 'scam' is not found
    return 'scam' if probas[scam_index] >= threshold else 'genuine'


def explain_message(text, model, num_features=6):
    """
    Returns readable LIME explanation summary + individual word effects.
    """
    explainer = LimeTextExplainer(class_names=model.classes_)
    predict_proba = lambda texts: model.predict_proba(texts)
    exp = explainer.explain_instance(text, predict_proba, num_features=num_features)

    explanation = exp.as_list()
    sorted_exp = sorted(explanation, key=lambda x: abs(x[1]), reverse=True)

    scam_words = [word for word, weight in sorted_exp if weight > 0]
    genuine_words = [word for word, weight in sorted_exp if weight < 0]

    if scam_words and not genuine_words:
        summary = f"This message is likely a scam due to words like {', '.join(scam_words[:3])}."
    elif genuine_words and not scam_words:
        summary = f"This message appears genuine because of words like {', '.join(genuine_words[:3])}."
    elif scam_words and genuine_words:
        summary = f"The message contains a mix of scam-like words ({', '.join(scam_words[:2])}) and genuine indicators ({', '.join(genuine_words[:2])})."
    else:
        summary = "No strong indicators detected."

    readable_exp = [(word, weight, f"{'Scam' if weight > 0 else 'Genuine'} word: {word}") for word, weight in sorted_exp[:5]]
    return [("Summary", 0.0, summary)] + readable_exp