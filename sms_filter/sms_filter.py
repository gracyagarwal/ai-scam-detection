import pandas as pd
import re
import nltk
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize English NLP tools
english_stopwords = set(stopwords.words('english'))
english_lemmatizer = WordNetLemmatizer()

# For Hindi tokenization we use the Indic NLP Library
from indicnlp.tokenize import indic_tokenize

# Optionally, you can define a refined Hindi stopword list
hindi_stopwords = set([
    'के', 'का', 'एक', 'में', 'है', 'यह', 'और', 'से', 'हैं', 'को', 'पर', 'इस', 'होता', 'कि', 'जो', 'कर', 'मे',
    'गया', 'करने', 'किया', 'लिया', 'गये', 'अपने', 'हो', 'उन्हें', 'भी', 'पे', 'जैसा', 'तक', 'उनकी', 'ही',
    'अभी', 'इसके', 'साथ', 'अपना', 'आप', 'पूरी', 'उसके', 'बिलकुल', 'भीतर', 'उनका', 'था', 'सकते', 'इसमें',
    'दो', 'होने', 'वह', 'वे', 'करते', 'बहुत', 'कुछ', 'वो', 'करना', 'वर्ग', 'कई', 'करें', 'होती', 'अपनी',
    'उनको', 'जा', 'कहा', 'हुआ', 'जब', 'होते', 'कोई', 'हुई', 'वहाँ', 'जहाँ', 'मेरे', 'कुछ', 'सभी', 'करता',
    'उनकी', 'तरह', 'उस', 'आदि'
])

# ------------------------ #
# LANGUAGE DETECTION
# ------------------------ #
def detect_language(text):
    """Detects language of the input text using langdetect."""
    try:
        lang = detect(text)
        if lang == 'hi':
            return 'hindi'
    except Exception:
        pass
    return 'english'

# ------------------------ #
# PREPROCESSING FUNCTIONS
# ------------------------ #
def preprocess_english(text):
    """
    Preprocesses English text:
      - Lowercases text,
      - Removes URLs,
      - Removes punctuation and numbers,
      - Filters out stopwords and applies lemmatization.
    """
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        return ' '.join(english_lemmatizer.lemmatize(word) for word in words if word not in english_stopwords)
    return ""

def preprocess_hindi(text):
    """
    Preprocesses Hindi text using Indic NLP's tokenization:
      - Removes URLs,
      - Uses the Indic NLP tokenizer to properly segment Hindi text.
      - Optionally, you can remove stopwords from the tokenized output.
    """
    if isinstance(text, str):
        text = text.strip()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Tokenize using Indic NLP's word_tokenize
        # This function tokenizes Devanagari properly.
        tokens = list(indic_tokenize.trivial_tokenize(text, lang='hi'))
        # Optionally remove stopwords:
        # tokens = [t for t in tokens if t not in hindi_stopwords]
        processed = " ".join(tokens)
        if processed == "":
            return "none"
        return processed
    return ""

# ------------------------ #
# BUILD ENGLISH PIPELINE
# ------------------------ #
def build_english_pipeline():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_english,
                                  ngram_range=(1, 3),
                                  max_df=0.9,
                                  min_df=3,
                                  stop_words='english',
                                  sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    return pipeline

# ------------------------ #
# BUILD HINDI PIPELINE
# ------------------------ #
def build_hindi_pipeline():
    """
    Builds a Hindi pipeline that combines word-level and character-level TF-IDF features.
    Uses the updated Hindi preprocessing function that leverages Indic NLP tokenization.
    """
    word_vectorizer = TfidfVectorizer(preprocessor=preprocess_hindi,
                                      analyzer='word',
                                      ngram_range=(1, 2),
                                      max_df=1.0,
                                      min_df=1,
                                      stop_words=None,  # You can experiment with hindi_stopwords here.
                                      sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(preprocessor=preprocess_hindi,
                                      analyzer='char',
                                      ngram_range=(2, 5),
                                      max_df=1.0,
                                      min_df=1,
                                      sublinear_tf=True)
    combined_features = FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer)
    ])
    pipeline = Pipeline([
        ('features', combined_features),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    return pipeline

# ------------------------ #
# DATA LOADING
# ------------------------ #
def load_combined_data(file_path="combined_dataset.csv"):
    """
    Loads the combined dataset.
    Expected columns: 'Msg', 'label', 'language'
    """
    df = pd.read_csv(file_path, encoding='latin1')
    df = df.dropna(subset=['Msg'])
    df.columns = ['Msg', 'label', 'language']
    df['label'] = df['label'].str.strip().str.lower()
    df['language'] = df['language'].str.strip().str.lower()
    return df

# ------------------------ #
# GRID SEARCH FOR HINDI PIPELINE
# ------------------------ #
def grid_search_hindi_pipeline(hindi_data):
    """
    Performs grid search on the Hindi pipeline using the Hindi subset of data.
    Prints the best parameters and best score.
    """
    hin_pipeline = build_hindi_pipeline()
    param_grid = {
        "clf__C": [0.1, 1, 10],
        "features__word__ngram_range": [(1,2), (1,3)],
        "features__word__min_df": [1, 2],
        "features__char__ngram_range": [(2,5), (3,5)],
        "features__char__min_df": [1, 2]
    }
    X = hindi_data['Msg']
    y = hindi_data['label']
    grid = GridSearchCV(hin_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid.fit(X, y)
    print("Best parameters found:", grid.best_params_)
    print("Best score:", grid.best_score_)
    return grid.best_estimator_

# ------------------------ #
# TRAINING ALL MODELS
# ------------------------ #
def train_models():
    """
    Trains separate pipelines for English and Hindi messages.
    For Hindi, performs grid search for parameter tuning.
    Returns the trained English pipeline and Hindi pipeline.
    """
    df = load_combined_data()
    english_data = df[df['language'] == 'english']
    hindi_data = df[df['language'] == 'hindi']
    
    eng_pipeline = build_english_pipeline()
    if not english_data.empty:
        X_eng = english_data['Msg']
        y_eng = english_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X_eng, y_eng, test_size=0.2, stratify=y_eng, random_state=42)
        eng_pipeline.fit(X_train, y_train)
        y_pred_eng = eng_pipeline.predict(X_test)
        print("\nEnglish Model Evaluation:\n" + "="*60)
        print(f"Accuracy: {accuracy_score(y_test, y_pred_eng):.4f}")
        print(classification_report(y_test, y_pred_eng))
    else:
        print("No English data available.")
    
    if not hindi_data.empty:
        print("\nPerforming grid search for Hindi pipeline...")
        hin_pipeline = grid_search_hindi_pipeline(hindi_data)
        X_hin = hindi_data['Msg']
        y_hin = hindi_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X_hin, y_hin, test_size=0.2, stratify=y_hin, random_state=42)
        hin_pipeline.fit(X_train, y_train)
        y_pred_hin = hin_pipeline.predict(X_test)
        print("\nHindi Model Evaluation (after grid search):\n" + "="*60)
        print(f"Accuracy: {accuracy_score(y_test, y_pred_hin):.4f}")
        print(classification_report(y_test, y_pred_hin))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_hin))
    else:
        print("No Hindi data available.")
    
    return eng_pipeline, hin_pipeline

# ------------------------ #
# MULTILINGUAL CLASSIFICATION
# ------------------------ #
def classify_message_multilingual(message, eng_pipeline, hin_pipeline):
    """Automatically detects language and uses the corresponding pipeline."""
    lang = detect_language(message)
    if lang == 'hindi':
        pred = hin_pipeline.predict([message])[0]
    else:
        pred = eng_pipeline.predict([message])[0]
    return pred

# ------------------------ #
# MAIN FUNCTION
# ------------------------ #
def main():
    print("Training separate models for English and Hindi...")
    eng_pipeline, hin_pipeline = train_models()
    print("Model training complete.")
    
    # Sample test messages
    sample_tests = [
        {"message": "Get FREE recharge now at http://freerecharge.com", "expected_language": "english"},
        {"message": "Hey, are we still on for dinner tonight?", "expected_language": "english"},
        {"message": "आपका अकाउंट ब्लॉक हो रहा है, तुरंत वेरिफाई करें।", "expected_language": "hindi"},
        {"message": "नया मोबाइल ऑफर अभी खरीदें!", "expected_language": "hindi"}
    ]
    
    for test in sample_tests:
        result = classify_message_multilingual(test["message"], eng_pipeline, hin_pipeline)
        detected_lang = detect_language(test["message"])
        print(f"\nMessage: {test['message']} (Detected: {detected_lang})")
        print(f"Predicted Label: {'SPAM' if result=='spam' else 'HAM'}")

    
if __name__ == "__main__":
    main()
