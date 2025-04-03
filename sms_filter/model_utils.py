import pandas as pd
import re
import nltk
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize English NLP tools
english_stopwords = set(stopwords.words('english'))
english_lemmatizer = WordNetLemmatizer()

# ------------------------ #
# PREPROCESSING FUNCTIONS
# ------------------------ #
def preprocess_english(text):
    """
    Preprocesses English text:
      - Converts text to lowercase,
      - Removes URLs,
      - Removes punctuation and numbers,
      - Tokenizes and lemmatizes while removing English stopwords.
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
    Preprocesses Hindi text:
      - Removes URLs,
      - Removes all characters except Devanagari (Unicode 0900-097F) and whitespace.
      - If the result is empty, returns "none" as a fallback token.
    """
    if isinstance(text, str):
        text = text.strip()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove all characters except Devanagari and whitespace.
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        text = text.strip()
        if text == "":
            return "none"
        return text
    return ""

# ------------------------ #
# BUILD PIPELINES
# ------------------------ #
def build_english_pipeline():
    """
    Builds an English pipeline using TF-IDF features and Logistic Regression.
    """
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

def build_hindi_pipeline():
    """
    Builds a Hindi pipeline that combines word-level and character-level TF-IDF features,
    using our best-found parameters:
      - Word-level: ngram_range (1,2), min_df=2
      - Character-level: ngram_range (2,5), min_df=1
      - Classifier: LogisticRegression with C=10.
    """
    word_vectorizer = TfidfVectorizer(preprocessor=preprocess_hindi,
                                      analyzer='word',
                                      ngram_range=(1, 2),
                                      max_df=1.0,
                                      min_df=2,
                                      stop_words=None,  # No stopword removal here
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
        ('clf', LogisticRegression(C=10, max_iter=1000, class_weight='balanced'))
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
    df = pd.read_csv(file_path, encoding="latin1")
    df = df.dropna(subset=['Msg'])
    df.columns = ['Msg', 'label', 'language']
    df['label'] = df['label'].str.strip().str.lower()
    df['language'] = df['language'].str.strip().str.lower()
    return df

# ------------------------ #
# TRAIN MODEL
# ------------------------ #
def train_model():
    """
    Trains separate pipelines for English and Hindi using the combined dataset.
    Returns a dictionary with keys 'english' and 'hindi' mapping to the respective trained pipelines.
    """
    df = load_combined_data()
    # Split the data by language
    english_data = df[df['language'] == 'english']
    hindi_data = df[df['language'] == 'hindi']
    
    eng_pipeline = build_english_pipeline()
    hin_pipeline = build_hindi_pipeline()
    
    if not english_data.empty:
        X_eng = english_data['Msg']
        y_eng = english_data['label']
        eng_pipeline.fit(X_eng, y_eng)
    if not hindi_data.empty:
        X_hin = hindi_data['Msg']
        y_hin = hindi_data['label']
        hin_pipeline.fit(X_hin, y_hin)
    
    # Return the pipelines as a dictionary
    return {"english": eng_pipeline, "hindi": hin_pipeline}

# ------------------------ #
# CLASSIFY MESSAGE
# ------------------------ #
def classify_message(text, model):
    """
    Classifies an input message using the appropriate pipeline.
    The model parameter is a dictionary with keys 'english' and 'hindi'.
    """
    try:
        lang = detect(text)
        if lang == 'hi':
            return model["hindi"].predict([text])[0]
        else:
            return model["english"].predict([text])[0]
    except Exception as e:
        return "error"