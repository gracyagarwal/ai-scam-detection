import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import re
import os

def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    return ""

def load_data():
    # List of encodings to try
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            # First try with .csv extension
            if os.path.exists('spam.csv'):
                df = pd.read_csv('spam.csv', encoding=encoding)
            # Then try without extension
            elif os.path.exists('spam'):
                df = pd.read_csv('spam', encoding=encoding)
            else:
                print("Error: Could not find the spam dataset file in the current directory.")
                print("Please make sure either 'spam' or 'spam.csv' is in the same folder as this script.")
                print(f"Current working directory: {os.getcwd()}")
                exit(1)
                
            print(f"Successfully loaded spam dataset using {encoding} encoding")
            print(f"Number of messages loaded: {len(df)}")
            
            # If we get here, the file was read successfully
            break
            
        except UnicodeDecodeError:
            # Try next encoding if this one failed
            continue
        except Exception as e:
            print(f"Error reading the file: {str(e)}")
            print("Please make sure the file is not corrupted and is in CSV format.")
            exit(1)
    else:
        # If we get here, none of the encodings worked
        print("Error: Could not read the file with any of the attempted encodings.")
        print("Please check if the file is properly formatted and not corrupted.")
        exit(1)
    
    # Rename columns for clarity
    df.columns = ['label', 'message'] + list(df.columns[2:])
    
    # Convert labels to binary (spam = 1, ham = 0)
    df['label'] = df['label'].map({'spam': 'spam', 'ham': 'genuine'})
    
    return df

def train_model():
    # Load dataset
    df = load_data()
    
    # Preprocess messages
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    
    # Transform text to TF-IDF features
    X = vectorizer.fit_transform(df['processed_message'])
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print("-" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer

def classify_message(message, model, vectorizer):
    # Preprocess the message
    processed_message = preprocess_text(message)
    
    # Transform the message
    message_vector = vectorizer.transform([processed_message])
    
    # Predict the class
    prediction = model.predict(message_vector)[0]
    
    return prediction

def main():
    print("Training SMS spam detection model...")
    model, vectorizer = train_model()
    
    # Test with some example messages from the dataset
    print("\nTesting some sample messages:")
    print("-" * 50)
    
    test_messages = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
        "Even my brother is not like to speak with me. They treat me like aids patent."
    ]
    
    for message in test_messages:
        prediction = classify_message(message, model, vectorizer)
        print(f"\nMessage: {message}")
        print(f"Classification: {'SPAM' if prediction == 'spam' else 'GENUINE'}")
        print("-" * 50)

    print("\nThe model is now ready to classify new messages!")
    
if __name__ == "__main__":
    main() 