import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from urllib.parse import urlparse
import re
import ssl
import socket
import whois
from datetime import datetime
import os
import requests
import urllib3
from dns import resolver
import dns.exception

# Globals to hold the lazy-loaded model, scaler, and feature columns order
model = None
scaler = None
feature_columns = None

def lazy_load_model():
    """
    Lazy-loads and trains the phishing detection model if not already loaded.
    Assumes phishing.csv exists with an ID column, feature columns, and target as last column.
    Target values should be -1 for phishing (converted to 0) and 1 for legitimate.
    """
    global model, scaler, feature_columns
    if model is not None and scaler is not None:
        return  # Already loaded

    print("🔁 Loading and training phishing model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'phishing.csv')
    df = pd.read_csv(csv_path)
    
    # Assume first column is ID, last column is target; features are in between.
    X = df.iloc[:, 1:-1]
    y = (df.iloc[:, -1] + 1) // 2  # Convert -1 to 0
    feature_columns = X.columns  # Save the order for later

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)
    print("✅ Model training complete.")

def extract_url_features(url):
    """
    Extracts features from the URL using various heuristics and WHOIS lookup.
    Returns a DataFrame with the same column order as the training set.
    """
    features = {}
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower() if parsed.netloc else parsed.path.lower()
        path = parsed.path

        # Protocol check: if missing, try to determine via SSL connection
        if not url.startswith(('http://', 'https://')):
            try:
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443), timeout=3) as sock:
                    with context.wrap_socket(sock, server_hostname=domain):
                        features['HTTPS'] = 1
                        features['HTTPSDomainURL'] = 1
            except:
                features['HTTPS'] = -1
                features['HTTPSDomainURL'] = -1
        else:
            features['HTTPS'] = 1 if url.startswith('https') else -1
            features['HTTPSDomainURL'] = features['HTTPS']

        features['UsingIP'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else -1
        features['LongURL'] = 1 if len(url) > 75 else -1
        features['ShortURL'] = 1 if len(url) < 54 else -1
        features['Symbol@'] = 1 if '@' in url else -1
        features['Redirecting//'] = 1 if '//' in path else -1
        features['PrefixSuffix-'] = -1 if '-' in domain else 1
        features['SubDomains'] = -1 if domain.count('.') > 2 else 1

        try:
            w = whois.whois(domain)
            if w.creation_date:
                creation = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
                domain_age = (datetime.now() - creation).days
                features['DomainRegLen'] = 1 if domain_age > 365 else -1
                features['AgeofDomain'] = 1 if domain_age > 180 else -1
            else:
                features['DomainRegLen'] = -1
                features['AgeofDomain'] = -1
        except:
            features['DomainRegLen'] = -1
            features['AgeofDomain'] = -1

        # For any additional features expected in training data, set default values.
        for feat in feature_columns:
            if feat not in features:
                suspicious = (bool(re.search(r'\d', domain.replace('www.', ''))) or 
                              (domain.count('-') > 0) or (domain.count('.') > 2) or len(domain) > 30)
                features[feat] = -1 if suspicious else 1

        return pd.DataFrame([features])[feature_columns]
    except Exception as e:
        print(f"[!] Feature extraction error: {e}")
        return None

def check_url_accessibility(url):
    """
    Checks if the URL is accessible by resolving its domain and sending an HTTP HEAD request.
    Returns (True, formatted_url) if accessible, else (False, formatted_url).
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc if parsed.netloc else parsed.path
        domain = domain.split(':')[0]

        try:
            resolver.resolve(domain, 'A')
            domain_exists = True
        except (dns.exception.DNSException, dns.resolver.NXDOMAIN):
            domain_exists = False
            return False, url

        if domain_exists:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            if not url.startswith(('http://', 'https://')):
                for protocol in ['https://', 'http://']:
                    try:
                        response = requests.get(f"{protocol}{domain}", timeout=10, verify=False)
                        if response.status_code in [200, 301, 302, 307, 308]:
                            return True, f"{protocol}{domain}"
                    except:
                        continue
                return False, f"https://{domain}"
            try:
                response = requests.get(url, timeout=10, verify=False)
                return response.status_code in [200, 301, 302, 307, 308], url
            except:
                return False, url
        return False, url
    except Exception as e:
        return False, url

def check_url_reputation(url):
    """
    Calls the VirusTotal API to retrieve URL reputation stats.
    Returns a dictionary of stats (e.g., malicious count) or None on error.
    """
    try:
        url_id = hashlib.sha256(url.encode()).hexdigest()
        headers = {"accept": "application/json", "x-apikey": VIRUSTOTAL_API_KEY}
        r = requests.get(f"https://www.virustotal.com/api/v3/urls/{url_id}", headers=headers)
        return r.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
    except Exception as e:
        print(f"[VirusTotal API error] {e}")
        return None

def check_url(url):
    """
    Main function to evaluate the URL.
    - Loads the model if not already loaded.
    - Checks URL accessibility.
    - Extracts features and scales them.
    - Calls the VirusTotal API to get reputation.
    - Incorporates all signals to produce a result string.
    Introduces a new category 'Suspected Phish' when confidence is moderate.
    """
    try:
        lazy_load_model()  # Ensure model, scaler, and feature_columns are loaded

        is_accessible, formatted_url = check_url_accessibility(url)
        features = extract_url_features(formatted_url)
        if features is None:
            return "Error: Could not extract features from URL"

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]

        # Start with the XGBoost prediction
        if prediction == 1:
            result = "Legitimate"
            confidence = proba[1]
        else:
            result = "Phishing"
            confidence = proba[0]

        # Incorporate VirusTotal reputation
        vt_rep = check_url_reputation(formatted_url)
        vt_note = ""
        if vt_rep is not None:
            if vt_rep.get('malicious', 0) > 0:
                result = "Phishing"
                confidence = max(confidence, 0.95)
                vt_note = "\nNote: VirusTotal flagged URL as malicious."
            else:
                vt_note = "\nNote: VirusTotal clean."

        # Incorporate accessibility info
        if not is_accessible:
            try:
                parsed = urlparse(formatted_url)
                domain = parsed.netloc if parsed.netloc else parsed.path
                resolver.resolve(domain, 'A')
                access_note = "\nNote: Domain exists but site is not accessible."
            except:
                result = "Phishing"
                confidence = 0.99
                access_note = "\nNote: Domain does not exist."
        else:
            access_note = ""

        # Introduce 'Suspected Phish' category based on confidence thresholds:
        # If predicted as Phishing but confidence is moderate (<90%), mark as Suspected Phish.
        # Also, if predicted as Legitimate but confidence is low (<60%), mark as Suspected Phish.
        if result == "Phishing" and confidence < 0.9:
            result = "Suspected Phish"
        elif result == "Legitimate" and confidence < 0.6:
            result = "Suspected Phish"

        return (
            f"URL: {formatted_url}\n"
            f"Result: {result}\n"
            f"Confidence: {confidence * 100:.1f}%"
            f"{vt_note}{access_note}"
        ).strip()
    except Exception as e:
        return f"Error analyzing URL: {e}"

def main():
    print("\nPhishing URL Detector")
    print("=" * 50)
    print("Enter 'quit' to exit")
    print("-" * 50)
    
    while True:
        url = input("\nEnter URL to check: ").strip()
        if url.lower() == 'quit':
            print("\nGoodbye!")
            break
        if not url:
            print("Please enter a valid URL")
            continue
        
        # If no protocol provided, let our functions handle it.
        result = check_url(url)
        print("\nAnalysis Result:")
        print("-" * 50)
        print(result)
        print("-" * 50)

if __name__ == "__main__":
    main()
