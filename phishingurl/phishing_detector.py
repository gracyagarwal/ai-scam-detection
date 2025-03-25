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

# Globals to hold the lazy-loaded model and scaler
model = None
scaler = None
X = None

def lazy_load_model():
    global model, scaler, X
    if model is not None and scaler is not None:
        return  # Already loaded

    print("🔁 Loading and training phishing model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'phishing.csv')

    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:-1]
    y = (df.iloc[:, -1] + 1) // 2  # Convert -1 to 0

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

def extract_url_features(url):
    features = {}
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower() if parsed.netloc else parsed.path.lower()
        path = parsed.path

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
                creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
                domain_age = (datetime.now() - creation_date).days
                features['DomainRegLen'] = 1 if domain_age > 365 else -1
                features['AgeofDomain'] = 1 if domain_age > 180 else -1
            else:
                features['DomainRegLen'] = -1
                features['AgeofDomain'] = -1
        except:
            features['DomainRegLen'] = -1
            features['AgeofDomain'] = -1

        suspicious = (
            bool(re.search(r'\d', domain.replace('www.', ''))) or
            (domain.count('-') > 0) or
            (domain.count('.') > 2) or
            len(domain) > 30
        )

        for feature in X.columns:
            if feature not in features:
                features[feature] = -1 if suspicious else 1

        return pd.DataFrame([features])[X.columns]
    except Exception as e:
        print(f"[!] Feature extraction error: {e}")
        return None

def check_url_accessibility(url):
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

def check_url(url):
    try:
        lazy_load_model()  # Load model only when needed

        is_accessible, formatted_url = check_url_accessibility(url)
        features = extract_url_features(formatted_url)

        if features is None:
            return "Error: Could not extract features from URL"

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        result = "Legitimate" if prediction == 1 else "Phishing"
        confidence = proba[1] if prediction == 1 else proba[0]

        if not is_accessible:
            try:
                parsed = urlparse(formatted_url)
                domain = parsed.netloc if parsed.netloc else parsed.path
                resolver.resolve(domain, 'A')
                accessibility_note = "\nNote: Domain exists but site is not accessible"
            except:
                result = "Phishing"
                confidence = 0.99
                accessibility_note = "\nNote: Domain does not exist"
        else:
            accessibility_note = ""

        return (
            f"URL: {formatted_url}\n"
            f"Result: {result}\n"
            f"Confidence: {confidence * 100:.1f}%"
            f"{accessibility_note}"
        ).strip()
    except Exception as e:
        return f"Error analyzing URL: {e}"


# Test with some example URLs
# print("\nTesting URLs:")
# test_urls = [
#     "https://www.google.com",
#     "https://www.facebook.com",
#     "http://suspicious-bank-login.com",
#     "http://verify-account-secure-login.com"
# ]

# for url in test_urls:
#     result = check_url(url)
#     print(f"\n{result}")


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
        
        # Don't automatically add http:// - try HTTPS check first
        if not url.startswith(('http://', 'https://')):
            url = url  # Let the feature extractor handle the protocol check
        
        result = check_url(url)
        print("\nAnalysis Result:")
        print("-" * 50)
        print(result)
        print("-" * 50)

if __name__ == "__main__":
    main()


