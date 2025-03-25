# AI Scam Detection

**Capstone Project**  
**Contributors**: Gracy Agarwal, Pranjal Pandey

---

## Overview

This project is a modular AI-powered web application built to detect and prevent a variety of digital scams prevalent in India. It provides a simple user interface where users can analyze different types of suspicious inputs using state-of-the-art machine learning models.

---

## Features

The platform currently supports detection for the following scam types:

### 1. Phishing URL Detection
- Paste a URL to determine if it’s a phishing attempt.
- **Model**: XGBoost trained on curated phishing datasets.

### 2. QR Code Malware Detection
- Upload a QR code image to check if it links to a malicious source.
- **Model**: Convolutional Neural Network (CNN) trained on benign and malicious QR images.

### 3. Scam Call Detection
- Upload a transcript or voice recording of a call to analyze if it’s fraudulent.
- **Model**: Hugging Face Transformer (for text classification) and OpenAI Whisper (for speech-to-text).

### 4. SMS Spam Detection
- Enter an SMS message to check if it’s spam or genuine.
- **Model**: NLP-based spam classifier to detect suspicious messages.

---

## Tech Stack

- **Backend**: Python, Flask
- **ML Frameworks**: TensorFlow, Scikit-learn, Hugging Face Transformers, XGBoost
- **Audio Transcription**: OpenAI Whisper
- **Frontend**: HTML, CSS, Bootstrap 5

---

This project is developed as part of an academic capstone and is intended for educational purposes.
