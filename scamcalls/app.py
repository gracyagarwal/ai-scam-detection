import os
os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"


from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import whisper
import os

app = Flask(__name__)

# Load trained scam detection model
scam_classifier = pipeline("text-classification", model="new_scam_call_model", tokenizer="new_scam_call_model")

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# Ensure "uploads" folder exists for audio files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['call_text']
    prediction = scam_classifier(data)[0]

    # Convert label_1 to "Fraud" and label_0 to "Normal"
    label_map = {"LABEL_1": "Fraud", "LABEL_0": "Normal"}
    readable_label = label_map.get(prediction['label'], "Unknown")

    # Convert confidence score to percentage
    confidence_percent = round(prediction['score'] * 100, 2)

    return jsonify({
        "text": data,
        "label": readable_label,
        "confidence": f"{confidence_percent}%"
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Transcribe using Whisper
    transcript = whisper_model.transcribe(file_path)["text"]

    # Run scam detection on transcribed text
    prediction = scam_classifier(transcript)[0]
    label_map = {"LABEL_1": "Fraud", "LABEL_0": "Normal"}
    readable_label = label_map.get(prediction['label'], "Unknown")
    confidence_percent = round(prediction['score'] * 100, 2)

    return jsonify({
        "transcript": transcript,
        "label": readable_label,
        "confidence": f"{confidence_percent}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
