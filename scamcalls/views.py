import os
from flask import Blueprint, request, jsonify, render_template_string
from transformers import pipeline
import whisper

print("✅ Scam Calls Blueprint Loaded with render_template_string")

os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"

call_bp = Blueprint("call", __name__, url_prefix="/call")

# Load model and tokenizer
scam_model_path = os.path.join(os.path.dirname(__file__), "new_scam_call_model")
scam_classifier = pipeline("text-classification", model=scam_model_path, tokenizer=scam_model_path)

# Load Whisper model
whisper_model = whisper.load_model("base")

# Upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HOME_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Scam Call Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #121212;
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
            padding-top: 40px;
        }
        .container {
            max-width: 700px;
        }
        .result-box {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            font-size: 18px;
        }
        .fraud {
            background-color: #ffebee;
            color: #b71c1c;
        }
        .normal {
            background-color: #e8f5e9;
            color: #1b5e20;
        }
    </style>
</head>
<body>
<div class="container">
    <h1 class="text-center mb-4">Scam Call Detection</h1>

    <form id="textForm">
        <div class="mb-3">
            <label for="call_text" class="form-label">Paste Call Transcript:</label>
            <textarea class="form-control" id="call_text" name="call_text" rows="5" required></textarea>
        </div>
        <button type="submit" class="btn btn-primary w-100">Analyze Transcript</button>
    </form>

    <div id="result" class="result-box d-none"></div>

    <hr class="my-4">

    <form id="audioForm" enctype="multipart/form-data">
        <div class="mb-3">
            <label for="audio" class="form-label">Upload Call Recording (mp3/wav):</label>
            <input class="form-control" type="file" id="audio" name="audio" accept="audio/*" required>
        </div>
        <button type="submit" class="btn btn-secondary w-100">Transcribe & Analyze Audio</button>
    </form>

    <div id="audioResult" class="result-box d-none"></div>

    <div style="margin-top: 30px; text-align: center;">
        <a href="{{ url_for('home') }}">
            <button style="padding: 8px 16px; background-color: #444; color: white; border: none; border-radius: 5px;">
                ⬅ Back to Home
            </button>
        </a>
    </div>
</div>

<script>
document.getElementById("textForm").addEventListener("submit", async function(e) {
    e.preventDefault();
    const text = document.getElementById("call_text").value;
    const resBox = document.getElementById("result");

    const response = await fetch("/call/predict", {
        method: "POST",
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: new URLSearchParams({ call_text: text })
    });
    const data = await response.json();
    resBox.className = "result-box " + (data.label === "Fraud" ? "fraud" : "normal");
    resBox.innerHTML = `<b>Result:</b> ${data.label}<br><b>Confidence:</b> ${data.confidence}`;
    resBox.classList.remove("d-none");
});

document.getElementById("audioForm").addEventListener("submit", async function(e) {
    e.preventDefault();
    const audioFile = document.getElementById("audio").files[0];
    const formData = new FormData();
    formData.append("audio", audioFile);
    const resBox = document.getElementById("audioResult");

    const response = await fetch("/call/transcribe", {
        method: "POST",
        body: formData
    });
    const data = await response.json();
    resBox.className = "result-box " + (data.label === "Fraud" ? "fraud" : "normal");
    resBox.innerHTML = `<b>Transcript:</b> ${data.transcript}<br><b>Result:</b> ${data.label}<br><b>Confidence:</b> ${data.confidence}`;
    resBox.classList.remove("d-none");
});
</script>
</body>
</html>
"""

@call_bp.route("/")
def home():
    return render_template_string(HOME_HTML)

@call_bp.route("/predict", methods=["POST"])
def predict():
    data = request.form['call_text']
    prediction = scam_classifier(data)[0]
    label_map = {"LABEL_1": "Fraud", "LABEL_0": "Normal"}
    return jsonify({
        "text": data,
        "label": label_map.get(prediction["label"], "Unknown"),
        "confidence": f"{round(prediction['score'] * 100, 2)}%"
    })

@call_bp.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if 'audio' not in request.files or request.files['audio'].filename == '':
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    transcript = whisper_model.transcribe(file_path)["text"]
    prediction = scam_classifier(transcript)[0]
    label_map = {"LABEL_1": "Fraud", "LABEL_0": "Normal"}

    return jsonify({
        "transcript": transcript,
        "label": label_map.get(prediction["label"], "Unknown"),
        "confidence": f"{round(prediction['score'] * 100, 2)}%"
    })
