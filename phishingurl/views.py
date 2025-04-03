from flask import Blueprint, request, render_template_string, url_for
import os
from .phishing_detector import check_url  # Ensure this import path is correct

url_bp = Blueprint('url', __name__, url_prefix='/url')

HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Phishing URL Detector</title>
    <style>
        body {
            background-color: #121212;
            color: #eee;
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 40px;
        }
        h1 {
            text-align: center;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            border: 1px solid #555;
            background: #1e1e1e;
            color: white;
        }
        button {
            padding: 10px 20px;
            background-color: #0d6efd;
            border: none;
            color: white;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #222;
            border-left: 4px solid #10a37f;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <h1>Phishing URL Checker</h1>
    <form method="post">
        <input type="text" name="url" placeholder="Paste URL here" required>
        <button type="submit">Check URL</button>
    </form>
    {% if result %}
    <div class="result">
        <pre style="margin: 0;">{{ result }}</pre>
    </div>
    {% endif %}
    <div style="margin-top: 30px; text-align: center;">
        <a href="{{ url_for('home') }}">
            <button style="padding: 8px 16px; background-color: #444; color: white; border: none; border-radius: 5px;">
                ⬅ Back to Home
            </button>
        </a>
    </div>
</body>
</html>
"""

@url_bp.route('/', methods=['GET', 'POST'])
def phishing_home():
    result = ""
    if request.method == 'POST':
        url = request.form.get('url')
        if url:
            result = check_url(url)
        else:
            result = "Please provide a valid URL."
    return render_template_string(HOME_HTML, result=result)