from flask import Blueprint, request, render_template_string, url_for
from .phishing_detector import check_url, explain_url

url_bp = Blueprint('url', __name__, url_prefix='/url')

HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Phishing URL Detector</title>
    <!-- Bootstrap CSS from CDN -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #121212;
            color: #eee;
        }
        .card {
            background-color: #1e1e1e;
            border: none;
            border-radius: 8px;
        }
        .card-body {
            padding: 20px;
        }
        .result, details.explanation {
            margin-top: 20px;
            padding: 15px;
            background-color: #222;
            border-left: 4px solid #10a37f;
            border-radius: 5px;
        }
        details.explanation summary {
            font-weight: bold;
            cursor: pointer;
            margin-bottom: 10px;
        }
        details.explanation ul {
            list-style-type: disc;
            padding-left: 25px;
            margin-top: 10px;
        }
        details.explanation li {
            margin-bottom: 8px;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center mb-4">Phishing URL Detector</h1>
        <div class="card">
            <div class="card-body">
                <form method="post">
                    <div class="mb-3">
                        <label for="urlInput" class="form-label">Paste URL here:</label>
                        <input type="text" name="url" id="urlInput" class="form-control" placeholder="https://example.com" required>
                    </div>
                    <div class="mb-3">
                        <label for="explainSelect" class="form-label">Explanation:</label>
                        <select name="explain" id="explainSelect" class="form-select">
                            <option value="no" selected>No Explanation</option>
                            <option value="yes">Get Explanation</option>
                        </select>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Check URL</button>
                </form>
            </div>
        </div>
        {% if result %}
        <div class="result mt-3">
            <pre class="m-0">{{ result }}</pre>
        </div>
        {% endif %}
        {% if explanation %}
        <details class="explanation mt-3" open>
            <summary>Show Explanation</summary>
            <ul class="mb-0">
                {% for feature, weight, message in explanation %}
                    <li><strong>{{ feature }}</strong>: {{ message }}</li>
                {% endfor %}
            </ul>
        </details>
        {% endif %}
        <div style="margin-top: 30px; text-align: center;">
            <a href="{{ url_for('home') }}">
                <button style="padding: 8px 16px; background-color: #444; color: white; border: none; border-radius: 5px; font-family: 'Segoe UI', Tahoma, sans-serif;">
                ← Back to Home
                </button>
            </a>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@url_bp.route('/', methods=['GET', 'POST'])
def phishing_home():
    result = ""
    explanation = None
    if request.method == 'POST':
        url_input = request.form.get('url')
        if url_input:
            result = check_url(url_input)
            if request.form.get('explain') == 'yes':
                explanation = explain_url(url_input)
        else:
            result = "Please provide a valid URL."
    return render_template_string(HOME_HTML, result=result, explanation=explanation)
