import os
from flask import Blueprint, request, jsonify, render_template_string
from sms_filter.model_utils import train_model, classify_message, explain_message

sms_bp = Blueprint('sms', __name__, url_prefix='/sms')

# Train the model on blueprint load
print("Training SMS scam filter model...")
model = train_model()
print("SMS model trained successfully!")

HTML_PAGE = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <title>SMS Scam Detection</title>
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
  <style>
    body {
      background-color: #121212;
      color: #fff;
      font-family: 'Segoe UI', Tahoma, sans-serif;
    }
    .container {
      margin-top: 60px;
      max-width: 700px;
    }
    .card {
      background-color: #1e1e1e;
      padding: 30px;
      border-radius: 10px;
    }
    .btn-primary {
      background-color: #0d6efd;
      border: none;
    }
    .btn-primary:hover {
      background-color: #0b5ed7;
    }
    #result {
      margin-top: 20px;
      padding: 15px;
      border-radius: 5px;
      font-size: 18px;
      display: none;
    }
    .green {
      background-color: #e0f7e9;
      color: #2e7d32;
      border-left: 5px solid #10a37f;
    }
    .red {
      background-color: #ffebee;
      color: #c62828;
      border-left: 5px solid #e53935;
    }
    label.form-label {
      color: white;
    }
    details {
      margin-top: 10px;
      font-size: 16px;
    }
  </style>
</head>
<body>
  <div class=\"container\">
    <h1 class=\"text-center mb-4\">SMS Scam Filter</h1>
    <div class=\"card\">
      <form id=\"smsForm\">
        <div class=\"mb-3\">
          <label for=\"message\" class=\"form-label\">Enter SMS Message:</label>
          <textarea class=\"form-control\" id=\"message\" rows=\"3\" required></textarea>
        </div>
        <button id=\"submitBtn\" type=\"submit\" class=\"btn btn-primary w-100\">Classify & Explain Message</button>
      </form>
      <div id=\"result\"></div>
    </div>
  </div>

  <div style="margin-top: 30px; text-align: center;">
    <a href="{{ url_for('home') }}">
      <button style="padding: 8px 16px; background-color: #444; color: white; border: none; border-radius: 5px; font-family: 'Segoe UI', Tahoma, sans-serif;">
        ← Back to Home
      </button>
    </a>
  </div>

  <script>
    document.getElementById('smsForm').addEventListener('submit', async function(e) {
      e.preventDefault();
      const msg = document.getElementById('message').value;
      const resultDiv = document.getElementById('result');
      const submitBtn = document.getElementById('submitBtn');

      resultDiv.style.display = 'none';
      submitBtn.disabled = true;
      submitBtn.innerHTML = 'Classifying...';

      const response = await fetch('/sms/explain', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: new URLSearchParams({ message: msg })
      });

      const data = await response.json();
      submitBtn.disabled = false;
      submitBtn.innerHTML = 'Classify & Explain Message';

      if (data.error) {
        resultDiv.className = 'alert alert-warning';
        resultDiv.textContent = data.error;
      } else {
        let summaryText = "";
        let detailsList = "";

        data.explanation.forEach((item, idx) => {
          if (idx === 0 && item[0] === "Summary") {
            summaryText = `<strong>Explanation:</strong> ${item[2]}`;
          } else {
            detailsList += `<li><b>${item[0]}</b>: ${item[2]} (${item[1].toFixed(3)})</li>`;
          }
        });

        let explanationHTML = `
          ${summaryText}
          <details>
            <summary>See word-level breakdown</summary>
            <ul>${detailsList}</ul>
          </details>
        `;

        resultDiv.innerHTML = `<strong>Classification:</strong> ${data.classification.toUpperCase()}<br/>${explanationHTML}`;
        resultDiv.className = data.class_color;
      }
      resultDiv.style.display = 'block';
    });
  </script>
</body>
</html>
"""

@sms_bp.route('/')
def home():
    return render_template_string(HTML_PAGE)

@sms_bp.route('/classify', methods=['POST'])
def classify():
    message = request.form.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'})

    result = classify_message(message, model, threshold=0.5)
    return jsonify({
        'message': message,
        'classification': 'SCAM' if result == 'scam' else 'GENUINE',
        'class_color': 'red' if result == 'scam' else 'green'
    })

@sms_bp.route('/explain', methods=['POST'])
def explain():
    message = request.form.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'})

    result = classify_message(message, model, threshold=0.5)
    explanation = explain_message(message, model)

    return jsonify({
        'message': message,
        'classification': 'SCAM' if result == 'scam' else 'GENUINE',
        'explanation': explanation,
        'class_color': 'red' if result == 'scam' else 'green'
    })