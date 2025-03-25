from flask import Flask, render_template_string
from phishingurl import url_bp  
from qrcode_module import qr_bp
from scamcalls import call_bp
from sms_filter import sms_bp

app = Flask(__name__)

# Register blueprint
app.register_blueprint(url_bp)
app.register_blueprint(qr_bp) 
app.register_blueprint(call_bp)
app.register_blueprint(sms_bp) 



HOME_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Scam Detection</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
        background-color: #121212;
        font-family: 'Segoe UI', Tahoma, sans-serif;
        color: #f0f0f0;
    }
    .card {
        background-color: #1e1e1e;
        border: none;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }
    .card-title {
        font-weight: 600;
        color: #ffffff;
    }
    .card-text {
        color: #d0d0d0;
        font-size: 15px;
    }
    .btn-primary {
        background-color: #0d6efd;
        border: none;
        border-radius: 8px;
        color: #ffffff;
    }
    .btn-primary:hover {
        background-color: #0b5ed7;
    }
    p.text-muted {
        color: #bbbbbb !important;
    }
    footer {
        margin-top: 60px;
        font-size: 14px;
        color: #bbbbbb !important;
        transition: opacity 0.3s ease;
    }
    footer:hover {
        opacity: 0.8;
    }
  </style>
</head>
<body>
  <div class="container py-5">
    <h1 class="text-center fw-bold mb-3">AI Scam Detection</h1>
    <p class="text-center text-muted mb-5">Choose a scam type to analyze using machine learning.</p>

    <div class="row g-4">

      <!-- QR Code -->
      <div class="col-md-6 col-lg-3">
        <div class="card h-100 text-center">
          <div class="card-body">
            <h5 class="card-title">QR Code</h5>
            <p class="card-text">Scan a QR to check if it's hiding malicious intent.</p>
            <a href="/qr/" class="btn btn-primary w-100">Try QR Check</a>
          </div>
        </div>
      </div>

      <!-- Scam Call -->
      <div class="col-md-6 col-lg-3">
        <div class="card h-100 text-center">
          <div class="card-body">
            <h5 class="card-title">Scam Call</h5>
            <p class="card-text">Upload transcripts or audio to catch suspicious calls.</p>
            <a href="/call/" class="btn btn-primary w-100">Try Call Check</a>
          </div>
        </div>
      </div>

      <!-- Phishing URL -->
      <div class="col-md-6 col-lg-3">
        <div class="card h-100 text-center">
          <div class="card-body">
            <h5 class="card-title">Phishing URL</h5>
            <p class="card-text">Paste any link to detect phishing websites.</p>
            <a href="/url/" class="btn btn-primary w-100">Try URL Check</a> 
          </div>
        </div>
      </div>

      <!-- SMS Scam -->
      <div class="col-md-6 col-lg-3">
        <div class="card h-100 text-center">
          <div class="card-body">
            <h5 class="card-title">SMS Filter</h5>
            <p class="card-text">Enter SMS content to check if it’s suspicious.</p>
            <a href="/sms/" class="btn btn-primary w-100">Try SMS Check</a>
          </div>
        </div>
      </div>

    </div>
  </div>

  <footer class="text-center py-4">
    Made by Gracy Agarwal & Pranjal Pandey – 2025
  </footer>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HOME_PAGE)

if __name__ == '__main__':
    app.run(debug=True)
