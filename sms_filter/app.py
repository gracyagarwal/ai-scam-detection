from flask import Flask, render_template, request, jsonify
from sms_filter import train_model, classify_message

app = Flask(__name__)

# Train the model when the app starts
print("Training the model...")
model, vectorizer = train_model()
print("Model training completed!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    message = request.form.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'})
    
    result = classify_message(message, model, vectorizer)
    return jsonify({
        'message': message,
        'classification': 'SPAM' if result == 'spam' else 'GENUINE',
        'class_color': 'red' if result == 'spam' else 'green'
    })

if __name__ == '__main__':
    app.run(debug=True) 