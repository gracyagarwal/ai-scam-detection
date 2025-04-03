from flask import Flask, render_template, request, jsonify
from model_utils import train_model, classify_message

app = Flask(__name__)

# Train the model when the app starts (returns a dictionary with 'english' and 'hindi' keys)
print("Training the model...")
model = train_model()
print("Model training completed!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    message = request.form.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'})
    
    # classify_message now takes the message and the model dictionary
    result = classify_message(message, model)
    return jsonify({
        'message': message,
        'classification': 'SPAM' if result == 'spam' else 'GENUINE',
        'class_color': 'red' if result == 'spam' else 'green'
    })

if __name__ == '__main__':
    app.run(debug=True)
