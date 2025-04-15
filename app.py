from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = load_model('xray_model.h5')

# Define class labels (you MUST match this with the order used during training)
class_names = ['Normal', 'Pneumonia', 'COVID-19']  # Change if needed

def preprocess(img):
    # Resize and normalize the image
    img = img.resize((64, 64))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    processed = preprocess(img)

    # Predict using model
    prediction = model.predict(processed)

    # Get class index and label
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index])

    return jsonify({
        'prediction': predicted_label,
        'confidence': round(confidence, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
