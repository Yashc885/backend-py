from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
import io
from PIL import Image
from flask import render_template

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = load_model('xray_model.h5')

def preprocess(img):
    # Resize the image to 64x64
    img = img.resize((64, 64))  # Resize to (64, 64)
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension, shape becomes (1, 64, 64, 3)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')  # Read and convert image to RGB
    processed = preprocess(img)  # Preprocess the image (resize and normalize)
    
    # Get the prediction
    prediction = model.predict(processed)
    
    # Assuming it's a classification task
    return jsonify({'prediction': prediction.tolist()})  # Return the prediction as a list

if __name__ == '__main__':
    app.run(debug=True)
