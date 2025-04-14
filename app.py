from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

model = load_model('xray_model.h5')

def preprocess(img):
    img = img.resize((224, 224))  # Adjust based on your model
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
def index():
    return 'X-ray Model API is Live 🚀'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    processed = preprocess(img)
    prediction = model.predict(processed)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
