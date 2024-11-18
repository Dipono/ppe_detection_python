from flask import Flask, request, jsonify
import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models
import numpy as np
from flask_cors import CORS
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

CORS(app)
# Load the pre-trained YOLO model (assuming the model is already trained)
model = models.get('yolo_nas_s', num_classes=7, checkpoint_path='trained/ckpt_best.pth')


# Define the route for the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Convert the image to RGB (as the model expects RGB format)
    img = Image.open(file).convert('RGB')
    img = np.array(img)

    # Ensure that the image is in the correct format (RGB, numpy array)
    if img is None:
        return jsonify({"error": "Image could not be read"}), 400

    # Make the prediction
    outputs = model.predict(img)

    # Get the predicted class labels
    class_names = []
    for detection in outputs.prediction.labels:
        class_names.append(detection)

    # Get the class names from the model's predefined class names
    class_names_mapped = [outputs.class_names[label] for label in class_names]

    # Return the class names in the response
    class_names_mapped = list(dict.fromkeys(class_names_mapped))
    return jsonify({"predicted_class_names": class_names_mapped})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
