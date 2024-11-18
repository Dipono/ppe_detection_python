
import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import  Models

img = cv2.imread("images/Make.png")
if img is None:
    print("Image not found. Check the file path.")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
model = models.get('yolo_nas_s', num_classes= 7, checkpoint_path='trained/ckpt_best(1).pth')

outputs = model.predict(img)
outputs.show()

#
# import torch
# import cv2
# import numpy as np


'''
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

MODEL_PATH = 'ptrained/ckpt_best.pth'

# Load the model
def load_model():
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()  # Set to evaluation mode
    return model

# Define preprocessing transformations
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model's expected input size
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define post-processing for bounding boxes (assuming model output includes them)
def postprocess_boxes(output, conf_threshold=0.5):
    boxes, labels, scores = [], [], []
    for item in output:
        if item['score'] >= conf_threshold:
            boxes.append(item['box'])
            labels.append(item['label'])
            scores.append(item['score'])
    return boxes, labels, scores

# Perform detection on an image
def detect_ppe(image_path, model):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess_image(image)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

    # Post-process to get bounding boxes, labels, and scores
    boxes, labels, scores = postprocess_boxes(output)

    # Display the results
    img_np = np.array(image)
    for box, label, score in zip(boxes, labels, scores):
        cv2.rectangle(img_np, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(img_np, f"{label} {score:.2f}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("PPE Detection", img_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    model = load_model()
    detect_ppe("images/images(4).jpg", model)
    
    '''