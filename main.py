import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import  Models

img = cv2.imread("images/images (2).jpg")
if img is None:
    print("Image not found. Check the file path.")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
model = models.get('yolo_nas_s', num_classes= 7, checkpoint_path='trained/ckpt_best.pth')

outputs = model.predict(img)
class_names = []
for detection in outputs.prediction.labels:
    class_names.append(detection)

print(class_names)

outputs.show()