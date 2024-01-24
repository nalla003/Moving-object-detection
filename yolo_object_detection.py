# yolo_object_detection.py
import cv2
import numpy as np
from object_detection_setup import setup_object_detection
from yolo_configuration import configure_yolo_model

def perform_object_detection(image_path):
    net, classes, layer_names = setup_object_detection()
    configure_yolo_model()

    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Rest of the object detection logic (as per the previous example)
    # ...

    # Display results
    cv2.imshow("Object Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)

# Example usage:
# perform_object_detection("path/to/your/image.jpg")