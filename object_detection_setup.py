# object_detection_setup.py
import cv2
import numpy as np

def setup_object_detection():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getUnconnectedOutLayersNames()

    return net, classes, layer_names