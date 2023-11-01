import torch
import tensorflow.keras.models as models
import cv2
import numpy as np
from draw import points_light


class Detector:
    def __init__(self):
        self.model = torch.hub.load('yolov5', 'custom', path='wieght/yolov5s.pt', force_reload=True, source='local')
        self.model.eval()
        self.model.conf = 0.35
        self.model.classes = [1, 2, 3, 5, 7, 13]
        self.model.iou_threshold = 0.7


class LightClassifier:
    def __init__(self):
        self.model = models.load_model('wieght/lightclassify.h5')
        self.bounds_light = points_light()
        self.label = {0: 'back', 1: 'green', 2: 'red', 3: 'yellow'}

    def predict(self, img):
        x, y, w, h = self.bounds_light
        croped = img[y:y + h, x:x + w].copy()

        img = cv2.resize(croped, (82, 46)) / 255
        img = np.expand_dims(img, axis=0)
        label = np.argmax(self.model.predict(img))
        return self.label[label]


class HelmetViolation:
    def __init__(self):
        self.model = models.load_model('wieght/helmetviolation.h5')

    def predict(self, image):
        img = cv2.resize(image, (256, 256)) / 255
        img = np.expand_dims(img, axis=0)
        label = np.argmax(self.model.predict(img))
        return
