import tensorflow.keras.models as models
import cv2
import numpy as np
from draw import points_light


class LightClassifier:
    def __init__(self):
        self.model = models.load_model('wieght/emotion_Classifi_ver (2).h5')
        self.bounds_light = points_light()
        self.label = {0:'back', 1:'green', 2:'red',3: 'yellow'}

    def predict(self, img):
        x, y, w, h = self.bounds_light
        croped = img[y:y + h, x:x + w].copy()

        img = cv2.resize(croped, (82, 46))/255
        img = np.expand_dims(img, axis=0)
        label = np.argmax(self.model.predict(img))
        return self.label[label]

