import cv2
import json
import numpy as np

data_point = json.load(open("Data/312.json"))
lane = None
for data in data_point['shapes']:
    if data['label'] == 'rightlane':
        lane = data['points']


def points_light():
    for data in data_point['shapes']:
        if data['label'] == 'light':
            rect = cv2.boundingRect(np.array(data['points'], np.int32))
            return rect


def draw_image(image, status_light):
    for data in data_point['shapes']:

        label = data['label']
        points = np.array(data['points'], np.int32)
        points = points.reshape((-1, 1, 2))
        color = (255, 0, 0)
        thickness = 2
        image = cv2.polylines(image,
                              [points],
                              isClosed=True,
                              color=color,
                              thickness=thickness
                              )
        x, y = data['points'][0]
        if data['label'] == 'light':
            label = status_light
        cv2.putText(image, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return image
