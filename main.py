import cv2
import os
from draw import draw_image
from deep_sort_realtime.deepsort_tracker import DeepSort
from Model import LightClassifier, Detector
from Object import Vehicle

# init model
tracker = DeepSort(max_age=5)
light_model = LightClassifier()
detector = Detector()

# Open video
cap = cv2.VideoCapture('Data/Videos/CAMERA NGÃ TƯ QUANG TRUNG - NGUYỄN THỊ MINH KHAI trưa 06_06_2021 10h55-12h55.mp4')

frame_rate = 5  # 1 frame per second
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / frame_rate)
frame_count = 0

road = dict()

while True:

    # Đọc khung hình hiện tại từ hàng đợi
    ret, frame = cap.read()

    frame_count += 1

    if frame_count % frame_interval == 0:

        status_light = light_model.predict(frame)

        result = detector.model(frame)
        # result.render()
        pd = result.pandas().xyxy[0]

        pd['w'] = pd['xmax'] - pd['xmin']
        pd['h'] = pd['ymax'] - pd['ymin']
        pd['points'] = list(pd[['xmin', 'ymin', 'w', 'h']].itertuples(index=False, name=None))

        bbs = list(pd[['points', 'confidence', 'class']].itertuples(index=False, name=None))

        tracks = tracker.update_tracks(bbs, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            idx = track.track_id
            bounding_box = track.to_ltrb()

            if road.get(idx) is None:
                road[idx] = Vehicle(bounding_box)

            road[idx].update(bounding_box, status_light)
            frame = road[idx].draw(frame, idx)
        frame = draw_image(frame, status_light)

        # Nếu không có khung hình nào nữa thì thoát
        if not ret:
            break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        cv2.imshow('video', frame)

# Đóng video
cap.release()
