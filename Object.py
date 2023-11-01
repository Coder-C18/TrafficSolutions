from shapely.geometry import Polygon, box
from draw import lane
import numpy as np
import cv2

red = (0, 255, 0)
blue = (255, 0, 0)
think_ness = 2


class Vehicle:

    def __init__(self, bounding_box):
        self.bb = np.array(bounding_box, np.int32).reshape(-1)
        self.location_moved = [self.bb]
        self.status_violate = None
        self.in_right_lane = self.check_in_right(bounding_box)

    def draw(self, frame, idx):
        print(self.bb)
        x1, y1, x2, y2 = self.bb
        color = red if self.status_violate else blue

        cv2.putText(img=frame,
                    text=str(idx),
                    org=(x1 - 10, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9,
                    color=color,
                    thickness=think_ness)
        cv2.rectangle(img=frame,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=color,
                      thickness=think_ness)
        return frame

    def check_in_right(self, bounding_box):
        xmin, ymin, xmax, ymax = bounding_box
        lane_ = Polygon(lane)
        bounding_box = box(xmin, ymin, xmax, ymax)

        intersection_area = bounding_box.intersection(lane_).area
        bbox_area = bounding_box.area

        percentage_overlap = (intersection_area / bbox_area) * 100

        if percentage_overlap > 20:
            return True
        else:
            return False

    def check_wrong_lane(self, current_location):
        self.status_violate = True

    def check_blow_the_red_light(self, light_status):
        if self.in_right_lane:
            if light_status == 'red':
                t1 = self.check_in_right(self.location_moved[-1])
                t2 = self.check_in_right(self.bb)
                if t1 and t2:
                    pass
                else:
                    self.status_violate = True
        else:
            pass

    def recognite_license_plates(self):
        self.status_violate = True

    def update(self, bounding_box, light_status):

        self.bb = np.array(bounding_box, np.int32).reshape(-1)
        self.check_blow_the_red_light(light_status)
        self.location_moved .append(self.bb)
        print(self.location_moved)
        # self.check_wrong_lane(location)
