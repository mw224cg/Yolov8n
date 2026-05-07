from detection_yolo import parse_detections
from config import PERSON_CLASS_ID

import cv2
import time

class CvHandler:
    def __init__(self, camera):
        self.camera = camera

    def get_frame(self):
        return self.camera.get_frame()
    
    def get_detections(self, frame):
        detections = parse_detections(frame)
        return detections
    
    def get_people_count(self, detections):
        people_count = 0
        for detection in detections:
            if detection[4] == PERSON_CLASS_ID:
                people_count +=1
        return people_count
    
    #captures a frame from the camera and returns num of people in the frame
    def capture_and_get_people_count(self):
        frame = self.get_frame()
        detections = self.get_detections(frame)
        people_count = self.get_people_count(detections)
        return people_count

    #draws bounding boxes around detections, num of detected people as top label
    def draw_detections(self, frame, detections):
        people_count = self.get_people_count(detections)

        for x, y, w, h, cls, conf in detections:
            label = f"{cls} ({conf:.2f})"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        cv2.putText(
            frame,
            f"People: {people_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return frame
    
    #Runs a preview of the camera model for 'duration' seconds then closes window
    def preview(self, duration):
        print("Starting camera model preview mode...")
        start_time = time.time()

        while time.time() - start_time < duration:
            raw_frame = self.get_frame()
            detections = self.get_detections(raw_frame)
            frame_with_boxes = self.draw_detections(raw_frame, detections)

            cv2.imshow("Spotiqa Camera Setup", frame_with_boxes)
            cv2.waitKey(1)

        cv2.destroyAllWindows()
