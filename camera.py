import cv2


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None

    def start(self):
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame = frame
        return frame

    def release(self):
        self.cap.release()
