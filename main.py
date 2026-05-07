from camera_webcam import Camera
from detection_yolo import parse_detections
from utils import draw
from cv_handler import CvHandler
import cv2


def main():
    cam = Camera()
    cam.start()
    cv_handler = CvHandler(cam)

    cv_handler.preview(60)


if __name__ == "__main__":
    main()
