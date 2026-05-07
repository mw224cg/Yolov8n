from camera_webcam import Camera
from cv_handler import CvHandler


def main():
    cam = Camera()
    cam.start()
    cv_handler = CvHandler(cam)

    cv_handler.preview(60)


if __name__ == "__main__":
    main()
