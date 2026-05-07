from ultralytics import YOLO
from config import MODEL_PATH, THRESHOLD

model = YOLO(MODEL_PATH)


def parse_detections(frame):
    results = model(frame, verbose=False)[0]

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if conf > THRESHOLD:
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)

            detections.append((x, y, w, h, cls, conf))

    return detections
