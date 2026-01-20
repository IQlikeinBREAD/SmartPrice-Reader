from ultralytics import YOLO
import cv2

class PriceTagDetector:
    def __init__(self, model_path = "models/yolo/custom_price_v1.pt", conf = 0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2])

        return detections