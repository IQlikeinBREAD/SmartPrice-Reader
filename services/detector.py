from ultralytics import YOLO
import numpy as np
from utils.image_processing import crop_image


class PriceTagDetector:
    def __init__(self, model_path: str = "models/yolo11_best.pt", confidence_threshold: float = 0.4):
        """
        Inicjalizacja modelu YOLO.
        :param model_path: Ścieżka do wytrenowanego modelu (.pt)
        :param confidence_threshold: Minimalna pewność detekcji (0.0 - 1.0)
        """
        # Load model explicitly
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> list:
        """
        Wykrywa etykiety cenowe na obrazie.
        Zwraca listę słowników zawierających bbox, pewność i wycięty fragment obrazu.
        """
        results = self.model(image, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])

                # Filtrujemy wyniki poniżej progu pewności
                if confidence < self.conf_threshold:
                    continue

                # Pobranie koordynatów [x1, y1, x2, y2]
                bbox = box.xyxy[0].tolist()

                # Wycięcie fragmentu obrazu (crop) do późniejszego OCR
                cropped_img = crop_image(image, bbox)

                detections.append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "cropped_image": cropped_img,
                    "class_id": int(box.cls[0])
                })

        return detections
