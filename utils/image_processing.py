import cv2
import numpy as np
from typing import Tuple, Optional


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """
    Konwertuje surowe bajty (np. z uploadu FastAPI) na obraz OpenCV (numpy array).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Nie udało się zdekodować obrazu. Sprawdź format pliku.")
    return img


def crop_image(image: np.ndarray, bbox: list) -> np.ndarray:
    """
    Wycina fragment obrazu na podstawie współrzędnych [x1, y1, x2, y2].
    Zabezpiecza przed wyjściem poza ramy obrazu.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w, _ = image.shape

    # Clamp coordinates to image dimensions
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return image[y1:y2, x1:x2]


def draw_bbox(image: np.ndarray, bbox: list, label: str = "", color: Tuple = (0, 255, 0)) -> np.ndarray:
    """
    Rysuje ramkę i opcjonalny tekst na obrazie (przydatne do debugowania/zwracania podglądu).
    """
    x1, y1, x2, y2 = map(int, bbox)
    img_copy = image.copy()
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

    if label:
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img_copy
