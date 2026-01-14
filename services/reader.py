from paddleocr import PaddleOCR
import numpy as np
import logging

# Wyłączamy logi systemowe PaddleOCR, żeby nie śmieciły w konsoli
logging.getLogger("ppocr").setLevel(logging.ERROR)


class PriceReader:
    def __init__(self, lang: str = 'en'):
        """
        Inicjalizacja PaddleOCR.
        use_angle_cls=True pozwala czytać tekst obrócony (np. o 90 stopni).
        """
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def read_text(self, image: np.ndarray) -> str:
        """
        Wykonuje OCR na podanym obrazie (lub jego wycinku).
        Zwraca połączony ciąg znaków.
        """
        if image is None or image.size == 0:
            return ""

        # Wykonanie OCR
        # cls=True włącza klasyfikację kąta obrotu
        result = self.ocr.ocr(image, cls=True)

        if not result or result[0] is None:
            return ""

        detected_lines = []
        # Struktura wyniku PaddleOCR: [[[[x1,y1],...], ("text", conf)], ...]
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]

            # Możemy dodać minimalny próg pewności dla samego tekstu
            if confidence > 0.5:
                detected_lines.append(text)

        # Łączymy linie tekstu spacją
        full_text = " ".join(detected_lines)
        return full_text.strip()
