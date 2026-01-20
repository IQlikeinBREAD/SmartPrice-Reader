import easyocr
import logging

logging.getLogger("easyocr").setLevel(logging.ERROR)

class PriceReader:
<<<<<<< HEAD
    def __init__(self, use_gpu=True):
        self.reader = easyocr.Reader(['pl', 'en'], gpu=use_gpu)
=======
    def __init__(self, lang: str = 'en'):
        """
        Inicjalizacja PaddleOCR.
        use_angle_cls=True pozwala czytać tekst obrócony (np. o 90 stopni).
        """
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
>>>>>>> 415decf793b429bdd82ad438bc5ec80c1d71304a

    def read_text(self, image_crop):
        if image_crop is None or image_crop.size == 0:
            return []
        
        results = self.reader.readtext(image_crop)

        detected_texts = []
        for(bbox, text, prob) in results:
            if prob > 0.3:
                detected_texts.append(text)
        
        return detected_texts