import easyocr
import logging

logging.getLogger("easyocr").setLevel(logging.ERROR)

class PriceReader:
    def __init__(self, use_gpu=True):
        self.reader = easyocr.Reader(['pl', 'en'], gpu=use_gpu)

    def read_text(self, image_crop):
        if image_crop is None or image_crop.size == 0:
            return []
        
        results = self.reader.readtext(image_crop)

        detected_texts = []
        for(bbox, text, prob) in results:
            if prob > 0.3:
                detected_texts.append(text)
        
        return detected_texts