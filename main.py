# main.py (fragment)
import fastapi
from fastapi import UploadFile, File
from services.detector import PriceTagDetector
from services.reader import PriceReader
from utils.image_processing import bytes_to_cv2

# Inicjalizacja usług (Singleton pattern - ładowane raz przy starcie)
detector = PriceTagDetector(model_path="yolo11n.pt")  # lub twoja ścieżka
reader = PriceReader()
app = fastapi.FastAPI()

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    # 1. Przetworzenie bajtów na obraz
    image_bytes = await file.read()
    image = bytes_to_cv2(image_bytes)

    # 2. Detekcja (gdzie jest cena?)
    detections = detector.detect(image)

    results = []

    # A. Jeśli wykryto etykiety -> czytaj tylko z nich
    if detections:
        for item in detections:
            text = reader.read_text(item["cropped_image"])
            results.append({
                "type": "tag_crop",
                "text": text,
                "confidence": item["confidence"],
                "bbox": item["bbox"]
            })

    # B. Jeśli nic nie wykryto -> czytaj cały obraz (fallback)
    else:
        full_text = reader.read_text(image)
        results.append({
            "type": "full_image_fallback",
            "text": full_text,
            "confidence": 0.0,
            "bbox": []
        })

    # Tu dodajesz logikę parsowania walut i NBP...

    def przeliczanie_walut(text, docelowa_waluta):
        # Przykładowa implementacja - zastąp rzeczywistą logiką
        waluty = ["USD", "EUR", "GBP", "CHF"]
        # URL do API NBP: https://api.nbp.pl/api/exchangerates/rates/a/chf/?format=json
        kursy = {"USD": 4.0, "EUR": 4.3, "GBP": 5.0, "CHF": 4.5}  # Przykładowe kursy
        for linia in text.split("\n"):
            for waluta, kurs in kursy.items():
                if waluta in linia:
                    try:
                        kwota = float(''.join(filter(lambda x: x.isdigit() or x == '.', linia)))
                        przeliczona_kwota = kwota * kurs
                        return f"{kwota} {waluta} to około {przeliczona_kwota:.2f} PLN"
                    except ValueError:
                        continue
        return "Nie znaleziono kwot do przeliczenia."
    return {"results": results}