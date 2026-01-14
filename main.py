from fastapi import FastAPI, File, UploadFile
from services.detector import PriceTagDetector
from services.reader import PriceReader
from utils.image_processing import bytes_to_cv2

# aplikacja FastAPI
app = FastAPI()

# Inicjalizacja usług (Singleton pattern - ładowane raz przy starcie)
detector = PriceTagDetector(model_path="/models/yolo11n.pt")  # lub twoja ścieżka
reader = PriceReader()


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
    return {"results": results}