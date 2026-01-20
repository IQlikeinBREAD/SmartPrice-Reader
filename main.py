# main.py (fragment)
import fastapi
from fastapi import UploadFile, File
from services.detector import PriceTagDetector
from services.reader import PriceReader
from utils.image_processing import bytes_to_cv2
import httpx
import re

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

    async def przeliczanie_walut(text, docelowa_waluta="PLN"):
        """
        Funkcja przelicza waluty na podstawie aktualnych kursów z API NBP.
        """
        waluty = ["USD", "EUR", "GBP", "CHF"]
        kursy = {}
        
        # Pobierz aktualne kursy z API NBP
        async with httpx.AsyncClient() as client:
            for waluta in waluty:
                try:
                    response = await client.get(
                        f"https://api.nbp.pl/api/exchangerates/rates/a/{waluta.lower()}/?format=json"
                    )
                    if response.status_code == 200:
                        data = response.json()
                        kursy[waluta] = data["rates"][0]["mid"]
                except Exception as e:
                    print(f"Błąd pobierania kursu {waluta}: {e}")
                    kursy[waluta] = None
        
        # Szukaj walut w tekście i przelicz
        for linia in text.split("\n"):
            for waluta, kurs in kursy.items():
                if kurs and waluta in linia:
                    try:
                        # Wyciągnij liczbę z linii
                        liczby = re.findall(r'\d+\.?\d*', linia)
                        if liczby:
                            kwota = float(liczby[0])
                            przeliczona_kwota = kwota * kurs
                            return {
                                "original": f"{kwota} {waluta}",
                                "converted": f"{przeliczona_kwota:.2f} {docelowa_waluta}",
                                "rate": kurs
                            }
                    except ValueError:
                        continue
        return None
    
    # Wywołaj przeliczanie walut dla każdego wyniku
    for result in results:
        conversion = await przeliczanie_walut(result["text"])
        if conversion:
            result["currency_conversion"] = conversion
    
    return {"results": results}