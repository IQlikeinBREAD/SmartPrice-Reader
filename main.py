# main.py (fragment)
import fastapi
from fastapi import UploadFile, File, HTTPException
from services.detector import PriceTagDetector
from services.reader import PriceReader
from utils.text_utils import clean_price
import numpy as np
import cv2
import httpx
import re
from typing import Dict, List, Optional

# Inicjalizacja usług (Singleton pattern - ładowane raz przy starcie)
detector = PriceTagDetector(model_path="models/yolo/custom_price_v1.pt")
reader = PriceReader(use_gpu=False)
app = fastapi.FastAPI()

CURRENCIES = ["USD", "EUR", "GBP", "CHF"]


def _decode_image(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Nie udało się odczytać obrazu z przesłanych bajtów")
    return frame


def _normalize_bbox(frame, bbox):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _build_results(frame) -> List[Dict]:
    detections = detector.detect(frame)
    if not detections:
        raw = reader.read_text(frame)
        return [{
            "type": "full_frame",
            "raw_text": raw,
            "price": clean_price(raw),
            "confidence": 0.0,
            "bbox": []
        }]

    normalized = _normalize_bbox(frame, detections[0])
    if normalized is None:
        raw = reader.read_text(frame)
        return [{
            "type": "full_frame",
            "raw_text": raw,
            "price": clean_price(raw),
            "confidence": 0.0,
            "bbox": []
        }]

    x1, y1, x2, y2 = normalized
    crop = frame[y1:y2, x1:x2]
    raw = reader.read_text(crop)
    return [{
        "type": "tag_crop",
        "raw_text": raw,
        "price": clean_price(raw),
        "confidence": None,
        "bbox": [x1, y1, x2, y2]
    }]


def _fetch_exchange_rates() -> Dict[str, Optional[float]]:
    rates: Dict[str, Optional[float]] = {}
    with httpx.Client() as client:
        for currency in CURRENCIES:
            try:
                response = client.get(f"https://api.nbp.pl/api/exchangerates/rates/a/{currency.lower()}/?format=json", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    rates[currency] = data["rates"][0]["mid"]
                else:
                    rates[currency] = None
            except Exception as exc:
                print(f"Błąd pobierania kursu {currency}: {exc}")
                rates[currency] = None
    return rates


def _parse_currency(text_lines: List[str], target: Optional[str] = None):
    joined = "\n".join(text_lines)
    rates = _fetch_exchange_rates()

    for line in joined.split("\n"):
        numbers = re.findall(r"\d+\.?\d*", line)
        if not numbers:
            continue
        amount = float(numbers[0])
        if "PLN" in line.upper():
            # Przelicz na wszystkie waluty z listy CURRENCIES
            conversions = {
                "original": f"{amount} PLN"
            }
            
            for currency in CURRENCIES:
                if currency in rates and rates[currency]:
                    rate = rates[currency]
                    converted = amount / rate
                    conversions[currency] = {
                        "value": round(converted, 2),
                        "rate": rate,
                        "full": f"{converted:.2f} {currency}"
                    }
                else:
                    conversions[currency] = {
                        "value": None,
                        "rate": None,
                        "full": "Błąd"
                    }
            
            return conversions
    return None


@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        frame = _decode_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    results = _build_results(frame)
    for item in results:
        conversion = _parse_currency(item["raw_text"])
        if conversion:
            item["currency_conversion"] = conversion
    return {"results": results}