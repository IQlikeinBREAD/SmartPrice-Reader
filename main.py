from fastapi import FastAPI, File, UploadFile
from services.currency import NBPService
from services.detector import PriceTagDetector
from services.reader import PriceReader
from utils.image_processing import bytes_to_cv2
import re

# aplikacja FastAPI
app = FastAPI()

# Inicjalizacja usług (Singleton pattern - ładowane raz przy starcie)
detector = PriceTagDetector(model_path="/models/yolo11n.pt")  # lub twoja ścieżka
reader = PriceReader()

# cache kursów, żeby nie wołać API wielokrotnie
_rate_cache = {}


def _get_cached_rate(currency_code: str) -> float | None:
    code = currency_code.upper()
    if code in _rate_cache:
        return _rate_cache[code]
    rate = NBPService.get_exchange_rate(code)
    _rate_cache[code] = rate
    return rate


def _normalize_currency_code(raw: str) -> str | None:
    normalized = raw.strip().upper()
    if normalized in {"ZŁ", "ZL", "PLN"}:
        return "PLN"
    if len(normalized) == 3 and normalized.isalpha():
        return normalized
    return None


def _parse_amount(value: str) -> float | None:
    cleaned = value.replace(" ", "").replace("\u00A0", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


CURRENCY_PATTERN = re.compile(
    r"(?P<amount>[\d ]+[\d,.]*)\s*(?P<currency>PLN|ZŁ|EUR|USD|[A-Za-z]{3})",
    re.IGNORECASE,
)


def parse_currency_matches(text: str) -> list[dict]:
    matches = []
    for match in CURRENCY_PATTERN.finditer(text):
        amount = _parse_amount(match.group("amount"))
        code = _normalize_currency_code(match.group("currency"))
        if amount is None or code is None:
            continue

        rate = _get_cached_rate(code)
        matches.append(
            {
                "raw": match.group(0).strip(),
                "amount": amount,
                "currency": code,
                "rate": rate,
                "pln": round(amount * rate, 2) if rate is not None else None,
            }
        )
    return matches


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
            currency_matches = parse_currency_matches(text)
            results.append(
                {
                    "type": "tag_crop",
                    "text": text,
                    "confidence": item["confidence"],
                    "bbox": item["bbox"],
                    "currency_matches": currency_matches,
                }
            )

    # B. Jeśli nic nie wykryto -> czytaj cały obraz (fallback)
    else:
        full_text = reader.read_text(image)
        currency_matches = parse_currency_matches(full_text)
        results.append(
            {
                "type": "full_image_fallback",
                "text": full_text,
                "confidence": 0.0,
                "bbox": [],
                "currency_matches": currency_matches,
            }
        )

    # Analiza walutowa przy użyciu NBP/regexów
    return {"results": results}