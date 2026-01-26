# SmartPrice-Reader

System do automatycznego odczytu cen z metek cenowych za pomocą technologii YOLO i OCR (EasyOCR). Projekt umożliwia skanowanie cen z kamery w czasie rzeczywistym, konwersję walut przez API NBP oraz zapis danych do bazy MySQL.

---



##  Funkcjonalności

- **Detekcja metek cenowych** - wykorzystanie modelu YOLO11 do lokalizacji cenówek na obrazie
- **OCR (Optical Character Recognition)** - odczyt tekstu z wykrytych obszarów za pomocą EasyOCR
- **Czyszczenie i parsowanie cen** - inteligentna ekstrakcja wartości numerycznych z rozpoznanego tekstu
- **Integracja z API NBP** - automatyczna konwersja PLN na waluty obce (EUR, USD, GBP, CHF)
- **Interfejs GUI** - aplikacja desktopowa z podglądem kamery w czasie rzeczywistym
- **REST API** - backend FastAPI do skanowania obrazów
- **Zapis do bazy danych** - archiwizacja zeskanowanych cen wraz z obrazami w MySQL

---

##  Struktura projektu

```
SmartPrice-Reader/
│
├── main.py                      # REST API (FastAPI) do skanowania cen
├── main_gui.py                  # Aplikacja GUI z kamerą (CustomTkinter)
├── database.py                  # Moduł obsługi bazy danych MySQL
├── requirements.txt             # Lista zależności Python
├── prices.db                    # Plik bazy SQLite (opcjonalny)
│
├── config/
│   └── yolo_config.yaml         # Konfiguracja modelu YOLO
│
├── models/
│   └── yolo/
│       ├── custom_price_v1.pt   # Wytrenowany model YOLO do detekcji cen
│       └── train_yolo.py        # Skrypt do trenowania modelu
│
├── services/
│   ├── detector.py              # Klasa PriceTagDetector (detekcja YOLO)
│   ├── reader.py                # Klasa PriceReader (OCR - EasyOCR)
│   └── currency.py              # Klasa NBPService (konwersja walut)
│
└── utils/
    └── text_utils.py            # Funkcja clean_price() do parsowania cen
```

---

##  Instalacja

### Wymagania wstępne
- Python 3.8 lub nowszy
- XAMPP lub inny serwer MySQL (dla funkcji bazodanowych)
- Kamera internetowa (dla aplikacji GUI)

### Instalacja zależności

Zainstaluj wszystkie wymagane pakiety jednym poleceniem:

```bash
pip install -r requirements.txt
```

**Lista głównych pakietów:**
- `fastapi` - framework do REST API
- `uvicorn` - serwer ASGI dla FastAPI
- `ultralytics` - biblioteka YOLO11 do detekcji obiektów
- `easyocr` - silnik OCR do rozpoznawania tekstu
- `customtkinter` - nowoczesny framework GUI
- `opencv-python` (cv2) - przetwarzanie obrazów
- `pillow` - operacje na obrazach
- `numpy` - obliczenia numeryczne
- `requests` / `httpx` - zapytania HTTP do API NBP
- `mysql-connector-python` - łącznik z bazą MySQL


---

##  Konfiguracja bazy danych

1. Uruchom **XAMPP** i włącz moduł **MySQL**
2. Otwórz **phpMyAdmin** (http://localhost/phpmyadmin)
3. Utwórz bazę danych o nazwie **`ocr_cen`**
4. Tabela `scanned_prices` zostanie utworzona automatycznie przy pierwszym uruchomieniu aplikacji

---

##  Uruchomienie projektu

### Wariant 1: Aplikacja GUI (zalecane dla użytkowników)

Uruchom aplikację z interfejsem graficznym:

```bash
python main_gui.py
```

**Funkcje GUI:**
- Podgląd kamery w czasie rzeczywistym
- Przycisk "Skanuj cenę" - analizuje aktualną klatkę
- Wyświetlanie rozpoznanej ceny w PLN
- Konwersja na waluty obce (EUR, USD, GBP, CHF)
- Automatyczny zapis do bazy danych
- Przycisk "Pokaż ostatni obraz" - wyświetla ostatni zeskanowany obraz z bazy

### Wariant 2: REST API (dla integracji z innymi systemami)

Uruchom serwer FastAPI:

```bash
uvicorn main:app --reload
```

API będzie dostępne pod adresem: **http://127.0.0.1:8000**

**Endpoint:**
- `POST /scan` - prześlij obraz (multipart/form-data) z metką cenową

**Przykład użycia (curl):**

```bash
curl -X POST "http://127.0.0.1:8000/scan" -F "file=@zdjecie_ceny.jpg"
```

**Dokumentacja API:**
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

##  Architektura i korelacje

### Przepływ danych w aplikacji GUI (`main_gui.py`):

1. **Kamera** → `cv2.VideoCapture(0)` → przechwytuje klatki wideo
2. **Przycisk "Skanuj"** → wywołuje `process_current_frame()`
3. **Detekcja** → `PriceTagDetector.detect()` → YOLO lokalizuje cenówkę → zwraca bbox `[x1, y1, x2, y2]`
4. **Kadrowanie** → wycięcie obszaru ROI (Region of Interest)
5. **OCR** → `PriceReader.read_text()` → EasyOCR rozpoznaje tekst → zwraca listę stringów
6. **Parsowanie** → `clean_price()` → wyciąga wartość numeryczną z tekstu
7. **Konwersja walut** → `_fetch_exchange_rates()` → pobiera kursy z API NBP
8. **Zapis** → `add_scan_to_db()` → zapisuje cenę + obraz do MySQL

### Przepływ danych w API (`main.py`):

```
POST /scan → _decode_image() → _build_results() → _parse_currency() → JSON response
                   ↓                    ↓                   ↓
              ndarray(frame)    PriceTagDetector    NBP API (httpx)
                                PriceReader
                                clean_price()
```

### Kluczowe klasy i funkcje:

#### 1. **PriceTagDetector** (`services/detector.py`)
- **Rola:** Lokalizacja metek cenowych na obrazie
- **Model:** YOLO11 (`custom_price_v1.pt`)
- **Metoda:** `detect(frame)` → zwraca listę bounding boxów `[[x1,y1,x2,y2], ...]`

#### 2. **PriceReader** (`services/reader.py`)
- **Rola:** Rozpoznawanie tekstu z obrazu (OCR)
- **Biblioteka:** EasyOCR (języki: polski, angielski)
- **Metoda:** `read_text(image_crop)` → zwraca listę rozpoznanych tekstów
- **Parametr:** `use_gpu=True/False` - przyspieszenie GPU (opcjonalne)

#### 3. **clean_price()** (`utils/text_utils.py`)
- **Rola:** Czyszczenie i ekstrakcja ceny z surowego tekstu OCR
- **Wejście:** lista stringów (np. `["12", "99", "zł"]`)
- **Wyjście:** liczba zmiennoprzecinkowa (np. `12.99`)
- **Logika:**
  - Usuwa jednostki (`zł`, `gr`)
  - Łączy rozdzielone cyfry (np. "12" + "99" → 12.99)
  - Preferuje ceny kończące się na `.99`, `.49` itp.
  - Filtruje fałszywe wyniki (daty, numery telefonów)

#### 4. **NBPService** (`services/currency.py`)
- **Rola:** Integracja z API Narodowego Banku Polskiego
- **Metody:**
  - `get_exchange_rate(currency_code)` → pobiera aktualny kurs PLN/waluta
  - `convert_pln_to_currency(amount_pln, target_currency)` → konwersja PLN → EUR/USD/GBP/CHF

#### 5. **Database** (`database.py`)
- **Funkcje:**
  - `get_connection()` - tworzy połączenie MySQL
  - `create_tables()` - inicjalizuje schemat bazy
  - `add_scan_to_db(name, price, image_source)` - zapisuje skan
  - `get_last_scan()` - pobiera ostatni zapis z BLOB obrazu

### Zależności między modułami:

```
main_gui.py
    ├─→ services/detector.py (PriceTagDetector)
    ├─→ services/reader.py (PriceReader)
    ├─→ utils/text_utils.py (clean_price)
    ├─→ main.py (_fetch_exchange_rates)
    └─→ database.py (add_scan_to_db, get_last_scan)

main.py (FastAPI)
    ├─→ services/detector.py
    ├─→ services/reader.py
    ├─→ utils/text_utils.py
    └─→ httpx (API NBP)

services/detector.py
    └─→ ultralytics (YOLO)

services/reader.py
    └─→ easyocr

database.py
    └─→ mysql.connector
```


