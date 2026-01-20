import cv2
import sys
import os

# Importy Twoich modułów
from services.detector import PriceTagDetector
from services.reader import PriceReader
from utils.text_utils import clean_price

# --- KONFIGURACJA ---
# Tu wpisz nazwę zdjęcia, które chcesz przetestować (musi być w folderze projektu lub podaj pełną ścieżkę)
IMAGE_PATH = "D:\Програмування\SmartPrice-Reader\photo_6_2026-01-20_21-20-47.jpg" 
# Upewnij się, że masz tu swój wytrenowany model!
MODEL_PATH = "models/yolo/custom_price_v1.pt" 

def main():
    # 1. Sprawdzenie czy plik istnieje
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ BŁĄD: Nie znaleziono pliku: {IMAGE_PATH}")
        print("Upewnij się, że ścieżka do zdjęcia jest poprawna.")
        return

    print("--- 1. ŁADOWANIE MODELI ---")
    try:
        detector = PriceTagDetector(model_path=MODEL_PATH, conf=0.5)
        # Jeśli masz GPU, zostaw use_gpu=True. Jeśli wywala błąd pamięci, zmień na False.
        reader = PriceReader(use_gpu=True) 
    except Exception as e:
        print(f"❌ Błąd ładowania modeli: {e}")
        return

    print(f"--- 2. ANALIZA ZDJĘCIA: {IMAGE_PATH} ---")
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("❌ Błąd: cv2 nie mógł otworzyć zdjęcia.")
        return

    # A. DETEKCJA (YOLO)
    bboxes = detector.detect(frame)
    print(f"🔍 YOLO znalazło obiektów: {len(bboxes)}")

    if not bboxes:
        print("⚠️ Nie wykryto żadnej cenówki na zdjęciu.")
        # Pokażmy chociaż oryginalne zdjęcie
        cv2.imshow("Test Result", frame)
        cv2.waitKey(0)
        return

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        print(f"\n--- OBIEKT NR {i+1} ---")
        print(f"   Współrzędne: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # B. WYCIĘCIE (CROP)
        # Zabezpieczenie przed wyjściem poza obraz
        h, w, _ = frame.shape
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        # C. OCR (EasyOCR)
        raw_texts = reader.read_text(crop)
        print(f"   👁️  EasyOCR przeczytał: {raw_texts}")

        # D. LOGIKA CENY (Regex)
        final_price = clean_price(raw_texts)
        
        if final_price:
            print(f"   💰 ZIDENTYFIKOWANA CENA: {final_price} PLN")
            
            # RYSOWANIE WYNIKU
            # Zielona ramka
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Tło pod napis (żeby było czytelnie)
            label = f"CENA: {final_price} zl"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + w_text, y1), (0, 255, 0), -1)
            
            # Napis
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else:
            print("   ⚠️ Nie udało się wyciągnąć liczby z tekstu.")
            # Czerwona ramka dla błędu
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 3. WYŚWIETLENIE
    print("\n--- KONIEC ---")
    print("Naciśnij dowolny klawisz na otwartym oknie, aby zamknąć...")
    
    # Skalowanie okna, jeśli zdjęcie jest ogromne (np. z telefonu)
    screen_res = 1280, 720
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    
    if scale < 1:
        window_width = int(frame.shape[1] * scale)
        window_height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (window_width, window_height))

    cv2.imshow("TEST JEDNEGO ZDJECIA", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()