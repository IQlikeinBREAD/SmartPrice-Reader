import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time

# Import Twoich modułów
from services.detector import PriceTagDetector
from services.reader import PriceReader
from utils.text_utils import clean_price

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class OCRApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Price Reader - Moduł OCR")
        self.geometry("1100x700")

        # 1. Inicjalizacja Twoich modeli (w tle, żeby GUI nie zamarzło)
        self.detector = None
        self.reader = None
        self.camera = None
        self.is_running = True

        # Ładowanie modeli w osobnym wątku
        threading.Thread(target=self.load_models, daemon=True).start()

        # UI Layout
        self.create_layout()

        # Start kamery
        self.start_camera()

    def load_models(self):
        # Tutaj ładujemy Twoje ciężkie modele AI
        try:
            # UPEWNIJ SIĘ, ŻE ŚCIEŻKA DO MODELU JEST DOBRA!
            self.detector = PriceTagDetector(model_path='models/yolo/custom_price_v1.pt')
            self.reader = PriceReader(use_gpu=False)
            print("✅ Modele załadowane pomyślnie!")

            # Zamiast bezpośrednio konfigurować, używamy .after()
            self.after(0, lambda: self.status_label.configure(
                text="System gotowy. Wyceluj w cenę.",
                text_color="green"
            ))

        except Exception as e:
            print(f"Błąd ładowania modeli: {e}")
            # Tu również używamy .after() dla błędu
            self.after(0, lambda: self.status_label.configure(
                text=f"Błąd modeli: {e}",
                text_color="red"
            ))

    def create_layout(self):
        self.grid_columnconfigure(0, weight=3)  # Lewa strona (Kamera) szersza
        self.grid_columnconfigure(1, weight=1)  # Prawa strona (Panel)
        self.grid_rowconfigure(0, weight=1)

        # --- LEWA STRONA: KAMERA ---
        self.frame_camera = ctk.CTkFrame(self)
        self.frame_camera.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.camera_label = ctk.CTkLabel(self.frame_camera, text="Uruchamianie kamery...", text_color="white")
        self.camera_label.pack(expand=True, fill="both", padx=5, pady=5)

        # --- PRAWA STRONA: PANEL STEROWANIA ---
        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Nagłówek
        ctk.CTkLabel(self.frame_controls, text="PANEL OCR", font=("Arial", 20, "bold")).pack(pady=20)

        # Status
        self.status_label = ctk.CTkLabel(self.frame_controls, text="Ładowanie modeli...", text_color="orange")
        self.status_label.pack(pady=10)

        # Przycisk SKANUJ (Wielki i widoczny)
        self.btn_scan = ctk.CTkButton(
            self.frame_controls,
            text="📸 SKANUJ CENĘ",
            command=self.process_current_frame,
            height=60,
            font=("Arial", 18, "bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.btn_scan.pack(pady=40, padx=20, fill="x")

        # Pole wyników (Tu pojawi się cena dla kolegi)
        ctk.CTkLabel(self.frame_controls, text="Wykryta Cena (PLN):").pack(anchor="w", padx=20)
        self.result_box = ctk.CTkEntry(self.frame_controls, font=("Arial", 30, "bold"), justify="center")
        self.result_box.pack(pady=5, padx=20, fill="x")

        # Logi tekstowe (co widzi OCR)
        ctk.CTkLabel(self.frame_controls, text="Szczegóły (Raw Text):").pack(anchor="w", padx=20, pady=(20, 0))
        self.log_box = ctk.CTkTextbox(self.frame_controls, height=150)
        self.log_box.pack(pady=5, padx=20, fill="x")

    def start_camera(self):
        # Otwieramy kamerę 0 (domyślna w laptopie/USB)
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera_label.configure(text="Błąd: Nie wykryto kamery!")
            return

        # Rozpoczynamy pętlę odświeżania obrazu
        self.update_camera_feed()

    def update_camera_feed(self):
        if self.is_running:
            ret, frame = self.camera.read()
            if ret:
                # 1. Konwersja kolorów BGR (OpenCV) -> RGB (Tkinter)
                self.current_frame = frame  # Zapisujemy klatkę do obróbki
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 2. Tworzenie obrazka dla GUI
                img = Image.fromarray(frame_rgb)

                # Skalowanie do okna (opcjonalne, żeby nie rozpychało GUI)
                img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))

                self.camera_label.configure(image=img_tk, text="")

            # Odśwież co 10 milisekund
            self.after(10, self.update_camera_feed)

    def process_current_frame(self):
        """
        To jest serce programu. Uruchamia się po kliknięciu przycisku.
        1. Bierze klatkę.
        2. YOLO wycina cenówkę.
        3. OCR czyta.
        4. Zwraca cenę.
        """
        if self.detector is None or self.reader is None:
            self.status_label.configure(text="Modele jeszcze nie gotowe!", text_color="red")
            return

        if not hasattr(self, 'current_frame'):
            return

        frame = self.current_frame.copy()
        self.status_label.configure(text="Analizowanie...", text_color="blue")
        self.update()  # Wymuś odświeżenie UI

        # 1. DETEKCJA (YOLO)
        bboxes = self.detector.detect(frame)

        if not bboxes:
            self.status_label.configure(text="❌ Nie znaleziono cenówki!", text_color="red")
            self.log_box.delete("1.0", "end")
            self.log_box.insert("end", "Brak detekcji YOLO.")
            return

        # Bierzemy pierwszą znalezioną cenówkę (zakładamy, że celujesz w jedną)
        x1, y1, x2, y2 = bboxes[0]

        # Rysujemy ramkę na podglądzie (opcjonalnie, dla efektu)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 2. WYCIĘCIE (CROP)
        crop = frame[y1:y2, x1:x2]

        # 3. CZYTANIE (OCR)
        raw_texts = self.reader.read_text(crop)

        # Wyświetl surowe teksty w logach
        self.log_box.delete("1.0", "end")
        self.log_box.insert("end", f"OCR widzi: {raw_texts}")

        # 4. OCZYSZCZANIE (Logika 3 49 -> 3.49)
        price = clean_price(raw_texts)

        if price:
            self.result_box.delete(0, "end")
            self.result_box.insert(0, str(price))
            self.status_label.configure(text="✅ Sukces!", text_color="green")

            # TU PRZEKAZUJESZ WYNIK KOLEDZE:
            print(f"--- PRZEKAZANO DO API WALUTOWEGO: {price} ---")
        else:
            self.status_label.configure(text="⚠️ Widzę cenówkę, ale nie widzę ceny", text_color="orange")
            self.result_box.delete(0, "end")
            self.result_box.insert(0, "???")

    def on_closing(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.destroy()


if __name__ == "__main__":
    app = OCRApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()