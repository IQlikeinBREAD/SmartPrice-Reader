import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from main import _fetch_exchange_rates

from services.detector import PriceTagDetector
from services.reader import PriceReader
from utils.text_utils import clean_price

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class OCRApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Price Reader - Moduł OCR")
        self.geometry("1200x700")

        self.detector = None
        self.reader = None
        self.camera = None
        self.current_frame = None
        self.is_running = True

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.create_layout()
        self.load_models()
        self.start_camera()

    def load_models(self):
        try:
            self.status_label.configure(text="Ładowanie modeli...", text_color="orange")
            self.update()

            self.detector = PriceTagDetector(model_path="models/yolo/custom_price_v1.pt")
            self.reader = PriceReader(use_gpu=True)

            self.status_label.configure(text="System gotowy.", text_color="green")
            print("Modele załadowane")
        except Exception as e:
            self.status_label.configure(text=f"Błąd modeli: {e}", text_color="red")
            print(e)

    def create_layout(self):
        self.frame_camera = ctk.CTkFrame(self)
        self.frame_camera.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.camera_label = ctk.CTkLabel(self.frame_camera, text="Uruchamianie kamery...")
        self.camera_label.pack(expand=True, fill="both")

        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(self.frame_controls, text="PANEL OCR", font=("Arial", 20, "bold")).pack(pady=20)

        self.status_label = ctk.CTkLabel(self.frame_controls, text="Ładowanie modeli...", text_color="orange")
        self.status_label.pack(pady=10)

        self.btn_scan = ctk.CTkButton(
            self.frame_controls,
            text="📸 SKANUJ CENĘ",
            command=self.process_current_frame,
            height=60,
            font=("Arial", 18, "bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.btn_scan.pack(pady=20, padx=20, fill="x")

        self.btn_show_last = ctk.CTkButton(
            self.frame_controls,
            text="Pokaż ostatni obraz",
            command=self.show_last_image,
            height=50,
            font=("Arial", 14, "bold"),
            fg_color="blue",
            hover_color="darkblue"
        )
        self.btn_show_last.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(self.frame_controls, text="Wykryta Cena (PLN):").pack(anchor="w", padx=20)
        self.result_box = ctk.CTkEntry(self.frame_controls, font=("Arial", 30, "bold"), justify="center")
        self.result_box.pack(pady=5, padx=20, fill="x")

        ctk.CTkLabel(self.frame_controls, text="Cena w innych walutach:").pack(anchor="w", padx=20, pady=(20, 0))
        self.currency_box = ctk.CTkTextbox(self.frame_controls, height=120)
        self.currency_box.pack(pady=5, padx=20, fill="x")
        self.currency_box.configure(state="disabled")

        ctk.CTkLabel(self.frame_controls, text="Szczegóły OCR:").pack(anchor="w", padx=20, pady=(20, 0))
        self.log_box = ctk.CTkTextbox(self.frame_controls, height=150)
        self.log_box.pack(pady=5, padx=20, fill="x")

    def start_camera(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera_label.configure(text="Brak dostępu do kamery")
            return
        self.update_camera_feed()

    def update_camera_feed(self):
        if not self.is_running:
            return
        ret, frame = self.camera.read()
        if ret:
            self.current_frame = frame.copy()

            frame_display = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(700, 500))
            self.camera_label.configure(image=img_tk, text="")
            self.camera_label.image = img_tk

        self.after(10, self.update_camera_feed)

    def process_current_frame(self):
        if self.detector is None or self.reader is None:
            self.status_label.configure(text="Modele jeszcze się ładują!", text_color="red")
            return
        if self.current_frame is None:
            return

        self.status_label.configure(text="Analizowanie...", text_color="blue")
        self.update()

        frame = self.current_frame.copy()

        bboxes = self.detector.detect(frame)
        if not bboxes:
            self.status_label.configure(text="Nie znaleziono cenówki", text_color="red")
            self.log_box.delete("1.0", "end")
            self.log_box.insert("end", "Brak detekcji YOLO")
            return

        x1, y1, x2, y2 = bboxes[0]
        crop = frame[y1:y2, x1:x2]

        raw_texts = self.reader.read_text(crop)
        self.log_box.delete("1.0", "end")
        self.log_box.insert("end", str(raw_texts))

        price = clean_price(raw_texts)

        EXCHANGE_RATES = _fetch_exchange_rates()

        if price:
            self.result_box.delete(0, "end")
            self.result_box.insert(0, str(price))
            self.status_label.configure(text="Sukces!", text_color="green")
            print(f"PRZEKAZANO DO API WALUTOWEGO: {price}")

            self.currency_box.configure(state="normal")
            self.currency_box.delete("1.0", "end")
            for curr, rate in EXCHANGE_RATES.items():
                converted = float(price) / rate
                self.currency_box.insert("end", f"{curr}: {converted:.2f}\n")
            self.currency_box.configure(state="disabled")

            try:
                from database import add_scan_to_db
                success, encoded_image = cv2.imencode('.png', crop)
                if success:
                    image_bytes = encoded_image.tobytes()
                    add_scan_to_db(name="Nieznany", price=float(price), image_source=image_bytes)
                else:
                    print("[DATABASE ERROR] Nie udało się zakodować obrazu")
            except Exception as e:
                print("[DATABASE ERROR] Nie udało się zapisać do bazy:", e)
        else:
            self.result_box.delete(0, "end")
            self.result_box.insert(0, "???")
            self.status_label.configure(text="Brak poprawnej ceny", text_color="orange")

    def show_last_image(self):
        try:
            from database import get_last_scan
            result = get_last_scan()
            if result is None:
                print("[DATABASE] Brak zapisanych skanów")
                return
            product_name, image_bytes = result

            import io
            import tkinter as tk

            image = Image.open(io.BytesIO(image_bytes))
            image.thumbnail((800, 600))

            win = tk.Toplevel(self)
            win.title(f"Ostatni skan: {product_name}")

            img_tk = ImageTk.PhotoImage(image)
            label = ctk.CTkLabel(win, text="", image=img_tk)
            label.image = img_tk
            label.pack()

        except Exception as e:
            print("[DATABASE ERROR] Nie udało się pobrać ostatniego obrazu:", e)

    def on_closing(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.destroy()


if __name__ == "__main__":
    app = OCRApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
