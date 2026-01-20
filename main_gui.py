import customtkinter as ctk
import time
import threading
import io
from PIL import Image
import database
from database import check_db_connection, create_tables

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class OCRApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OCR Cenówek - Analizator Ceny")
        self.geometry("1000x600")
        self.minsize(800, 500)

        is_connected, message = check_db_connection()

        if not is_connected:
            self.show_loading_screen()
            self.after(0, lambda: self.loading_label.configure(text=message, text_color="red"))
            return

        database.create_tables()
        self.show_loading_screen()
        threading.Thread(target=self.initialize_data, daemon=True).start()

    def show_loading_screen(self):
        self.loading_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.loading_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.loading_label = ctk.CTkLabel(
            self.loading_frame,
            text="Trwa pobieranie kursów walut...",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.loading_label.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(self.loading_frame, orientation="horizontal", width=300)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

    def initialize_data(self):
        min_duration = 2.0
        steps = 40
        delay = min_duration / steps

        time.sleep(0.5)

        for i in range(steps):
            time.sleep(delay)
            current_progress = (i + 1) / steps
            self.after(0, lambda val=current_progress: self.progress_bar.set(val))

        self.after(100, self.launch_main_ui)

    def launch_main_ui(self):
        if hasattr(self, 'loading_frame'):
            self.loading_frame.destroy()
        self.create_layout()

    def create_layout(self):
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        #Miejsce na kamere
        self.frame_camera_feed = ctk.CTkFrame(self)
        self.frame_camera_feed.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.frame_camera_feed.grid_columnconfigure(0, weight=1)
        self.frame_camera_feed.grid_rowconfigure(0, weight=1)

        self.camera_label = ctk.CTkLabel(
            self.frame_camera_feed,
            text="PODGLĄD KAMERY\n(Miejsce na obraz z OpenCV)",
            text_color="gray",
            fg_color=("gray80", "gray20"),
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.camera_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        #Prawy layout
        self.frame_data = ctk.CTkFrame(self)
        self.frame_data.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        self.frame_data.grid_columnconfigure(0, weight=1)
        self.frame_data.grid_rowconfigure(2, weight=1)

        data_label = ctk.CTkLabel(
            self.frame_data,
            text="Wyniki OCR i Tabela Walut",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        data_label.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="nw")

        #Lista walut do wyboru
        self.currency_var = ctk.StringVar(value="Waluta 1")

        currency_frame = ctk.CTkFrame(self.frame_data, fg_color="transparent")
        currency_frame.grid(row=1, column=0, padx=20, pady=(5, 5), sticky="ew")
        currency_frame.grid_columnconfigure(0, weight=1)

        self.currency_dropdown = ctk.CTkOptionMenu(
            currency_frame,
            variable=self.currency_var,
            values=[f"Waluta {i}" for i in range(1, 11)]
        )
        self.currency_dropdown.grid(row=0, column=0, padx=(0, 10), sticky="ew")

        self.convert_button = ctk.CTkButton(
            currency_frame,
            text="Konwertuj",
            command=self.convert_currency,
            width=120
        )
        self.convert_button.grid(row=0, column=1)

        #Miejsce do wyswietlania wynikow
        self.text_results = ctk.CTkTextbox(self.frame_data, width=300)
        self.text_results.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

        #Przycisk ostatniego skanu
        self.btn_history = ctk.CTkButton(
            self.frame_data,
            text="Historia",
            command=self.show_history_window,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        self.btn_history.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")

    def convert_currency(self):
        selected_currency = self.currency_var.get()
        self.text_results.delete("1.0", "end")
        self.text_results.insert("end", f"Wybrana waluta: {selected_currency}")

    def show_history_window(self):
        result = database.get_last_scan()

        if not result:
            self.text_results.insert("end", "\n[!] Brak wpisów w historii.")
            return

        product_name, image_data = result

        history_win = ctk.CTkToplevel(self)
        history_win.title("Ostatni skan: " + str(product_name))
        history_win.geometry("500x550")
        history_win.attributes('-topmost', True)

        try:
            img = Image.open(io.BytesIO(image_data))
            orig_w, orig_h = img.size
            scale = min(450 / orig_w, 450 / orig_h)
            new_size = (int(orig_w * scale), int(orig_h * scale))

            img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=new_size)

            img_label = ctk.CTkLabel(history_win, image=img_ctk, text="")
            img_label.pack(pady=10)

            name_label = ctk.CTkLabel(
                history_win,
                text="Produkt: " + str(product_name),
                font=("Arial", 16, "bold")
            )
            name_label.pack(pady=10)

        except Exception as e:
            error_label = ctk.CTkLabel(
                history_win,
                text="Błąd ładowania obrazu:\n" + str(e)
            )
            error_label.pack(pady=20)


if __name__ == "__main__":
    app = OCRApp()
    app.mainloop()
