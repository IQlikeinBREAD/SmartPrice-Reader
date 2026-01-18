import customtkinter as ctk
import time
import threading
from database import create_tables, add_price, get_prices

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class OCRApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Podstawowe ustawienia okna
        self.title("OCR Cenówek - Analizator Ceny")
        self.geometry("1000x600")
        self.minsize(800, 500)

        # 1. Inicjalizacja bazy (tworzenie tabel, jeśli nie istnieją)
        create_tables()

        # 2. Wyświetlenie ekranu ładowania (Splash Screen)
        self.show_loading_screen()

        # 3. Uruchomienie logiki pobierania danych w tle (osobny wątek)
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
        self.progress_bar.set(0)  # Zaczynamy od pustego paska

    def initialize_data(self):
        min_duration = 2.0  # Minimalny czas trwania ekranu ładowania
        steps = 40  # Liczba kroków paska (dla płynności)
        delay = min_duration / steps

        print("Rozpoczęto pobieranie kursów walut...")

        # --- MIEJSCE NA REALNE POBIERANIE Z API ---
        # Tutaj w przyszłości wywołasz np. fetch_currency_from_nbp()
        time.sleep(0.5)  # Symulacja krótkiego zapytania sieciowego
        # ------------------------------------------

        # Płynne dopełnienie paska do końca
        for i in range(steps):
            time.sleep(delay)
            current_progress = (i + 1) / steps
            # Aktualizacja UI musi odbywać się przez .after
            self.after(0, lambda val=current_progress: self.progress_bar.set(val))

        print("Pobieranie zakończone. Uruchamiam interfejs.")

        # Po zakończeniu paska, przełączamy na główny interfejs
        self.after(100, self.launch_main_ui)

    def launch_main_ui(self):
        """Usuwa ekran ładowania i buduje docelowy układ aplikacji."""
        if hasattr(self, 'loading_frame'):
            self.loading_frame.destroy()

        self.create_layout()

    def create_layout(self):
        """Główny układ aplikacji po załadowaniu."""
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Lewa strona: Ramka pod podgląd kamery
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

        # Prawa strona: Ramka pod wyniki OCR i dane
        self.frame_data = ctk.CTkFrame(self)
        self.frame_data.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        self.frame_data.grid_rowconfigure(1, weight=1)
        self.frame_data.grid_columnconfigure(0, weight=1)

        data_label = ctk.CTkLabel(
            self.frame_data,
            text="Wyniki OCR i Tabela Walut",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        data_label.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="nw")

        self.text_results = ctk.CTkTextbox(self.frame_data, width=300)
        self.text_results.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # Przykładowy wpis do bazy na start (testowy)
        add_price("Przykładowy Produkt", 19.99, "PLN", 4.65, "EUR")
        self.text_results.insert("0.0", "System gotowy...\nBaza danych zaktualizowana.")


if __name__ == "__main__":
    app = OCRApp()
    app.mainloop()