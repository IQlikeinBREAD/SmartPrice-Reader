import sqlite3
import time

DB_FILE = "prices.db"
#Zalozenie do projektu- kursy walut z API beda pobierać sie i aktualizować w bazie danych przy kazdym uruchomieniu aplikacji

def create_tables():
    """
    Tworzy tabelę prices, jeśli nie istnieje.
    Kolumny:
        - id: automatyczne ID
        - product_name: nazwa produktu/cenówki
        - price: odczytana cena
        - currency: waluta odczytana
        - converted_price: przeliczona cena
        - converted_currency: waluta docelowa
        - timestamp: czas dodania rekordu
    """
    connection = sqlite3.connect(DB_FILE)
    connection.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT,
            price REAL,
            currency TEXT,
            converted_price REAL,
            converted_currency TEXT,
            timestamp TEXT
        )
    ''')
    connection.commit()
    connection.close()


def add_price(product_name: str, price: float, currency: str, converted_price: float, converted_currency: str):
    """
    Dodaje nowy rekord do tabeli prices.
    """
    connection = sqlite3.connect(DB_FILE)
    connection.execute(
        "INSERT INTO prices (product_name, price, currency, converted_price, converted_currency, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (product_name, price, currency, converted_price, converted_currency, time.strftime("%Y-%m-%d %H:%M:%S"))
    )
    connection.commit()
    connection.close()


def get_prices(limit: int = 100):
    """
    Zwraca listę rekordów z tabeli prices, domyślnie ostatnie 100 rekordów.
    """
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.execute("SELECT * FROM prices ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    connection.close()
    return rows
