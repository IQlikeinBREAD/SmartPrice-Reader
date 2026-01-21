import mysql.connector
from mysql.connector import errorcode

def get_connection():
    #Tworzy połączenie z lokalną instancją bazy MySQL o nazwie ocr_cen
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",
        database="ocr_cen"
    )

def check_db_connection():
    #Weryfikuje dostępność serwera MySQL i zwraca status połączenia.
    try:
        conn = get_connection()
        conn.close()
        print("[OK] Połączono z serwerem MySQL.")
        return True, "Połączono"
    except mysql.connector.Error as err:
        if err.errno == errorcode.CR_CONN_HOST_ERROR:
            msg = "BŁĄD: Nie można połączyć się z serwerem. Czy XAMPP (MySQL) jest włączony?"
        elif err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            msg = "BŁĄD: Zły użytkownik lub hasło."
        else:
            msg = "BŁĄD bazy danych: " + str(err)
        print(msg)
        return False, msg

def create_tables():
    #Tworzy tabelę scanned_prices, jeśli nie istnieje ona w bazie danych.
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # @formatter:off
        query = """
                CREATE TABLE IF NOT EXISTS scanned_prices (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    product_name VARCHAR(255) DEFAULT 'Nieznany',
                    price_pln DECIMAL(10, 2),
                    image_data LONGBLOB,
                    scan_date DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
        # @formatter:on
        cursor.execute(query)

        conn.commit()
        cursor.close()
        conn.close()
        print("Baza gotowa!")
    except mysql.connector.Error as err:
        print("Błąd: " + str(err))

def add_scan_to_db(name, price, image_source):
    #Konwertuje obraz na bajty i zapisuje go wraz z metadanymi do bazy.
    try:
        if isinstance(image_source, str):
            with open(image_source, 'rb') as file:
                image_bytes = file.read()
        else:
            image_bytes = image_source

        conn = get_connection()
        cursor = conn.cursor()

        query = """
                INSERT INTO scanned_prices (product_name, price_pln, image_data)
                VALUES (%s, %s, %s)
                """

        cursor.execute(query, (name, price, image_bytes))
        conn.commit()

        print("[DATABASE] Pomyślnie zapisano produkt: " + str(name))
        return True

    except Exception as e:
        print("[DATABASE ERROR] Błąd zapisu: " + str(e))
        return False
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def get_last_scan():
    #Pobiera z bazy nazwę oraz dane binarne zdjęcia z ostatniego rekordu.
    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = "SELECT product_name, image_data FROM scanned_prices ORDER BY id DESC LIMIT 1"
        cursor.execute(query)
        result = cursor.fetchone()

        return result
    except Exception as e:
        print("[DATABASE ERROR] Błąd pobierania: " + str(e))
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()