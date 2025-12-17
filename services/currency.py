import requests


class NBPService:
    BASE_URL = "http://api.nbp.pl/api/exchangerates/rates/a/"

    @staticmethod
    def get_exchange_rate(currency_code: str) -> float:
        """
        Pobiera aktualny kurs średni dla danej waluty (np. EUR, USD) względem PLN.
        Zwraca 1.0 dla PLN.
        """
        code = currency_code.upper()
        if code == "PLN":
            return 1.0

        try:
            url = f"{NBPService.BASE_URL}{code}/?format=json"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data["rates"][0]["mid"]
        except requests.exceptions.RequestException as e:
            print(f"Błąd podczas pobierania kursu dla {code}: {e}")
            return None

    @staticmethod
    def convert_to_pln(amount: float, currency_code: str) -> float:
        rate = NBPService.get_exchange_rate(currency_code)
        if rate:
            return round(amount * rate, 2)
        return None