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

    @staticmethod
    def convert_pln_to_currency(amount_pln: float, target_currency: str) -> float:
        """
        Przelicza kwotę w PLN na docelową walutę (GBP, EUR, USD, CHF).
        """
        code = target_currency.upper()
        if code == "PLN":
            return amount_pln

        rate = NBPService.get_exchange_rate(code)
        if rate:
            # Kurs z NBP to ile PLN za 1 jednostkę waluty obcej
            # Więc aby przeliczyć PLN na walutę obcą: PLN / kurs
            return round(amount_pln / rate, 2)
        return None

    @staticmethod
    def convert_to_multiple_currencies(amount_pln: float, currencies: list = None) -> dict:
        """
        Przelicza kwotę w PLN na wiele walut jednocześnie.
        Domyślnie przelicza na GBP, EUR, USD, CHF.
        """
        if currencies is None:
            currencies = ["GBP", "EUR", "USD", "CHF"]

        results = {"PLN": amount_pln}

        for currency in currencies:
            converted = NBPService.convert_pln_to_currency(amount_pln, currency)
            if converted is not None:
                results[currency] = converted
            else:
                results[currency] = "Błąd"

        return results