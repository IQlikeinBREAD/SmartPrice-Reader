import re

def clean_price(text_list: list) -> float:
    """
    Naprawiona logika: Priorytetyzuje liczby stojące OBOK siebie.
    Rozwiązuje problem błędnego łączenia '99' z góry i '99' z dołu etykiety.
    """
    if not text_list:
        return None

    clean_items = []
    
    # --- KROK 1: CZYSZCZENIE LISTY ---
    for text in text_list:
        t_raw = text.strip()
        
        # Odrzucamy daty (np. 13-01, 28/02)
        if '-' in t_raw or '/' in t_raw or ':' in t_raw:
            continue
            
        # Odrzucamy długie kody kreskowe (>8 znaków)
        if len(t_raw) > 8:
            continue

        # Czyścimy z walut i śmieci
        t_clean = t_raw.lower().replace('zł', '').replace('zl', '').replace('gr', '').replace(',', '.').replace(' ', '')
        
        # Jeśli to liczba, dodajemy do listy
        if re.search(r'\d', t_clean):
            clean_items.append(t_clean)

    # --- KROK 2: BUDOWANIE KANDYDATÓW Z PUNKTACJĄ ---
    # candidates = [(cena, punkty), (cena, punkty)...]
    candidates = []

    for i in range(len(clean_items)):
        current = clean_items[i]
        
        # A. Samotna liczba z kropką (np. "14.99") - BARDZO SILNY KANDYDAT
        if re.fullmatch(r'\d+\.\d{2}', current):
            val = float(current)
            candidates.append((val, 100)) # 100 punktów pewności
            continue

        # B. Łączenie z sąsiadami (Szukamy groszy)
        # Sprawdzamy sąsiadów: bliskiego (+1) i dalszego (+2)
        for offset in [1, 2]: 
            if i + offset < len(clean_items):
                neighbor = clean_items[i + offset]
                
                # Ustalamy "karę" za odległość.
                # Jeśli sąsiedzi są obok siebie (offset 1) -> 0 kary.
                # Jeśli rozdzieleni (offset 2) -> -20 punktów.
                dist_penalty = 0 if offset == 1 else -20
                
                price = None
                
                # WARIANT 1: Integer (current) + Decimal (neighbor) -> np. 14 + 99
                if re.fullmatch(r'\d{1,4}', current) and re.fullmatch(r'\d{2}', neighbor):
                    price = float(f"{current}.{neighbor}")

                # WARIANT 2: Decimal (current) + Integer (neighbor) -> np. 99 + 14 (Odwrócone przez OCR)
                elif re.fullmatch(r'\d{2}', current) and re.fullmatch(r'\d{1,4}', neighbor):
                    price = float(f"{neighbor}.{current}")

                if price is not None:
                    score = 50 + dist_penalty # Bazowo 50 pkt
                    
                    # BONUSY:
                    # Cena kończy się na .99, .98, .49? To typowe dla sklepu!
                    if f"{price:.2f}".endswith(('99', '98', '49', '50')):
                        score += 30
                    
                    # KARA ZA ROK:
                    if price in [2024.0, 2025.0, 2026.0, 2027.0]:
                        score -= 1000

                    # KARA ZA NIEREALNĄ CENĘ:
                    if price > 2000: # Powyżej 2000 zł to raczej błąd (chyba że RTV/AGD)
                        score -= 50
                        
                    candidates.append((price, score))

    if not candidates:
        return None

    # --- KROK 3: WYBÓR NAJLEPSZEGO KANDYDATA ---
    # Sortujemy: najpierw po punktach (malejąco), potem po cenie (malejąco)
    # Dzięki temu 14.99 (80 pkt) wygra z 99.99 (60 pkt, bo było dalej od siebie)
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    
    best_price, best_score = candidates[0]
    
    # Debugowanie w konsoli (żebyś widział co wygrało)
    # print(f"DEBUG: Kandydaci: {candidates}") 
    
    return best_price