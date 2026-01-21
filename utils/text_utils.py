import re

def clean_price(text_list: list) -> float:
    if not text_list:
        return None

    clean_items = []
    
    for text in text_list:
        t_raw = text.strip()
        
        if '-' in t_raw or '/' in t_raw or ':' in t_raw:
            continue

        if len(t_raw) > 8:
            continue

        t_clean = t_raw.lower().replace('zł', '').replace('zl', '').replace('gr', '').replace(',', '.').replace(' ', '')

        if re.search(r'\d', t_clean):
            clean_items.append(t_clean)

    candidates = []

    for i in range(len(clean_items)):
        current = clean_items[i]

        if re.fullmatch(r'\d+\.\d{2}', current):
            val = float(current)
            candidates.append((val, 100))
            continue

        for offset in [1, 2]: 
            if i + offset < len(clean_items):
                neighbor = clean_items[i + offset]

                dist_penalty = 0 if offset == 1 else -20
                
                price = None

                if re.fullmatch(r'\d{1,4}', current) and re.fullmatch(r'\d{2}', neighbor):
                    price = float(f"{current}.{neighbor}")

                elif re.fullmatch(r'\d{2}', current) and re.fullmatch(r'\d{1,4}', neighbor):
                    price = float(f"{neighbor}.{current}")

                if price is not None:
                    score = 50 + dist_penalty

                    if f"{price:.2f}".endswith(('99', '98', '49', '50')):
                        score += 30
                    
                    if price in [2024.0, 2025.0, 2026.0, 2027.0]:
                        score -= 1000

                    if price > 2000:
                        score -= 50
                        
                    candidates.append((price, score))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    
    best_price, best_score = candidates[0]
    
    return best_price