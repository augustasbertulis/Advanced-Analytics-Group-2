# data_preprocessing_gamescsv.py
# Clean CSV with broken JSON in price_overview; keep only final price (cents/100) and currency.
# Enforce useful dtypes and print a brief report (row count + dtypes).

import csv
import re
import pandas as pd
from pathlib import Path
from math import nan

# --- PATHS ---
SRC = Path(r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/raw data/games.csv")
OUT = Path(r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/clean/games_clean.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# --- HELPERS ---

def clean_null(x):
    """Normalize '\\N' and empty strings to None."""
    if x is None:
        return None
    x = str(x).strip()
    return None if x in {"\\N", ""} else x

def parse_price(raw: str):
    """Extract 'final' (in cents) and 'currency' from a broken JSON-ish blob."""
    if not raw:
        return None, None
    s = raw.strip()
    b, e = s.find("{"), s.rfind("}")
    if b == -1 or e == -1 or e <= b:
        return None, None
    # normalize broken quoting
    s = s[b+1:e].replace('\\"', '"').replace('""', '"')

    # final in cents (int)
    m_final = re.search(r'"final"\s*:\s*(-?\d+)', s)
    final_int = int(m_final.group(1)) if m_final else None

    # currency string
    m_curr = re.search(r'"currency"\s*:\s*"([^"]*)"', s)
    currency = m_curr.group(1) if m_curr else None

    return final_int, currency

def extract_price_blob(tokens, start_idx):
    """Stitch CSV tokens until JSON braces balance; return (raw_blob, next_index_after_blob)."""
    ahead = ",".join(tokens[start_idx:])
    if "{" not in ahead:
        return None, start_idx

    buf, braces, i = [], 0, start_idx
    started = False
    while i < len(tokens):
        t = tokens[i]
        buf.append(t)
        braces += t.count("{") - t.count("}")
        if "{" in t:
            started = True
        if started and braces == 0:
            i += 1
            break
        i += 1

    return ",".join(buf), i

def report_dataframe(df: pd.DataFrame):
    print("\n--- DataFrame Report ---")
    print(f"Rows: {len(df):,}")
    print("Columns and dtypes:")
    print(df.dtypes)

# --- MAIN ---

def main():
    rows = []
    with SRC.open(newline='', encoding='utf-8') as f:
        rdr = csv.reader(f)
        next(rdr, None)  # skip first header row

        for r in rdr:
            if not r:
                continue
            # skip any accidental header lines that reappear mid-file
            if r[0].strip().lower() == "app_id":
                continue

            app_id       = clean_null(r[0])
            name         = clean_null(r[1])
            release_date = clean_null(r[2])
            is_free_raw  = clean_null(r[3])

            try:
                is_free = int(is_free_raw) if is_free_raw is not None else None
            except ValueError:
                is_free = None

            if is_free == 1:
                # Free app: no price blob; languages and type follow directly
                languages = clean_null(r[4]) if len(r) > 4 else None
                type_     = clean_null(r[5]) if len(r) > 5 else None
                price     = nan
                currency  = None
            else:
                # Paid app (or unknown): stitch price blob, then languages/type
                raw_blob, next_i = extract_price_blob(r, 4)
                cents, currency  = parse_price(raw_blob) if raw_blob else (None, None)
                price            = (cents / 100.0) if isinstance(cents, int) else nan
                languages        = clean_null(r[next_i])     if next_i     < len(r) else None
                type_            = clean_null(r[next_i + 1]) if next_i + 1 < len(r) else None

            rows.append({
                "app_id": app_id,
                "name": name,
                "release_date": release_date,
                "is_free": is_free,
                "languages": languages,
                "type": type_,
                "price_overview.final": price,
                "price_overview.currency": currency,
            })

    df = pd.DataFrame(rows).replace({"\\N": pd.NA})

    # --- Enforce useful dtypes ---
    # app_id as nullable integer (primary key friendly)
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")

    # is_free as boolean (treat NaN as False)
    df["is_free"] = pd.to_numeric(df["is_free"], errors="coerce").fillna(0).astype(bool)

    # release_date as datetime
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    # text columns as pandas string dtype
    for col in ["name", "languages", "type", "price_overview.currency"]:
        df[col] = df[col].astype("string")

    # Save cleaned file
    df.to_csv(OUT, index=False, encoding="utf-8")

    # Brief report (no data rows shown)
    report_dataframe(df)
    print(f"\nSaved cleaned file to: {OUT}")

if __name__ == "__main__":
    main()
