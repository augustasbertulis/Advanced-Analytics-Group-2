#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import csv
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

INPUT = RAW_DATA_DIR / "descriptions.csv"
OUTPUT_CSV = PROCESSED_DATA_DIR / "descriptions_clean.csv"
# INPUT = "data/raw data/steam-insights-main/descriptions.csv"
# OUTPUT_CSV = "data/clean data/descriptions_clean.csv"

def process_data(df):
    # Typisieren
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")
    for col in ("summary", "extensive", "about"):
        if col in df.columns:
            # Setze 1, wenn etwas anderes als "N" vorhanden ist, sonst 0
            df[col] = df[col].apply(lambda x: 1 if pd.notna(x) and str(x).strip().lower() != "n" else 0)
    # Null-IDs raus
    df = df[df["app_id"].notna()]
    return df

def main():
    read_kwargs = dict(engine="python", sep=",", quotechar='"',
                       quoting=csv.QUOTE_MINIMAL, skipinitialspace=True, encoding="utf-8-sig")
    # Ganze Datei einlesen
    df = pd.read_csv(INPUT, **read_kwargs, on_bad_lines="warn")

    # Daten verarbeiten
    df = process_data(df)

    # Deduplizieren
    df = df.drop_duplicates()

    # Ergebnis speichern
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[done] written {len(df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
