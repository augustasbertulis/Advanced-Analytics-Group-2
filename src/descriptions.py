#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import csv

# Resolve repo root from this file (assumes this script is in <repo>/src)
ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw data" / "descriptions.csv"
OUTPUT_CSV = ROOT / "data" / "processed data" / "descriptions_clean.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # app_id -> nullable Int64
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")

    # make summary/extensive/about binary: 1 if present and not literal "N", else 0
    for col in ("summary", "extensive", "about"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: 1 if (pd.notna(x) and str(x).strip().lower() != "n") else 0
            )

    # drop rows without app_id
    df = df[df["app_id"].notna()]
    return df

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT}\n"
            f"CWD: {Path.cwd()}\n"
            "Tip: keep this script in <repo>/src or adjust ROOT accordingly."
        )

    # Read file (robust to messy quoting)
    read_kwargs = dict(
        engine="python",
        sep=",",                # set to None to auto-detect if needed
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        skipinitialspace=True,
        encoding="utf-8-sig",
        on_bad_lines="warn",
        dtype=str,             # read as strings; we cast in process_data
    )
    df = pd.read_csv(INPUT, **read_kwargs)

    df = process_data(df)
    df = df.drop_duplicates()

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[done] written {len(df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
