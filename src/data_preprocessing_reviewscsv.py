# data_preprocessing_reviewscsv.py
# Stage 1 — keep only clean numeric app_id rows

import csv
import re
from pathlib import Path
import pandas as pd

RAW = Path(r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/raw data/reviews.csv")
OUT = Path(r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/reviews_clean.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

ID_OK = re.compile(r"^\d+$")

def reparse_if_single_cell(row):
    """If the whole line is in one cell, try to re-parse it as CSV."""
    if len(row) == 1 and row[0]:
        inner = row[0].strip().strip('"')
        row = next(csv.reader([inner.replace('""', '"')]))
    return row

def main():
    rows = []
    with RAW.open(newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = reparse_if_single_cell(next(rdr))
        header = [h.strip() for h in header]
        ncols  = len(header)

        total = kept = 0
        for r in rdr:
            if not r:
                continue
            r = reparse_if_single_cell(r)
            # skip stray header repeats
            if r and r[0].strip().lower() == "app_id":
                continue

            total += 1
            # conform width (pad/truncate) so DataFrame aligns
            r = (r + [""] * ncols)[:ncols]

            first = r[0].strip()
            if not ID_OK.fullmatch(first):
                continue  # drop junk like '",5,"Mixed",...' in first cell

            kept += 1
            rows.append(r)

    df = pd.DataFrame(rows, columns=header)

    # Coerce IDs and drop anything that still failed (belt & suspenders)
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")
    before = len(df)
    df = df.dropna(subset=["app_id"]).reset_index(drop=True)

    # Stats
    print(f"Kept {kept:,} rows; dropped {total - kept:,} on raw ID check (from {total:,}).")
    print(f"Dropped {before - len(df):,} more on numeric coercion. Final rows: {len(df):,}")

    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Saved → {OUT}")

if __name__ == "__main__":
    main()
