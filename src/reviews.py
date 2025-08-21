import csv
import re
import pandas as pd
from pathlib import Path

ID_OK = re.compile(r"^\d+$")

def reparse_if_single_cell(row):
    """If the whole line is in one cell, try to re-parse it as CSV."""
    if len(row) == 1 and row[0]:
        inner = row[0].strip().strip('"')
        row = next(csv.reader([inner.replace('""', '"')]))
    return row

def main():
    rows = []
    with Path(INPUT).open(newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = reparse_if_single_cell(next(rdr))
        header = [h.strip() for h in header]
        ncols = len(header)

        total = kept = 0
        for r in rdr:
            if not r:
                continue
            r = reparse_if_single_cell(r)
            if r and r[0].strip().lower() == "app_id":
                continue
            total += 1
            r = (r + [""] * ncols)[:ncols]

                continue
            kept += 1
            rows.append(r)

    df = pd.DataFrame(rows, columns=header)

    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["app_id"]).reset_index(drop=True)





if __name__ == "__main__":
    main()
