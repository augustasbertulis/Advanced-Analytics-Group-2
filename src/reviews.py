import csv
import re
import pandas as pd
from pathlib import Path
from math import nan

INPUT = "raw data/steam-insights-main/reviews.csv"
OUTPUT_CSV = "clean data/reviews_clean.csv"
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

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

    ID_OK = re.compile(r"^\d+$")

    def reparse_if_single_cell(row):
        """If the whole line is in one cell, try to re-parse it as CSV."""
        if len(row) == 1 and row[0]:
            inner = row[0].strip().strip('"')
            row = next(csv.reader([inner.replace('""', '"')]))
        return row


    def clean_second_column_numeric(df, col_name_guess="review_score", *, min_val=None, max_val=None):

        col = col_name_guess if col_name_guess in df.columns else df.columns[1]

        s = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({r"\N": None, "": None}, regex=False)  # note raw "\N"
        )

        # coerce to numeric; bad values become NaN
        s = pd.to_numeric(s, errors="coerce")

        # optional bounds
        if min_val is not None:
            s = s.where(s >= min_val)
        if max_val is not None:
            s = s.where(s <= max_val)

        # store back as nullable float
        df[col] = s.astype("Float64")

        # tiny summary
        print(f'Cleaned 2nd column "{col}": {df[col].notna().sum():,} numeric; {df[col].isna().sum():,} null.')
        return df


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

        # 1) Coerce IDs and drop anything that still failed (belt & suspenders)
        df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")
        before = len(df)
        df = df.dropna(subset=["app_id"]).reset_index(drop=True)

        # 2) Clean the SECOND column to numeric; non-numeric -> NaN
        df = clean_second_column_numeric(df, col_name_guess="review_score", min_val=0, max_val=10)

        # Stats
        print(f"Kept {kept:,} rows; dropped {total - kept:,} on raw ID check (from {total:,}).")
        print(f"Dropped {before - len(df):,} more on numeric coercion. Final rows: {len(df):,}")

        df.to_csv(Path(OUTPUT_CSV), index=False, encoding="utf-8")
        print(f"Saved → {OUTPUT_CSV}")


    if __name__ == "__main__":
        main()

def clean_third_column_review_desc(df, col_name_guess="review_score_description"):
    # 1) Locate the 3rd column safely
    col = col_name_guess if col_name_guess in df.columns else df.columns[2]

    print("Columns:", list(df.columns))
    print("Using 3rd column:", col)

    # 2) Snapshot BEFORE
    before_sample = df[col].head(5).tolist()
    print("Before (head):", before_sample)

    # Normalize spaces
    s = (
        df[col].astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Null rules
    mask_backslash_N     = s.eq(r"\N")                                # literal "\N"
    mask_no_user_reviews = s.str.fullmatch(r"(?i)no user reviews")    # case-insensitive
    mask_small_counts    = s.str.fullmatch(r"(?i)[1-9]\s+users?\s+reviews?")  # 1..9 user review(s)
    mask_null = mask_backslash_N | mask_no_user_reviews | mask_small_counts


    s_clean = s.mask(mask_null, other=pd.NA)

    # Ranking map (covers common Steam wordings; adjust if you only want 3 classes)
    rank_map = {
        "overwhelmingly negative": 1,
        "very negative":          2,
        "negative":               3,
        "mostly negative":        4,
        "mixed":                  5,
        "mostly positive":        6,
        "positive":               7,
        "very positive":          8,
        "overwhelmingly positive":9,
    }

    base_lower = s_clean.str.lower()
    rank = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for phrase, val in rank_map.items():
        rank = rank.mask(base_lower == phrase, other=val)

    # Prefix the ranked labels (e.g., "8 Very Positive")
    s_ranked = s_clean.copy()
    for phrase, val in rank_map.items():
        m = base_lower == phrase
        s_ranked.loc[m] = s_clean.loc[m].map(lambda x: f"{val} {x}" if pd.notna(x) else x)

    # Store back
    df[col] = s_ranked
    df[f"{col}_rank"] = rank

    # 3) Snapshot AFTER + counts
    after_sample = df[col].head(5).tolist()
    nulled = int(mask_null.sum())
    ranked = int(rank.notna().sum())
    print(f'Nulls set: {nulled:,} | Ranked: {ranked:,}')
    print("After  (head):", after_sample)

    # Also show quick value counts for sanity
    print("Top values after cleaning:")
    print(df[col].value_counts(dropna=False).head(10))

    return df

def clean_fourth_column_positive(df, col_name_guess="positive"):

    # identify the 4th column by name or position
    col = col_name_guess if col_name_guess in df.columns else df.columns[3]

    # normalize placeholders to actual nulls
    s = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({r"\N": None, "": None}, regex=False)  # literal \N
    )
    s = s.where(~s.str.fullmatch(r"(?i)nan|none"), other=None)

    s = pd.to_numeric(s, errors="coerce")

    # only allow non-negative counts
    s = s.where(s >= 0)

    # store back as nullable integer
    df[col] = s.astype("Int64")

    print(f'Cleaned 4th column "{col}": {df[col].notna().sum():,} numeric; {df[col].isna().sum():,} null.')
    df = clean_fourth_column_positive(df, col_name_guess="positive")
    return df

