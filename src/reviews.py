#reviews
"""
Cleans up data
Applies text len score, sentiment score
and flagging based on most often used words in reviews, categorized as emotional, qualitative or buggy
"""
import csv
import re
import pandas as pd
from pathlib import Path

#INPUT = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/raw data/reviews.csv"
#OUTPUT = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/reviews_clean.csv"
INPUT = "data/raw data/steam-insights-main/reviews.csv"
OUTPUT = "data/clean data/reviews_clean.csv"
ID_OK = re.compile(r"^\d+$")

# values that should always become null
NULL_MARKERS = {
    r"\N", "N", "n", "", "None", "none", "nan", "NaN",
    "No user review", "No user reviews",
    "1 user reviews", "2 user reviews", "3 user reviews",
    "4 user reviews", "5 user reviews", "6 user reviews",
    "7 user reviews", "8 user reviews", "9 user reviews"
}

# --- Category word sets ---
QUALITY_WORDS = {
    "great", "good", "amazing", "fantastic", "excellent", "perfect",
    "brilliant", "wonderful", "superb", "beautiful", "solid", "polished"
}

EMOTION_WORDS = {
    "fun", "love", "enjoy", "enjoyed", "exciting", "addictive",
    "awesome", "happy", "satisfying", "relaxing", "immersive", "engaging"
}

BUG_WORDS = {
    "bug", "bugs", "buggy", "glitch", "glitches", "crash", "crashes",
    "freezes", "lag", "laggy", "issue", "issues", "problem", "problems"
}


def reparse_if_single_cell(row):
    """If the whole line is in one cell, try to re-parse it as CSV."""
    if len(row) == 1 and row[0]:
        inner = row[0].strip().strip('"')
        row = next(csv.reader([inner.replace('""', '"')]))
    return row


def clean_column(series: pd.Series):
    """Normalize placeholders → NA, attempt numeric conversion."""
    s = series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    # replace placeholders with NA
    s = s.where(~s.isin(NULL_MARKERS), other=pd.NA)

    # try numeric
    s_num = pd.to_numeric(s, errors="coerce")

    # if conversion works for most rows → keep numeric
    if s_num.notna().sum() >= 0.5 * len(s):
        return s_num.astype("Float64")
    else:  # keep as cleaned strings
        return s


def tokenize(text):
    """Lowercase + tokenize words."""
    text = str(text).lower()
    return re.findall(r"\b\w+\b", text)


def assign_flag(tokens):
    """Assign category flag based on keyword matches."""
    counts = {
        "Quality & Evaluation": sum(t in QUALITY_WORDS for t in tokens),
        "Emotions & Feelings": sum(t in EMOTION_WORDS for t in tokens),
        "Bug / Technical": sum(t in BUG_WORDS for t in tokens)
    }
    if all(v == 0 for v in counts.values()):
        return "none"
    return max(counts, key=counts.get)


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

            if not ID_OK.fullmatch(r[0].strip()):
                continue
            kept += 1
            rows.append(r)

    df = pd.DataFrame(rows, columns=header)

    # enforce numeric app_id
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["app_id"]).reset_index(drop=True)

    print(f"Columns detected ({len(df.columns)}): {list(df.columns)}")

    # clean every column except app_id
    for col in df.columns[1:]:
        df[col] = clean_column(df[col])

    # add flag column if reviews exist
    if "reviews" in df.columns:
        df["flag"] = df["reviews"].apply(lambda x: assign_flag(tokenize(x)))

    # save result
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[done] cleaned {len(df)} rows × {len(df.columns)} cols → {OUTPUT}")


if __name__ == "__main__":
    main()
