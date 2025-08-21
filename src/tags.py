# data_preprocessing_tags.py
from pathlib import Path
import pandas as pd

# Resolve repo root from this file (assuming this script is in <repo>/src)
ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw data" / "tags.csv"
OUTPUT_CSV = ROOT / "data" / "processed data" / "tags_clean.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # Count tags per app_id
    out = df.groupby("app_id", as_index=False)["tag"].count()
    out.rename(columns={"tag": "tag_count"}, inplace=True)
    return out

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT}\n"
            f"CWD: {Path.cwd()}\n"
            "Tip: Run from PyCharm with working directory set to the repo root, "
            "or keep using these Path-based absolute paths."
        )

    df = pd.read_csv(
        INPUT,
        engine="python",
        sep=",",
        quotechar='"',
        escapechar="\\",
        skipinitialspace=True,
        encoding="utf-8-sig",
        dtype={"app_id": "Int64", "tag": "string"},
    )

    df_transformed = process_data(df)
    df_transformed.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[done] written {len(df_transformed)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
