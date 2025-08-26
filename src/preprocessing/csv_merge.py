import argparse
import os
import pandas as pd
from functools import reduce
import glob
from paths import PROCESSED_DATA_DIR, DATA_DIR

# ---------------------------------------------------------------------
# DEFAULT CONFIG
# ---------------------------------------------------------------------

input_folder_default = PROCESSED_DATA_DIR
output_file_default = PROCESSED_DATA_DIR / "combined.csv"


# input_folder_default = "data/clean data/"
# output_file_default = "data/clean data/combined.csv"

# ---------------------------------------------------------------------
# CORE PIPELINE
# ---------------------------------------------------------------------
def combine_csvs(input_folder: str, output_file: str) -> None:
    """
    Load all cleaned CSVs from the input folder and merge them on 'app_id'.
    """

    # Load CSV files
    des = pd.read_csv(os.path.join(input_folder, "descriptions_clean.csv"))
    gam = pd.read_csv(os.path.join(input_folder, "games_clean.csv"))
    gen = pd.read_csv(os.path.join(input_folder, "genres_clean.csv"))
    pro = pd.read_csv(os.path.join(input_folder, "promotional_clean.csv"))
    ste = pd.read_csv(os.path.join(input_folder, "steamspy_insights_clean.csv"))
    tag = pd.read_csv(os.path.join(input_folder, "tags_clean.csv"))

    # Set low_memory=False to resolve mixed types warning
    rev = pd.read_csv(os.path.join(input_folder, "reviews_clean.csv"), low_memory=False)

    cat = pd.read_csv(os.path.join(input_folder, "categories_clean.csv"))

    # Start with games DataFrame
    df = gam.copy()

    # Merge step by step
    df = df.merge(des, on="app_id", how="left")
    df = df.merge(gen, on="app_id", how="left")
    df = df.merge(pro, on="app_id", how="left")
    df = df.merge(ste, on="app_id", how="left")
    df = df.merge(tag, on="app_id", how="left")
    df = df.merge(rev, on="app_id", how="left")
    df = df.merge(cat, on="app_id", how="left")

    # Save the combined DataFrame to a new CSV
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[done] combined dataset {df.shape} → {output_file}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine multiple cleaned Steam CSV datasets into one file.")
    p.add_argument("-i", "--input", default=input_folder_default, help="Folder containing cleaned CSVs")
    p.add_argument("-o", "--output", default=output_file_default, help="Path to output combined CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    combine_csvs(args.input, args.output)


if __name__ == "__main__":
    main()
