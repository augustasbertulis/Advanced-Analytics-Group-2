#!/usr/bin/env python3
import pandas as pd
import csv


    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")
    for col in ("summary", "extensive", "about"):
        if col in df.columns:
    df = df[df["app_id"].notna()]
    return df

def main():

    df = process_data(df)
    df = df.drop_duplicates()

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[done] written {len(df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
