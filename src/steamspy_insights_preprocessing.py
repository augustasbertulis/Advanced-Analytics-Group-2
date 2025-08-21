import pandas as pd
import re
import os

# File paths
INPUT_FILE = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\raw data\steam-insights-main\steamspy_insights\steamspy_insights.csv"
OUTPUT_FILE = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\processed data\steamspy_insights_cleaned.csv"


def split_range(s):
    """Splits a string like '1,000 .. 5,000' into [1000, 5000]."""
    if pd.isna(s):
        return pd.Series([pd.NA, pd.NA])

    match = re.match(r"\s*([\d,._ ]+)\s*\.\.\s*([\d,._ ]+)\s*$", str(s))
    if not match:
        return pd.Series([pd.NA, pd.NA])

    def normalize(x):
        return pd.to_numeric(re.sub(r"[^\d]", "", x), errors="coerce")

    return pd.Series([normalize(match.group(1)), normalize(match.group(2))])


def clean_steamspy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the SteamSpy dataset."""

    # === Normalize missing values ===
    missing_values = {"\\N", "0", "(none)", "none", "None", "-"}
    df = df.replace(list(missing_values), pd.NA)

    # === Remove rows with missing app_id ===
    missing_app_id_count = df["app_id"].isna().sum()
    df = df.dropna(subset=["app_id"])

    # === Remove rows where both developer and publisher are missing ===
    missing_dev_pub_count = ((df["developer"].isna()) & (df["publisher"].isna())).sum()
    df = df.dropna(subset=["developer", "publisher"], how='all')

    # === Fill missing publishers based on developer ===
    dev_pub_map = (
        df.dropna(subset=["publisher"])
        .groupby("developer")["publisher"]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
    )

    missing_publisher_before = df["publisher"].isna().sum()
    mask = df["publisher"].isna() & df["developer"].notna()
    df.loc[mask, "publisher"] = df.loc[mask, "developer"].map(dev_pub_map)
    filled_publishers = missing_publisher_before - df["publisher"].isna().sum()

    # === Remove any remaining rows with missing publisher ===
    removed_after_mapping = df["publisher"].isna().sum()
    df = df.dropna(subset=["publisher"])

    # === Convert price-related columns to decimal ===
    for col in ["price", "initial_price", "discount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100

    # === Print summary ===
    print(f"Rows removed due to missing app_id: {missing_app_id_count}")
    print(f"Rows removed due to missing developer and publisher: {missing_dev_pub_count}")
    print(f"Publishers filled from developer mapping: {filled_publishers}")
    print(f"Rows removed due to missing publisher after mapping: {removed_after_mapping}")
    print(f"Remaining rows after cleaning: {len(df)}")

    return df


def main():
    # Read CSV
    df = pd.read_csv(INPUT_FILE, sep=",", quotechar='"', on_bad_lines="skip")

    # Clean data
    cleaned_df = clean_steamspy_data(df)

    # Split owners_range into min/max
    cleaned_df[["owners_min", "owners_max"]] = cleaned_df["owners_range"].apply(split_range)
    cleaned_df = cleaned_df.drop(columns=["owners_range"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Export to CSV
    cleaned_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
