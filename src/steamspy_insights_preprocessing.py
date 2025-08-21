import pandas as pd, re
import os

# File paths
input_file = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\raw data\steam-insights-main\steamspy_insights\steamspy_insights.csv"
output_file = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\processed data\steamspy_insights_cleaned.xlsx"

def split_range(s):
    if pd.isna(s):
        return None
    m = re.match(r"\s*([0-9,._ ]+)\s*\.\.\s*([0-9,._ ]+)\s*$", str(s))
    if not m: 
        return pd.Series([pd.NA, pd.NA])
    def norm(x):
        x = re.sub(r"[^\d]", "", x)
        return pd.to_numeric(x, errors="coerce").astype("Int64")
    return pd.Series([norm(m.group(1)), norm(m.group(2))])

def clean_steamspy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the steamspy dataframe by:
    1. Removing rows where app_id is None or '\\N'
    2. Transforming '\\N' and '0' in developer/publisher into None
    3. Removing rows where BOTH developer and publisher are None
    Prints a summary of rows removed for each condition.
    """
    # Normalize missing values (\N → None, "0" → None)
    df = df.replace({"\\N": None, "0": None})

    # Count rows with missing app_id before removing
    missing_app_id_count = df["app_id"].isna().sum()

    # Drop rows with missing app_id
    df = df[df["app_id"].notna()]

    # Count rows with BOTH developer and publisher missing before removing
    missing_dev_pub_count = ((df["developer"].isna()) & (df["publisher"].isna())).sum()

    # Drop rows where BOTH developer and publisher are missing
    df = df[~(df["developer"].isna() & df["publisher"].isna())]



    # Print summary
    print(f"Rows removed due to missing app_id: {missing_app_id_count}")
    print(f"Rows removed due to missing developer and publisher: {missing_dev_pub_count}")
    print(f"Remaining rows after cleaning: {len(df)}")

    return df



def main():
    # Read the CSV file while handling messy formatting:
    # - on_bad_lines="skip" → skips rows with the wrong number of columns instead of raising an error
    df = pd.read_csv(input_file, sep=";", on_bad_lines="skip")

    # Clean data
    cleaned_df = clean_steamspy_data(df)
    cleaned_df[["owners_min","owners_max"]] = cleaned_df["owners_range"].apply(split_range)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Export to Excel
    cleaned_df.to_excel(output_file, index=False)

    print(f"Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    main()
