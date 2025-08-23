import pandas as pd

#Path Fabrizio:
#INPUT = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/raw data/tags.csv"
#OUTPUT_CSV = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/tags_clean.csv"
INPUT = "data/steam-insights-main/raw data/tags.csv"
OUTPUT_CSV = "data/clean data/tags_clean.csv"

def process_data(df):
    # Tags pro app_id zählen
    df_transformed = df.groupby("app_id")["tag"].count().reset_index()
    df_transformed.rename(columns={"tag": "tag_count"}, inplace=True)
    return df_transformed

def main():
    # CSV-Datei einlesen
    df = pd.read_csv(INPUT, engine="python", sep=",", quotechar='"',
                     escapechar="\\", skipinitialspace=True, encoding="utf-8-sig")

    # Daten verarbeiten
    df_transformed = process_data(df)

    # Ergebnis speichern
    df_transformed.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[done] written {len(df_transformed)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()