import pandas as pd

INPUT = "data/raw data/steam-insights-main/promotional.csv"
OUTPUT_CSV = "data/clean data/promotional_clean.csv"


def clean_promotional_data(input_path: str, output_path: str) -> None:
    # CSV-Datei einlesen
    df = pd.read_csv(input_path,
                     engine="python", sep=",", quotechar='"',
                     escapechar="\\", skipinitialspace=True, encoding="utf-8-sig")

    # Logik auf alle Spalten außer 'app_id' anwenden
    for column in df.columns:
        if column != 'app_id':
            df[column] = df[column].apply(
                lambda x: len(x.split(',')) if pd.notna(x) and x != 'N' else 0
            )

    # Ergebnis speichern
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(df.head())


def main():
    clean_promotional_data(INPUT, OUTPUT_CSV)


if __name__ == "__main__":
    main()
