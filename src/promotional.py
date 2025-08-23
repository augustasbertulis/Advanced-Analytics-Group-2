import pandas as pd

#Path Fabrizio:
#INPUT = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/raw data/promotional.csv"
#OUTPUT_CSV = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/promotional_clean.csv"
INPUT = "data/raw data/steam-insights-main/promotional.csv"
OUTPUT_CSV = "data/clean data/promotional_clean.csv"
# CSV-Datei einlesen
df = pd.read_csv(INPUT,
                 engine="python", sep=",", quotechar='"',
                 escapechar="\\", skipinitialspace=True, encoding="utf-8-sig")

# Logik auf alle Spalten außer 'app_id' anwenden
for column in df.columns:
    if column != 'app_id':
        df[column] = df[column].apply(
            lambda x: len(x.split(',')) if pd.notna(x) and x != 'N' else 0
        )

# Ergebnis speichern oder anzeigen
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(df.head())