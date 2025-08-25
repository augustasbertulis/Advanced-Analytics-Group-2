#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import csv
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

INPUT = RAW_DATA_DIR / "categories.csv"
OUTPUT_CSV = PROCESSED_DATA_DIR / "categories_clean.csv"
# INPUT = "data/raw data/steam-insights-main/categories.csv"
# OUTPUT_CSV = "data/clean data/categories_clean.csv"

CATEGORY_TRANSLATIONS = {
    "Family Sharing": ["Family Sharing", "Familienbibliothek", "Семейный доступ", "Partage familial", "Compartilhamento em família"],
    "Multi-player": ["Multi-player", "Mehrspieler", "Для нескольких игроков", "Multijoueur", "Multijogador"],
    "Single-player": ["Single-player", "Um Spieler", "Для одного игрока", "Solo", "Um jogador"],
    "Co-op": ["Co-op", "Кооператив", "Coopération", "Cooperação"],
    "Steam Cloud": ["Steam Cloud", "Облако Steam", "Cloud Steam"],
    "Steam Achievements": ["Steam Achievements", "Достижения Steam", "Succès Steam", "Conquistas Steam"],
    "Remote Play Together": ["Remote Play Together", "Совместная игра удаленно", "Remote Play sur tablette", "Remote Play na TV"],
    "Valve Anti-Cheat enabled": ["Valve Anti-Cheat enabled", "Включён античит Valve", "Valve Anti-Cheat integriert"],
    "Stats": ["Stats", "Статистика", "Statistiques"],
    "Steam Trading Cards": ["Steam Trading Cards", "Коллекционные карточки Steam", "Cartes à échanger Steam"]
}

def normalize_category(category):
    for english, translations in CATEGORY_TRANSLATIONS.items():
        if category in translations:
            return english
    return category  # Wenn keine Übersetzung gefunden wird, Originalwert zurückgeben

def process_data(df):
    # Kategorien normalisieren
    df["category"] = df["category"].apply(normalize_category)

    # Kategorien pro app_id zusammenfassen
    df_transformed = df.groupby("app_id")["category"].apply(lambda x: ", ".join(x)).reset_index()

    # Alle Kategorien extrahieren und zählen
    category_counts = df_transformed["category"].str.split(", ").explode().value_counts()

    # Top 10 Kategorien extrahieren
    top_10_categories = category_counts.head(10).index.tolist()

    # One-Hot-Encoding für die Top 10 Kategorien
    for category in top_10_categories:
        df_transformed[category] = df_transformed["category"].apply(lambda x: 1 if category in x else 0)

    # Restliche Kategorien in einer separaten Spalte speichern
    df_transformed["other_categories"] = df_transformed["category"].apply(
        lambda x: ", ".join([cat for cat in x.split(", ") if cat not in top_10_categories])
    )

    return df_transformed

def main():
    read_kwargs = dict(engine="python", sep=",", quotechar='"',
                       quoting=csv.QUOTE_MINIMAL, skipinitialspace=True, encoding="utf-8-sig")
    # Ganze Datei einlesen
    df = pd.read_csv(INPUT, **read_kwargs, on_bad_lines="warn")

    # Daten verarbeiten
    df_transformed = process_data(df)

    df_transformed = df_transformed.drop(columns=['category'])
    df_transformed.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[done] written {len(df_transformed)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

