# data_preprocessing_categories.py
from pathlib import Path
import pandas as pd
import re
import csv  # <-- FIX: needed for quoting option

# Resolve repo root from this file (assuming this script is in <repo>/src)
ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw data" / "categories.csv"
OUTPUT_CSV = ROOT / "data" / "processed data" / "categories_clean.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Basic normalization for common translations -> English
CATEGORY_TRANSLATIONS = {
    "Family Sharing": ["Family Sharing", "Familienbibliothek", "Семейный доступ", "Partage familial", "Compartilhamento em família"],
    "Multi-player": ["Multi-player", "Mehrspieler", "Для нескольких игроков", "Multijoueur", "Multijogador"],
    "Single-player": ["Single-player", "Einzelspieler", "Для одного игрока", "Solo", "Um jogador"],
    "Co-op": ["Co-op", "Кооператив", "Coopération", "Cooperação"],
    "Steam Cloud": ["Steam Cloud", "Облако Steam", "Cloud Steam"],
    "Steam Achievements": ["Steam Achievements", "Достижения Steam", "Succès Steam", "Conquistas Steam"],
    "Remote Play Together": ["Remote Play Together", "Совместная игра удаленно", "Remote Play sur tablette", "Remote Play na TV"],
    "Valve Anti-Cheat enabled": ["Valve Anti-Cheat enabled", "Включён античит Valve", "Valve Anti-Cheat integriert"],
    "Stats": ["Stats", "Статистика", "Statistiques"],
    "Steam Trading Cards": ["Steam Trading Cards", "Коллекционные карточки Steam", "Cartes à échanger Steam"],
}

# Precompute reverse lookup for faster normalization
_REV = {alt: eng for eng, alts in CATEGORY_TRANSLATIONS.items() for alt in alts}
_SPLIT_RE = re.compile(r"[;,/|]+")

def normalize_category(cat: str) -> str:
    if pd.isna(cat):
        return None
    s = str(cat).strip()
    if not s or s in {"\\N", "\\n"}:
        return None
    # if a cell accidentally contains multiple categories, split & normalize each
    parts = [p.strip() for p in _SPLIT_RE.split(s) if p.strip()] or [s]
    normed = []
    for p in parts:
        normed.append(_REV.get(p, p))  # map translation -> English (fallback to original)
    # return a comma-joined canonical string for this cell (will be deduped later)
    return ", ".join(normed)

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize
    df = df.copy()
    df["app_id"] = df["app_id"].astype("string").str.strip()
    df["category"] = df["category"].apply(normalize_category)

    # Drop rows with missing app_id or category
    df = df[(df["app_id"].notna()) & (df["app_id"] != "") & df["category"].notna()]

    # Explode any multi-category cells safely
    exploded = []
    for app, cat in zip(df["app_id"], df["category"]):
        parts = [p.strip() for p in _SPLIT_RE.split(cat)] if _SPLIT_RE.search(cat) else [c.strip() for c in cat.split(",")]
        for p in parts:
            if p:
                exploded.append((app, p))
    if not exploded:
        return pd.DataFrame(columns=["app_id", "categories"])

    long = pd.DataFrame(exploded, columns=["app_id", "category"])

    # Deduplicate categories per app_id and join as a single string
    grouped = (
        long.drop_duplicates(["app_id", "category"])
            .groupby("app_id")["category"]
            .apply(lambda s: ", ".join(sorted(s)))
            .reset_index()
            .rename(columns={"category": "categories"})
    )

    # Make app_id an Int64 key when possible (nullable int)
    grouped["app_id"] = pd.to_numeric(grouped["app_id"], errors="coerce").astype("Int64")

    return grouped

def main():
    read_kwargs = dict(
        engine="python",
        sep=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        skipinitialspace=True,
        encoding="utf-8-sig",
        dtype={"app_id": "string", "category": "string"},
        on_bad_lines="skip",
    )

    df = pd.read_csv(INPUT, **read_kwargs)
    df_transformed = process_data(df)

    df_transformed.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[done] written {len(df_transformed)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
