"""
Description:
- drop playtime  --> values are missing (--> maybe fill up with data from 'how long to beat')
- drop columns where there is no diverse data in whole set
- calculate price_in_eur; drop price_overview.final, price_overview.currency, price, initial_price
- combine languages_x and languages_y; genres_x and genres_y
- one-hot encode top 10 genres, drop the rest
- drop rows where both developer and publisher are missing
- Levenshtein distance for publisher & developer
"""

import re
import pandas as pd
import Levenshtein  # pip install python-Levenshtein

# I/O
import_path  = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/combined.csv"
output_path  = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/combined_clean.csv"
df = pd.read_csv(import_path, low_memory=False)

# --- FX map & EUR price ------------------------------------------------------
fx = {"EUR":1.0,"USD":0.85,"GBP":1.17,"MXN":0.05,"RUB":0.011,"IDR":0.000056,
      "PLN":0.22,"BRL":0.17,"CNY":0.12,"AUD":0.55,"SGD":0.62,"ILS":0.23}
df["price_in_eur"] = df["price"] * df["price_overview.currency"].map(fx).fillna(0)

# --- small helpers -----------------------------------------------------------
def split_list(s: str) -> list[str]:
    return [] if pd.isna(s) else [x.strip() for x in str(s).split(",") if x.strip()]

def merge_unique(a: list[str], b: list[str]) -> list[str]:
    seen, out = set(), []
    for x in a + b:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def safe_name(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s).lower())
    s = re.sub(r"[^a-z0-9_]", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

# --- Merge genres_x & genres_y ----------------------------------------------
df["merged_genres"] = df.apply(lambda r: merge_unique(split_list(r.get("genres_x")),
                                                      split_list(r.get("genres_y"))), axis=1)

all_genres = pd.Series([g for lst in df["merged_genres"] for g in lst])
top10 = list(all_genres.value_counts().head(10).index)

for i, g in enumerate(top10, 1):
    df[f"genre{i}_{safe_name(g)}"] = df["merged_genres"].apply(lambda lst, gg=g: int(gg in lst))

df.drop(columns=["genres_x","genres_y","merged_genres"], inplace=True, errors="ignore")

# --- Merge languages_x & languages_y ----------------------------------------
df["languages"] = df.apply(
    lambda r: ", ".join(merge_unique(split_list(r.get("languages_x")),
                                     split_list(r.get("languages_y")))) or None,
    axis=1
)
df.drop(columns=["languages_x","languages_y"], inplace=True, errors="ignore")

# --- Drop rows where both developer and publisher are missing ---------------
df = df.dropna(subset=["developer", "publisher"], how="all")

# --- Drop unneeded columns --------------------------------------------------
cols_to_drop = [
    "playtime_average_forever","playtime_average_2weeks",
    "playtime_median_forever","playtime_median_2weeks",
    "summary","extensive","about",
    "price_overview.final","price_overview.currency","price","initial_price",
    "header_image","steamspy_scorer_rank","exchange_rate"  # if present
]
df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# --- Levenshtein overwrite for publisher & developer -----------------------
ref_pub = "Valve"
ref_dev = "Valve"

df["publisher"] = df["publisher"].apply(
    lambda x: ref_pub if pd.notna(x) and Levenshtein.distance(str(x).lower(), ref_pub.lower()) <= 3 else x
)

df["developer"] = df["developer"].apply(
    lambda x: ref_dev if pd.notna(x) and Levenshtein.distance(str(x).lower(), ref_dev.lower()) <= 3 else x
)

# --- Save -------------------------------------------------------------------
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"[done] saved cleaned dataset {df.shape} → {output_path}")
