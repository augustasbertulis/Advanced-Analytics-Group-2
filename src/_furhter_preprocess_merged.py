"""
Description:
- drop playtime  --> values are missing (--> maybe fill up with data from 'how long to beat')
- drop columns where there is no diverse data in howl set
- calculate price_in_eur; drop price_overview.final, price_overview.currency, price, initial_price
- combine languages_x and languages_y; genres_x and genres_y
- one-hot encode top 10 genres, drop the rest
- drop rows where both developer and publisher are missing
- normalize publisher and developer names (Levenshtein-like clustering)
"""

import re
import pandas as pd

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

# frequency across apps (presence-based)
all_genres = pd.Series([g for lst in df["merged_genres"] for g in lst])
top10 = list(all_genres.value_counts().head(10).index)

# one-hot with rank-based names
for i, g in enumerate(top10, 1):
    df[f"genre{i}_{safe_name(g)}"] = df["merged_genres"].apply(lambda lst, gg=g: int(gg in lst))

# drop raw genre columns and temp
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

# --- Company name normalization ---------------------------------------------
_SUFFIX_STOPWORDS = {
    "inc","ltd","llc","co","corp","corporation","company","studios","studio",
    "games","game","interactive","entertainment","soft","software","plc","gmbh",
    "srl","sa","sas","bv","kg","oy","ab","limited"
}
WORD_RE = re.compile(r"[0-9A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF&+]+", re.UNICODE)

def _tokenize_company(name: str) -> list[str]:
    text = str(name)
    text = (text.lower()
                 .replace("@", " ")
                 .replace("/", " ")
                 .replace("-", " ")
                 .replace("_", " "))
    raw_tokens = WORD_RE.findall(text)

    kept = []
    for t in raw_tokens:
        if t == "&":
            t = "and"
        t = t.rstrip(".")
        if t and t not in _SUFFIX_STOPWORDS:
            kept.append(t)

    if kept:
        return kept
    return [max(raw_tokens, key=len)] if raw_tokens else []

def _canon_key(name: str) -> str:
    toks = _tokenize_company(name)
    return " ".join(sorted(toks))

def _choose_representative(names: list[str]) -> str:
    counts = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    best = max(counts.items(), key=lambda x: (x[1], -len(x[0])))
    return best[0]

def normalize_company_column(df_: pd.DataFrame, col: str) -> pd.Series:
    s = df_[col].astype("string")
    missing_mask = s.isna() | s.str.strip().eq("")
    keys = s.fillna("").map(_canon_key)

    groups = {}
    for orig, key in zip(s.fillna(""), keys):
        if key:
            groups.setdefault(key, []).append(orig)
    rep_map = {k: _choose_representative(v) for k, v in groups.items()}

    def _fallback_pretty(orig) -> str | None:
        if pd.isna(orig): return None
        keep_upper = {"VR","AR","III","II","IV","V","VI"}
        words = str(orig).strip()
        if not words: return None
        return " ".join(w if w.upper() in keep_upper else w.title() for w in words.split())

    out = []
    for orig, key, is_missing in zip(s, keys, missing_mask):
        if is_missing:
            out.append(pd.NA)
        elif key:
            out.append(rep_map.get(key) or _fallback_pretty(orig))
        else:
            out.append(_fallback_pretty(orig))
    return pd.Series(out, index=df_.index, dtype="string")

# apply normalization
df["publisher"] = normalize_company_column(df, "publisher")
df["developer"] = normalize_company_column(df, "developer")

print("Unique publishers (cleaned):", df["publisher"].nunique(dropna=True))
print("Unique developers (cleaned):", df["developer"].nunique(dropna=True))

# --- Drop unneeded columns ---------------------------------------------------
cols_to_drop = [
    "playtime_average_forever","playtime_average_2weeks",
    "playtime_median_forever","playtime_median_2weeks",
    "summary","extensive","about",
    "price_overview.final","price_overview.currency","price","initial_price",
    "header_image","steamspy_scorer_rank","exchange_rate"  # if present
]
df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# --- Save --------------------------------------------------------------------
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"[done] saved cleaned dataset {df.shape} → {output_path}")
