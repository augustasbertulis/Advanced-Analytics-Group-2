import re
import pandas as pd

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
#import_path  = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/combined.csv"
#output_path  = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/combined_clean.csv"
import_path = "data/clean data/combined.csv"
output_path = "data/clean data/combined_clean.csv"
fx = {
    "EUR":1.0,"USD":0.85,"GBP":1.17,"MXN":0.05,"RUB":0.011,"IDR":0.000056,
    "PLN":0.22,"BRL":0.17,"CNY":0.12,"AUD":0.55,"SGD":0.62,"ILS":0.23
}

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
df = pd.read_csv(import_path, low_memory=False)

# ---------------------------------------------------------------------
# PRICE HANDLING
# ---------------------------------------------------------------------
df["price_in_eur"] = df["price"] * df["price_overview.currency"].map(fx).fillna(0)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# GENRES (merge + one-hot top 10)
# ---------------------------------------------------------------------
df["merged_genres"] = df.apply(lambda r: merge_unique(
    split_list(r.get("genres")), split_list(r.get("genres"))
), axis=1)

all_genres = pd.Series([g for lst in df["merged_genres"] for g in lst])
top10 = list(all_genres.value_counts().head(10).index)

for i, g in enumerate(top10, 1):
    df[f"genre{i}_{safe_name(g)}"] = df["merged_genres"].apply(lambda lst, gg=g: int(gg in lst))

df.drop(columns=["genres","merged_genres"], inplace=True, errors="ignore")

# ---------------------------------------------------------------------
# LANGUAGES (merge)
# ---------------------------------------------------------------------
df["languages"] = df.apply(
    lambda r: ", ".join(merge_unique(split_list(r.get("languages")),
                                     split_list(r.get("languages_y")))) or None,
    axis=1
)
df.drop(columns=["languages_x","languages_y"], inplace=True, errors="ignore")

# ---------------------------------------------------------------------
# COMPANY NAME NORMALIZATION
# ---------------------------------------------------------------------
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
    s = df_[col].astype("string").fillna("Unknown")
    missing_mask = s.str.strip().eq("")
    keys = s.map(_canon_key)

    groups = {}
    for orig, key in zip(s, keys):
        if key:
            groups.setdefault(key, []).append(orig)
    rep_map = {k: _choose_representative(v) for k, v in groups.items()}

    def _fallback_pretty(orig) -> str:
        keep_upper = {"VR","AR","III","II","IV","V","VI"}
        words = str(orig).strip()
        if not words: return "Unknown"
        return " ".join(w if w.upper() in keep_upper else w.title() for w in words.split())

    out = []
    for orig, key, is_missing in zip(s, keys, missing_mask):
        if is_missing:
            out.append("Unknown")
        elif key:
            out.append(rep_map.get(key) or _fallback_pretty(orig))
        else:
            out.append(_fallback_pretty(orig))
    return pd.Series(out, index=df_.index, dtype="string")

# Apply normalization (no row drops)
df["publisher"] = normalize_company_column(df, "publisher")
df["developer"] = normalize_company_column(df, "developer")

# ---------------------------------------------------------------------
# DROP UNUSED COLUMNS
# ---------------------------------------------------------------------
cols_to_drop = [
    "playtime_average_forever","playtime_average_2weeks",
    "playtime_median_forever","playtime_median_2weeks",
    "summary","extensive","about",
    "price_overview.final_x", "price_overview.final_y","price_overview.currency_x", "price_overview.currency_y","price","initial_price",
    "header_image","steamspy_scorer_rank","exchange_rate", "name_y", "release_date_y", "is_free_y", "type_y"
]
df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------
"""
Active Engagement Index = (concurrent_users_yesterday / owners_max) --> measures how many owners are still active.
Positivity Ratio = positive / total --> proxy for community reception
Revenue Proxy = owners_avg * price_in_eur
"""
df['owners_avg'] = (df['owners_min'] + df['owners_max']) / 2
df['positive_score'] = df['positive'] / df['total']
df['revenue_proxy'] = df['owners_avg'] * df['price_in_eur']
df['active_engagement_score'] = df['concurrent_users_yesterday'] / df['owners_avg']

# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"[done] saved cleaned dataset {df.shape} → {output_path}")
