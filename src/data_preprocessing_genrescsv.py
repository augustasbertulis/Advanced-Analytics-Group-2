# data_preprocessing_genrescsv.py
# One-hot encode the TOP-10 genres for each app_id; fold the rest into a single "other_genres" column.

import re
from pathlib import Path
import pandas as pd

#-------------------- File paths (adjust if needed) --------------------
input_file  = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/raw data/genres.csv"
output_file = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/genres_clean.csv"
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

# -------------------- helpers --------------------
def to_none(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    return None if s in {"", r"\N", r"\n"} else s

# split on common separators
SPLIT_RE = re.compile(r"[;,/|]+")

# HTML / BBCode / stray markup
BR_RE       = re.compile(r"(?i)[<>\s]*/?\s*br\s*/?\s*[<>]?")
HTML_TAG_RE = re.compile(r"<[^>]+>")
BBCODE_RE   = re.compile(r"\[/?[a-zA-Z]+]")
PUNCT_TRIM  = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)

# multilingual mapping → canonical English genres (extend as needed)
CANON_MAP = [
    (re.compile(r"(?i)^(action|acción|azione|aktion|ação|akcja|экшен|动作)$"), "Action"),
    (re.compile(r"(?i)^(adventure|aventura|avventura|abenteuer|приключен(?:ие|ия)|冒险)$"), "Adventure"),
    (re.compile(r"(?i)^(rpg|role[- ]?playing|jeu\s*de\s*rôle|juego\s*de\s*rol|gioco\s*di\s*ruolo|rollenspiel|ролев(?:ая|ые)|角色扮演)$"), "RPG"),
    (re.compile(r"(?i)^(strategy|estrategia|strategie|strategia|стратег(?:ия|ии)|策略)$"), "Strategy"),
    (re.compile(r"(?i)^(simulation|simulaci[oó]n|simulazione|симулятор|模拟)$"), "Simulation"),
    (re.compile(r"(?i)^(casual|décontracté|ocasional|casuale|gelegentlich|казуальн(?:ая|ые)|休闲)$"), "Casual"),
    (re.compile(r"(?i)^(sports?|deportes?|sport|sportivi|спорт|体育)$"), "Sports"),
    (re.compile(r"(?i)^(racing|carreras|rennen|corse|гонки|赛车)$"), "Racing"),
    (re.compile(r"(?i)^(indie|indépendant|independiente|indipendente|инди|独立)$"), "Indie"),
    (re.compile(r"(?i)^(shooter|fps|tps|tir|disparos|sparatutto|sch(?:ü|u)tzer|шутер|射击)$"), "Shooter"),
    (re.compile(r"(?i)^(horror|terror|horreur|orrore|ужасы|恐怖)$"), "Horror"),
    (re.compile(r"(?i)^(survival|supervivencia|survie|sopravvivenza|überleben|выживание|生存)$"), "Survival"),
    (re.compile(r"(?i)^(platform(?:er)?|plataformas|plate[- ]?forme|plattform|платформер|平台)$"), "Platformer"),
    (re.compile(r"(?i)^(puzzle|puzle|casse[- ]?t(?:e|ê)te|rompicapo|rätsel|головоломка|益智)$"), "Puzzle"),
]

def canon_genre(token: str) -> str:
    t = token.strip()
    t = BR_RE.sub(" ", t)
    t = HTML_TAG_RE.sub("", t)
    t = BBCODE_RE.sub("", t)
    t = PUNCT_TRIM.sub("", t).strip()
    if not t:
        return ""
    for rx, name in CANON_MAP:
        if rx.fullmatch(t):
            return name
    return t.title()  # keep unknowns but tidy

# -------------------- main --------------------
def main():
    # Read as strings, auto-detect delimiter
    raw = pd.read_csv(input_file, sep=None, engine="python", dtype=str, on_bad_lines="skip")

    # Keep only the two expected columns (tolerate header variants)
    cols = {c.lower(): c for c in raw.columns}
    app_col   = cols.get("app_id") or cols.get("appid") or cols.get("id")
    genre_col = cols.get("genre") or cols.get("genres")
    if app_col is None or genre_col is None:
        raise KeyError("Expected columns 'app_id' and 'genre' (or close variants).")

    df = raw[[app_col, genre_col]].rename(columns={app_col: "app_id", genre_col: "genre"})
    df["app_id"] = df["app_id"].map(to_none)
    df["genre"]  = df["genre"].map(to_none)

    total_rows = len(df)

    # explode genres per row -> (app_id, genre_token)
    rows = []
    for app, g in zip(df["app_id"], df["genre"]):
        if app is None or g is None:
            continue
        parts = [p.strip() for p in SPLIT_RE.split(g) if p.strip()] or [g]
        for p in parts:
            cg = canon_genre(p)
            if cg:
                rows.append((app, cg))

    if not rows:
        pd.DataFrame(columns=["app_id"]).to_csv(output_file, index=False, encoding="utf-8")
        print("No valid genres found. Wrote empty file with header only.")
        return

    long = pd.DataFrame(rows, columns=["app_id", "genre_canon"])

    # aggregate to unique set per app_id
    per_app = (long.groupby("app_id")["genre_canon"]
                    .apply(lambda s: sorted(set(s)))
                    .reset_index())

    # global frequencies (by app presence)
    freq = (long.drop_duplicates(["app_id", "genre_canon"])
                 .groupby("genre_canon")["app_id"].nunique()
                 .sort_values(ascending=False))

    top10 = list(freq.head(10).index)

    # one-hot for top 10; keep a single text column for the rest
    def one_hot_row(genres):
        present = set(genres)
        cols = {f"genre_{g}": int(g in present) for g in top10}
        other = sorted(g for g in present if g not in top10)
        cols["other_genres"] = ", ".join(other) if other else None
        return pd.Series(cols)

    wide = per_app.join(per_app["genre_canon"].apply(one_hot_row))
    wide = wide.drop(columns=["genre_canon"], errors="ignore")

    # make app_id a proper nullable integer key
    wide["app_id"] = pd.to_numeric(wide["app_id"], errors="coerce").astype("Int64")
    bad_ids = int(wide["app_id"].isna().sum())
    if bad_ids:
        print(f"Warning: {bad_ids} app_id values could not be parsed to integers.")

    # save
    wide.to_csv(output_file, index=False, encoding="utf-8")

    # summary
    print(f"Raw rows read: {total_rows:,}")
    print(f"Unique apps with genres: {wide.shape[0]:,}")
    print(f"Distinct canonical genres: {freq.shape[0]:,}")
    print("Top 10 genres:")
    for g, c in freq.head(10).items():
        print(f"  {g:<15} {c:,} apps")
    print(f"Saved one-hot file → {output_file}")

if __name__ == "__main__":
    main()
