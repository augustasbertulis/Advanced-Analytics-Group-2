# publisher_summary.py
import pandas as pd
from paths import PROCESSED_DATA_DIR

# Input paths
CONS_PATH = PROCESSED_DATA_DIR / "publisher_ranked_consensus.xlsx"
GAMES_PATH = PROCESSED_DATA_DIR / "combined_clean.csv"
OUT_PATH = PROCESSED_DATA_DIR / "attributes.csv"


def main():
    # Load datasets
    consensus = pd.read_excel(CONS_PATH)   # <-- FIXED: read_excel for xlsx
    games = pd.read_csv(GAMES_PATH, low_memory=False)

    # --- Add publisher_id if missing ---
    if "publisher_id" not in consensus.columns:
        consensus = consensus.reset_index().rename(columns={"index": "publisher_id"})
        consensus["publisher_id"] = consensus.index + 1

    # Clean publisher names
    consensus["publisher"] = consensus["publisher"].astype(str).str.strip()
    games["publisher"] = games["publisher"].astype(str).str.strip()

    # --- Restrict to first 20 publishers ---
    consensus = consensus.head(20)

    # --- Feature engineering for games ---
    if "release_date" in games.columns:
        games["release_date"] = pd.to_datetime(games["release_date"], errors="coerce")
        today = pd.Timestamp("2025-08-25")
        games["days_since_release"] = (today - games["release_date"]).dt.days.replace(0, 1)
    else:
        games["days_since_release"] = 1

    if {"owners_max", "owners_min"}.issubset(games.columns):
        games["growth_potential"] = (games["owners_max"] - games["owners_min"]) / games["days_since_release"]
    else:
        games["growth_potential"] = 0

    # --- Merge ---
    merge_cols = ["publisher_id", "publisher"]
    if {"num_developers", "num_languages"}.issubset(consensus.columns):
        merge_cols += ["num_developers", "num_languages"]

    df = games.merge(consensus[merge_cols], on="publisher", how="inner")

    # --- Handle genres ---
    if {"genres_x", "genres_y"}.issubset(df.columns):
        df["genres_combined"] = df[["genres_x", "genres_y"]].fillna("").agg(", ".join, axis=1)
    elif "genres" in df.columns:
        df["genres_combined"] = df["genres"].fillna("")
    else:
        df["genres_combined"] = ""
    df["genres_combined"] = (
        df["genres_combined"]
        .str.replace(r",\s+", ", ", regex=True)
        .str.strip(", ")
    )

    # --- Drop per-game fields ---
    df = df.drop(columns=["app_id", "name", "release_date", "genres_x", "genres_y", "genres"], errors="ignore")

    # --- KPI and numerical columns ---
    kpi_cols = [
        "recommendations", "price_in_eur", "owners_avg", "revenue_proxy",
        "positive", "total", "positive_score", "concurrent_users_yesterday",
        "active_engagement_score", "growth_potential", "num_developers", "num_languages"
    ]

    # --- Convert to numeric & fill NaN with 0 ---
    for col in kpi_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Aggregation dictionary ---
    agg_dict = {col: "mean" for col in kpi_cols if col in df.columns}
    agg_dict["genres_combined"] = lambda x: ", ".join(sorted(set(", ".join(x).split(", "))))

    # --- Group by publisher_id + publisher ---
    publisher_summary = df.groupby(["publisher_id", "publisher"], as_index=False).agg(agg_dict)

    # Round floats
    publisher_summary = publisher_summary.round(2)

    # Save
    publisher_summary.to_csv(OUT_PATH, index=False)
    publisher_summary.to_excel(PROCESSED_DATA_DIR / "attributes.xlsx", index=False)

    print("Publisher summary saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
