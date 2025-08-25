import pandas as pd
from datetime import datetime
from paths import PROCESSED_DATA_DIR

# Load files
publishers = pd.read_csv(PROCESSED_DATA_DIR / "publisher_ranked_consensus.csv")
games = pd.read_csv(PROCESSED_DATA_DIR / "combined_clean.csv", low_memory=False)

# Take only the first 20 publishers and assign IDs
publishers_20 = publishers.head(20).copy()
publishers_20["publisher_id"] = range(1, len(publishers_20) + 1)

# --- Feature Engineering (for KPIs like growth_potential) ---
# owners_avg if missing
if "owners_avg" not in games.columns:
    games["owners_avg"] = (games["owners_min"] + games["owners_max"]) / 2

# release_date → days_since_release
if "release_date" in games.columns:
    games["release_date"] = pd.to_datetime(games["release_date"], errors="coerce")
else:
    games["release_date"] = pd.NaT

today = pd.Timestamp(datetime.today().date())
games["days_since_release"] = (today - games["release_date"]).dt.days
games["days_since_release"] = games["days_since_release"].replace(0, 1)

# growth_potential
games["growth_potential"] = (games["owners_max"] - games["owners_min"]) / games["days_since_release"]

# genres combined
games["genres_combined"] = games[["genres_x", "genres_y"]].fillna("").agg("; ".join, axis=1).str.strip("; ")

# --- Merge games with publishers ---
merged = pd.merge(
    games,
    publishers_20[["publisher", "publisher_id"]],
    on="publisher",
    how="inner"
)

# Sort
merged_sorted = merged.sort_values(by=["publisher_id", "publisher"])

# --- Final columns in requested order ---
keep_cols = [
    "publisher_id", "publisher",
    "app_id", "name", "release_date", "recommendations", "genres_combined", "price_in_eur",
    "owners_avg", "revenue_proxy", "positive", "total", "positive_score",
    "concurrent_users_yesterday", "active_engagement_score", "growth_potential"
]

# Keep only available columns
final_table = merged_sorted[[c for c in keep_cols if c in merged_sorted.columns]]

# Save output
out_path = PROCESSED_DATA_DIR / "attributes.csv"
final_table.to_csv(out_path, index=False)

print("✅ Attribute table saved to:", out_path)