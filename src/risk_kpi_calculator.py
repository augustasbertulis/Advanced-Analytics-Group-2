# --- Step 0: Import packages ---
import pandas as pd
import os
from paths import PROCESSED_DATA_DIR

#-------------------- File paths --------------------
data_path = PROCESSED_DATA_DIR / "combined_clean.csv"
output_dir = PROCESSED_DATA_DIR
os.makedirs(output_dir, exist_ok=True)

# Load relevant columns
df = pd.read_csv(data_path, usecols=["app_id", "developer", "publisher", "languages"])

# --- Step 1: Build Publisher KPIs ---
# Unique developers per publisher
devs_per_pub = df.groupby("publisher")["developer"].nunique().reset_index(name="unique_developers")
# Total games per publisher
games_per_pub = df.groupby("publisher")["app_id"].nunique().reset_index(name="games_per_pub")
# Unique languages per publisher
langs_per_pub = df.groupby("publisher")["languages"].nunique().reset_index(name="languages_per_pub")

# Merge all publisher KPIs
publisher_kpis = devs_per_pub.merge(games_per_pub, on="publisher").merge(langs_per_pub, on="publisher")

# --- Step 2: Save results ---
publisher_kpis.to_csv(os.path.join(output_dir, "publisher_kpis.csv"), index=False)
