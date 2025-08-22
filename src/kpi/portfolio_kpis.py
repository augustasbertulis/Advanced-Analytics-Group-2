# --- Step 0: Import packages ---
import pandas as pd
import os

# --- Step 1: Data ---
data_path = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\processed data\combined_clean.csv"
output_dir = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\kpi data"
os.makedirs(output_dir, exist_ok=True)

# Load only relevant columns
df = pd.read_csv(data_path, usecols=["app_id", "developer", "publisher"])

# # Handle missing & split multiple entries
# df["developer"] = df["developer"].fillna("").str.split(",|;")
# df["publisher"] = df["publisher"].fillna("").str.split(",|;")
# df = df.explode("developer").explode("publisher")
# df["developer"] = df["developer"].str.strip()
# df["publisher"] = df["publisher"].str.strip()
# df = df[(df["developer"] != "") & (df["publisher"] != "")]

# --- Step 2: Build KPIs ---
# Publisher KPIs
devs_per_pub = df.groupby("publisher")["developer"].nunique().reset_index(name="unique_developers")
games_per_pub = df.groupby("publisher")["app_id"].nunique().reset_index(name="games_per_pub")
publisher_kpis = devs_per_pub.merge(games_per_pub, on="publisher")

# Developer KPIs
developer_kpis = df.groupby("developer")["app_id"].nunique().reset_index(name="games_per_dev")

# --- Step 3: Output top 5 by KPI ---
top_publishers_by_devs = publisher_kpis.sort_values("unique_developers", ascending=False).head(5)
top_publishers_by_games = publisher_kpis.sort_values("games_per_pub", ascending=False).head(5)
top_developers_by_games = developer_kpis.sort_values("games_per_dev", ascending=False).head(5)

# --- Step 4: Save results ---
top_publishers_by_devs.to_csv(os.path.join(output_dir, "top_publishers_by_devs.csv"), index=False)
top_publishers_by_games.to_csv(os.path.join(output_dir, "top_publishers_by_games.csv"), index=False)
top_developers_by_games.to_csv(os.path.join(output_dir, "top_developers_by_games.csv"), index=False)

print("✅ Top 5 KPI results saved in:", output_dir)
print("\nTop 5 Publishers by Unique Developers:\n", top_publishers_by_devs)
print("\nTop 5 Publishers by Games:\n", top_publishers_by_games)
print("\nTop 5 Developers by Games:\n", top_developers_by_games)
