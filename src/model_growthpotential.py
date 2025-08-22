import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

# --- Step 0: Load Data ---
input_path  = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/combined_clean.csv"
df = pd.read_csv(input_path, low_memory=False)

# Convert release_date_x to datetime
df['release_date_x'] = pd.to_datetime(df['release_date_x'], errors='coerce')

# Compute days since release (using today as reference)
today = pd.Timestamp(datetime.today().date())
df['days_since_release'] = (today - df['release_date_x']).dt.days

# Calculate owners_avg
df['owners_avg'] = (df['owners_min'] + df['owners_max']) / 2

# --- Step 1: New KPI: Growth Potential ---
# Avoid division by zero (replace 0 days with 1)
df['days_since_release'] = df['days_since_release'].replace(0, 1)
df['growth_potential'] = (df['owners_max'] - df['owners_min']) / df['days_since_release']

# --- Step 2: Aggregate per publisher ---
publisher_df = df.groupby("publisher").agg({
    "growth_potential": "mean",
    "owners_avg": "mean",
    "concurrent_users_yesterday": "mean"
}).reset_index()

# --- Step 3: Handle NaN values ---
publisher_df = publisher_df.fillna(0)   # replace NaN with 0

# --- Step 4: Clustering ---
features = ["growth_potential", "concurrent_users_yesterday", "owners_avg"]
X = publisher_df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
publisher_df["cluster"] = kmeans.fit_predict(X_scaled)

# --- Step 5: View Results ---
print("Cluster Summary:\n", publisher_df.groupby("cluster")[features].mean())
print("\nExamples per Cluster:")

for c in publisher_df["cluster"].unique():
    pubs = publisher_df[publisher_df["cluster"] == c]["publisher"].head(5).tolist()
    print(f"Cluster {c}: {pubs}")
