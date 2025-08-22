import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

"""
df['owners_avg'] = (df['owners_min'] + df['owners_max']) / 2
df['active_engagement_score'] = df['concurrent_users_yesterday'] / df['owners_avg']
"""

input_path  = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/combined_clean.csv"

#Load Data
df = pd.read_csv(input_path)

# Use only the relevant columns including the KPIs
df = df[['app_id', 'publisher', 'owners_avg', 'active_engagement_score', 'concurrent_users_yesterday']]

#Aggregate per publisher
publisher_df = df.groupby("publisher").agg({
    "active_engagement_score": "mean",
    "owners_avg": "mean",
    "concurrent_users_yesterday": "mean"
}).reset_index()

#Clustering
features = ["active_engagement_score", "concurrent_users_yesterday", "owners_avg"]
X = publisher_df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
publisher_df["cluster"] = kmeans.fit_predict(X_scaled)

#View Results
print("Cluster Summary:\n", publisher_df.groupby("cluster")[features].mean())
print("Examples per Cluster:")

for c in publisher_df["cluster"].unique():
    pubs = publisher_df[publisher_df["cluster"] == c]["publisher"].head(5).tolist()
    print(f"Cluster {c}: {pubs}")
