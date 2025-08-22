# --- Step 0: Import packages ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# --- Step 1: Data ---
data_path = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\processed data\combined_clean.csv"
output_dir = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\kpi_data_multi"
os.makedirs(output_dir, exist_ok=True)

# Load only relevant columns
df = pd.read_csv(data_path, usecols=["app_id", "developer", "publisher"])

# Handle missing & split multiple entries
df["developer"] = df["developer"].fillna("").str.split(",|;")
df["publisher"] = df["publisher"].fillna("").str.split(",|;")
df = df.explode("developer").explode("publisher")
df["developer"] = df["developer"].str.strip()
df["publisher"] = df["publisher"].str.strip()
df = df[(df["developer"] != "") & (df["publisher"] != "")]

# --- Step 2: Build KPIs ---
devs_per_pub = df.groupby("publisher")["developer"].nunique().reset_index(name="unique_developers")
games_per_dev = df.groupby("developer")["app_id"].nunique().reset_index(name="games_per_dev")
games_per_pub = df.groupby("publisher")["app_id"].nunique().reset_index(name="games_per_pub")

# Merge KPIs for publishers
publisher_kpis = devs_per_pub.merge(games_per_pub, on="publisher")
# Merge KPIs for developers
developer_kpis = games_per_dev.copy()


# --- Step 3: Multi-KPI Clustering Function ---
def run_multi_kpi_clustering(kpi_df, entity_col, features, name_prefix, n_clusters=3):
    # Standardize features
    X = kpi_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kpi_df["cluster"] = kmeans.fit_predict(X_scaled)

    # Save CSV
    csv_path = os.path.join(output_dir, f"{name_prefix}.csv")
    kpi_df.to_csv(csv_path, index=False)

    # Scatter plot for first two features
    plt.figure(figsize=(10, 6))
    for c in kpi_df["cluster"].unique():
        subset = kpi_df[kpi_df["cluster"] == c]
        plt.scatter(subset[features[0]], subset[features[1]], label=f"Cluster {c}", alpha=0.6)

    # Add centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c="black", marker="X", s=200, label="Centroids")

    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(f"KMeans Clustering: {name_prefix}")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{name_prefix}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"✅ Saved CSV: {csv_path}")
    print(f"✅ Saved Plot: {plot_path}")

    # Cluster summary
    print(f"\n=== Clustering Summary for {name_prefix} ===")
    print(kpi_df.groupby("cluster")[features].mean())
    for c in kpi_df["cluster"].unique():
        examples = kpi_df[kpi_df["cluster"] == c][entity_col].head(5).tolist()
        print(f"Cluster {c} examples: {examples}")

    return kpi_df


# --- Step 4: Run clustering ---
publisher_features = ["unique_developers", "games_per_pub"]
developer_features = ["games_per_dev"]

publisher_clusters = run_multi_kpi_clustering(publisher_kpis, "publisher", publisher_features, "publisher_clusters")
developer_clusters = run_multi_kpi_clustering(developer_kpis, "developer", developer_features, "developer_clusters")

print("\n🎯 All clustering results and plots saved into:", output_dir)
