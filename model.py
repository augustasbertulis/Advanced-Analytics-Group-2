# run_all_kmeans.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from datetime import datetime
from paths import PROCESSED_DATA_DIR, DATA_DIR
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# --------------------------- Config ---------------------------
IMPORT_PATH = PROCESSED_DATA_DIR / "combined_clean.csv"
KPI_CSV_PATH = PROCESSED_DATA_DIR / "publisher_kpis.csv"
out = PROCESSED_DATA_DIR

MODEL_CONFIGS = {
    "model1": {
        "groupby": "publisher",
        "features": ["revenue_proxy", "price_in_eur", "owners_avg"],
        "k": 4,
        "weights": {"revenue_proxy": 0.5, "owners_avg": 0.4, "price_in_eur": -0.1},
        "top_n_clusters": 1,
        "agg": {"revenue_proxy": "mean", "price_in_eur": "mean", "owners_avg": "mean"},
        "source_cols": ["app_id", "name", "publisher", "revenue_proxy", "price_in_eur", "owners_avg"],
    },
    "model2": {
        "groupby": "publisher",
        "features": ["positive_score", "positive", "total"],
        "k": 4,
        "weights": {"positive_score": 0.5, "total": 0.5, "positive": 0.0},
        "top_n_clusters": 1,
        "agg": {"positive_score": "mean", "positive": "mean", "total": "mean"},
        "source_cols": ["app_id", "name", "publisher", "total", "positive", "positive_score"],
        "fillna": {"positive": 0, "total": 0},
    },
    "model3": {
        "groupby": "publisher",
        "features": ["active_engagement_score", "concurrent_users_yesterday", "owners_avg"],
        "k": 4,
        "weights": {"active_engagement_score": 0.5, "concurrent_users_yesterday": 0.3, "owners_avg": 0.2},
        "top_n_clusters": 1,
        "agg": {
            "active_engagement_score": "mean",
            "owners_avg": "mean",
            "concurrent_users_yesterday": "mean",
        },
        "source_cols": ["app_id", "publisher", "owners_avg", "active_engagement_score", "concurrent_users_yesterday"],
    },
    "model4": {
        "groupby": "publisher",
        "features": ["growth_potential", "concurrent_users_yesterday", "owners_avg"],
        "k": 3,
        "weights": {"growth_potential": 0.6, "concurrent_users_yesterday": 0.3, "owners_avg": 0.1},
        "top_n_clusters": 1,
        "agg": {
            "growth_potential": "mean",
            "owners_avg": "mean",
            "concurrent_users_yesterday": "mean",
        },
        "source_cols": None,  # constructed after feature engineering
    },
    "model5": {
        "groupby": "publisher",
        "features": ["num_developers", "num_languages", "num_games"],
        "k": 3,
        "weights": {"num_developers": -0.4, "num_languages": -0.3, "num_games": 0.3},
        "top_n_clusters": 1,
        "agg": {"num_developers": "mean", "num_languages": "mean", "num_games": "mean"},
        "source_cols": ["publisher", "num_developers", "num_languages", "num_games"],
    },
}

MODEL_VOTE_WEIGHTS = {
    "model1": 0.4,
    "model2": 0.15,
    "model3": 0.3,
    "model4": 0.15,
    "model5": 0.1,
}

# ------------------------ Helpers ------------------------
def minmax_weighted_score(summary_df: pd.DataFrame, weights: dict) -> pd.Series:
    if summary_df.empty:
        return pd.Series(dtype=float)
    common_cols = [c for c in weights.keys() if c in summary_df.columns]
    mm = MinMaxScaler()
    scaled = mm.fit_transform(summary_df[common_cols])
    scaled_df = pd.DataFrame(scaled, index=summary_df.index, columns=common_cols)
    score = sum(scaled_df[col] * w for col, w in weights.items() if col in scaled_df.columns)
    return score

def elbow_curve_all(models_data: dict, k_max=10, out_dir=None):
    """
    Plot elbow curves (SSE vs k) for all models in one figure (stacked/overlayed).
    models_data: dict {model_key: X_scaled}
    """
    plt.figure(figsize=(8,6))

    Ks = range(1, k_max+1)
    for model_key, X_scaled in models_data.items():
        sse = []
        for k in Ks:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            sse.append(kmeans.inertia_)
        plt.plot(Ks, sse, marker='o', label=model_key)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("SSE (Inertia)")
    plt.title("Elbow Curves – All Models")
    plt.legend()
    plt.grid(True)
    if out_dir:
        plt.savefig(out_dir / "elbow_curves_all.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def run_kmeans_model(df_grouped: pd.DataFrame, features: list, k: int, label_name: str) -> pd.DataFrame:
    X = df_grouped[features].to_numpy()
    X = np.nan_to_num(X, nan=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_grouped[label_name] = kmeans.fit_predict(X_scaled)
    return df_grouped

def best_clusters_by_score(df_grouped: pd.DataFrame, cluster_col: str, features: list, weights: dict, top_n: int = 1):
    summary = df_grouped.groupby(cluster_col)[features].mean()
    summary["score"] = minmax_weighted_score(summary[features], weights)
    top_clusters = summary["score"].sort_values(ascending=False).head(top_n).index
    return top_clusters, summary

def load_risk_data_from_kpis(df: pd.DataFrame, kpis_csv_path: str):
    kpis_df = pd.read_csv(kpis_csv_path)
    required_cols = ["publisher", "unique_developers", "games_per_pub", "languages_per_pub"]
    missing_cols = [c for c in required_cols if c not in kpis_df.columns]
    if missing_cols:
        raise ValueError(f"KPIs CSV missing columns: {missing_cols}")
    kpis_df = kpis_df.rename(columns={
        "unique_developers": "num_developers",
        "games_per_pub": "num_games",
        "languages_per_pub": "num_languages"
    })
    df = df.merge(kpis_df, on="publisher", how="left")
    df[["num_developers", "num_languages", "num_games"]] = df[["num_developers", "num_languages", "num_games"]].fillna(0)
    return df

# ------------------------ Main function ------------------------
def run_all_kmeans(import_path: str = IMPORT_PATH,
                   kpis_csv_path: str = KPI_CSV_PATH,
                   model_configs: dict = MODEL_CONFIGS,
                   vote_weights: dict = MODEL_VOTE_WEIGHTS):
    df = pd.read_csv(import_path, low_memory=False)

    # Load risk data for model5 from KPI CSV
    df = load_risk_data_from_kpis(df, kpis_csv_path)

    # Feature engineering
    if "owners_avg" not in df.columns:
        df["owners_avg"] = (df["owners_min"] + df["owners_max"]) / 2
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    else:
        df["release_date"] = pd.NaT
    today = pd.Timestamp("2025-08-25")
    df["days_since_release"] = (today - df["release_date"]).dt.days.replace(0, 1)
    if {"owners_max", "owners_min"}.issubset(df.columns):
        df["growth_potential"] = (df["owners_max"] - df["owners_min"]) / df["days_since_release"]
    else:
        df["growth_potential"] = np.nan

    fill_cols = ["revenue_proxy", "price_in_eur", "owners_avg", "positive_score", "positive",
                 "total", "active_engagement_score", "concurrent_users_yesterday", "growth_potential"]
    df[fill_cols] = df[fill_cols].fillna(0)

    MODEL_CONFIGS["model4"]["source_cols"] = ["publisher", "growth_potential", "owners_avg", "concurrent_users_yesterday"]

    # ------------------------ Run models ------------------------
    per_model_flags = {}
    per_model_summaries = {}
    per_model_cluster_cols = {}
    robustness_stats = {}
    models_data = {}

    for model_key, cfg in MODEL_CONFIGS.items():
        group_col = cfg["groupby"]
        feats = cfg["features"]
        k = cfg["k"]
        weights = cfg["weights"]
        top_n = cfg.get("top_n_clusters", 1)

        sub = df[cfg["source_cols"]].copy() if cfg.get("source_cols") else df[[group_col] + feats].copy()
        if cfg.get("fillna"):
            sub = sub.fillna(cfg["fillna"])
        grouped = sub.groupby(group_col, as_index=False).agg(cfg.get("agg", {f: "mean" for f in feats}))
        cluster_col = f"cluster_{model_key}"
        grouped = run_kmeans_model(grouped, feats, k, cluster_col)
        top_clusters, summary = best_clusters_by_score(grouped, cluster_col, feats, weights, top_n=top_n)
        grouped[f"is_best_{model_key}"] = grouped[cluster_col].isin(top_clusters)

        per_model_flags[model_key] = grouped[[group_col, f"is_best_{model_key}"]]
        per_model_summaries[model_key] = summary
        per_model_cluster_cols[model_key] = (grouped[[group_col, cluster_col]], feats)

        # Robustness check
        X = np.nan_to_num(grouped[feats].to_numpy(), nan=0.0)
        X_scaled = StandardScaler().fit_transform(X)
        labels = grouped[cluster_col]
        robustness_stats[model_key] = {
            "silhouette": silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else None,
        }
        models_data[model_key] = X_scaled

        # score clusters & mark best
        top_clusters, summary = best_clusters_by_score(grouped, cluster_col, feats, weights, top_n=top_n)
        grouped[f"is_best_{model_key}"] = grouped[cluster_col].isin(top_clusters)

        # stash results
        per_model_flags[model_key] = grouped[[group_col, f"is_best_{model_key}"]]
        per_model_summaries[model_key] = summary
        per_model_cluster_cols[model_key] = (grouped[[group_col, cluster_col]], feats)

    # ---- Plot combined elbow curves once after all models ----
    elbow_curve_all(models_data, k_max=10, out_dir=out)

    # ------------------------ Merge & voting ------------------------
    merged = None
    for model_key, flags in per_model_flags.items():
        merged = flags.copy() if merged is None else merged.merge(flags, on="publisher", how="outer")
    for model_key in MODEL_CONFIGS.keys():
        col = f"is_best_{model_key}"
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)
    best_cols = [f"is_best_{m}" for m in MODEL_CONFIGS.keys()]
    merged["votes"] = merged[best_cols].sum(axis=1)
    if MODEL_VOTE_WEIGHTS:
        merged["weighted_votes"] = sum(
            merged[f"is_best_{m}"].astype(float) * MODEL_VOTE_WEIGHTS.get(m, 1.0) for m in MODEL_CONFIGS.keys()
        )
    sort_cols = ["weighted_votes", "votes"] if "weighted_votes" in merged.columns else ["votes"]
    ranked = merged.sort_values(sort_cols + ["publisher"], ascending=[False]*len(sort_cols) + [True])

    # ------------------------ Output ------------------------
    final_pick = ranked.head(100)
    final_pick.to_excel(out / "publisher_ranked_consensus.xlsx", index=False)
    ranked.to_csv(out / "publisher_ranked_consensus.csv", index=False)
    for m, summary in per_model_summaries.items():
        summary.to_csv(out / f"{m}_cluster_summary.csv")

    print("\n=== Model Robustness Stats ===")
    for m, stats in robustness_stats.items():
        print(f"{m}: {stats}")

    return ranked, per_model_summaries, robustness_stats

def main():
    run_all_kmeans()

if __name__ == "__main__":
    main()