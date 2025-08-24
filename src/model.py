# run_two_kmeans_models.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from datetime import datetime
from paths import PROCESSED_DATA_DIR

# Example: saving processed output
IMPORT_PATH = PROCESSED_DATA_DIR / "combined_clean.csv"
out = PROCESSED_DATA_DIR


# Per-model config
# - features: columns used for k-means (after aggregation)
# - groupby: "publisher" (kept for flexibility)
# - k: number of clusters
# - weights: dict for summary scoring (after MinMax per-feature in the summary)
# - top_n_clusters: how many top-scoring clusters to mark as "best" per model
# - agg: how to aggregate per publisher (defaults to mean if omitted)
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
        # Your KPIs model: engagement, concurrents, owners
        "groupby": "publisher",
        "features": ["active_engagement_score", "concurrent_users_yesterday", "owners_avg"],
        "k": 4,
        # >>> New weights applied here <<<
        # Emphasize engagement & concurrents; owners is useful but slightly less predictive of activity.
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
        # Growth-potential model: growth per day + concurrents + owners
        "groupby": "publisher",
        "features": ["growth_potential", "concurrent_users_yesterday", "owners_avg"],
        "k": 3,
        # >>> New weights applied here <<<
        # Emphasize growth_potential; concurrents matter; owners provides base scale.
        "weights": {"growth_potential": 0.6, "concurrent_users_yesterday": 0.3, "owners_avg": 0.1},
        "top_n_clusters": 1,
        "agg": {
            "growth_potential": "mean",
            "owners_avg": "mean",
            "concurrent_users_yesterday": "mean",
        },
        # source_cols built from base df (we compute growth_potential below)
        "source_cols": None,  # constructed after feature engineering
    },
}

# Optional **model-level** weights (for weighted voting across models)
# If you don't want weighted voting, leave as None.
MODEL_VOTE_WEIGHTS = {
    "model1": 1.0,
    "model2": 1.0,
    "model3": 1.0,
    "model4": 1.0,
}

# ------------------------ Helpers ------------------------
def minmax_weighted_score(summary_df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Scale each column in the cluster summary to [0,1] with MinMax, then
    compute weighted sum. Negative weights allowed.
    """
    if summary_df.empty:
        return pd.Series(dtype=float)

    common_cols = [c for c in weights.keys() if c in summary_df.columns]
    if not common_cols:
        raise ValueError("No overlapping feature columns between summary and weights.")

    mm = MinMaxScaler()
    scaled = mm.fit_transform(summary_df[common_cols])
    scaled_df = pd.DataFrame(scaled, index=summary_df.index, columns=common_cols)

    score = sum(scaled_df[col] * w for col, w in weights.items() if col in scaled_df.columns)
    return score

def run_kmeans_model(df_grouped: pd.DataFrame, features: list, k: int, label_name: str) -> pd.DataFrame:
    """
    Standardize features and assign k-means clusters to the grouped dataframe.
    """
    X = df_grouped[features].to_numpy()
    # Handle any all-NaN columns gracefully
    X = np.nan_to_num(X, nan=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_grouped[label_name] = kmeans.fit_predict(X_scaled)
    return df_grouped

def best_clusters_by_score(df_grouped: pd.DataFrame, cluster_col: str, features: list, weights: dict, top_n: int = 1):
    """
    Compute per-cluster summary (means), score with MinMax+weights, return the top cluster labels.
    """
    summary = df_grouped.groupby(cluster_col)[features].mean()
    summary["score"] = minmax_weighted_score(summary[features], weights)
    top_clusters = summary["score"].sort_values(ascending=False).head(top_n).index
    return top_clusters, summary
def run_all_kmeans(import_path: str = IMPORT_PATH,
                          model_configs: dict = MODEL_CONFIGS,
                          vote_weights: dict = MODEL_VOTE_WEIGHTS,
                          excel_out: str = "data/clean data/publisher_ranked_consensus.xlsx"):
# ------------------------ Load & feature engineering ------------------------
    df = pd.read_csv(IMPORT_PATH, low_memory=False)

    # owners_avg early (used by multiple models)
    if "owners_avg" not in df.columns:
        df["owners_avg"] = (df["owners_min"] + df["owners_max"]) / 2

    # release_date → growth-related features
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    else:
        df["release_date"] = pd.NaT

    today = pd.Timestamp(datetime.today().date())
    df["days_since_release"] = (today - df["release_date"]).dt.days
    df["days_since_release"] = df["days_since_release"].replace(0, 1)  # avoid div by zero

    # growth_potential: (owners_max - owners_min) / days_since_release
    if {"owners_max", "owners_min"}.issubset(df.columns):
        df["growth_potential"] = (df["owners_max"] - df["owners_min"]) / df["days_since_release"]
    else:
        df["growth_potential"] = np.nan

    # Null safety
    df = df.fillna({
        "revenue_proxy": 0,
        "price_in_eur": 0,
        "owners_avg": 0,
        "positive_score": 0,
        "positive": 0,
        "total": 0,
        "active_engagement_score": 0,
        "concurrent_users_yesterday": 0,
        "growth_potential": 0,
    })

    # Build source_cols for model4 after feature engineering
    MODEL_CONFIGS["model4"]["source_cols"] = ["publisher", "growth_potential", "owners_avg", "concurrent_users_yesterday"]

    # ------------------------ Run models ------------------------
    per_model_flags = {}           # publisher → is_best flag per model
    per_model_summaries = {}       # cluster summaries incl. score per model
    per_model_cluster_cols = {}    # cluster column names for each model

    for model_key, cfg in MODEL_CONFIGS.items():
        group_col = cfg["groupby"]
        feats = cfg["features"]
        k = cfg["k"]
        weights = cfg["weights"]
        top_n = cfg.get("top_n_clusters", 1)

        # subset + fillna per-model if requested
        if cfg.get("source_cols"):
            sub = df[cfg["source_cols"]].copy()
        else:
            # fallback: keep group_col + features
            cols = [group_col] + [c for c in feats if c in df.columns]
            sub = df[cols].copy()

        # model-specific fillna if provided
        if cfg.get("fillna"):
            sub = sub.fillna(cfg["fillna"])

        # aggregate per publisher (or group_col)
        agg = cfg.get("agg")
        if agg is None:
            agg = {f: "mean" for f in feats}
        grouped = sub.groupby(group_col, as_index=False).agg(agg)

        # run k-means
        cluster_col = f"cluster_{model_key}"
        grouped = run_kmeans_model(grouped, feats, k, cluster_col)

        # score clusters & mark best
        top_clusters, summary = best_clusters_by_score(grouped, cluster_col, feats, weights, top_n=top_n)
        grouped[f"is_best_{model_key}"] = grouped[cluster_col].isin(top_clusters)

        # stash results
        per_model_flags[model_key] = grouped[[group_col, f"is_best_{model_key}"]]
        per_model_summaries[model_key] = summary
        per_model_cluster_cols[model_key] = (grouped[[group_col, cluster_col]], feats)

    # ------------------------ Merge & voting ------------------------
    # Outer-join best flags across all models
    merged = None
    for model_key, flags in per_model_flags.items():
        if merged is None:
            merged = flags.copy()
        else:
            merged = merged.merge(flags, on="publisher", how="outer")

    # Coerce boolean (fills missing with False)
    for model_key in MODEL_CONFIGS.keys():
        col = f"is_best_{model_key}"
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)

    # Unweighted votes
    best_cols = [f"is_best_{m}" for m in MODEL_CONFIGS.keys()]
    merged["votes"] = merged[best_cols].sum(axis=1)

    # Optional weighted votes
    if MODEL_VOTE_WEIGHTS:
        merged["weighted_votes"] = sum(
            merged[f"is_best_{m}"].astype(float) * MODEL_VOTE_WEIGHTS.get(m, 1.0)
            for m in MODEL_CONFIGS.keys()
        )

    # Sort by consensus
    sort_cols = ["votes"]
    if "weighted_votes" in merged.columns:
        sort_cols = ["weighted_votes", "votes"]
    ranked = merged.sort_values(sort_cols, ascending=False)

    # ------------------------ Output ------------------------
    pd.set_option("display.max_rows", 50)
    final_pick = ranked.head(20)
    final_pick.to_excel("data/clean data/publisher_ranked_consensus.xlsx", index=False)

    # If you want CSV outputs
    ranked.to_csv("out/publisher_ranked_consensus.csv", index=False)
    for m, (summary) in per_model_summaries.items():
        summary.to_csv(f"out/{m}_cluster_summary.csv")

    return ranked, per_model_summaries
def main():
    run_all_kmeans()
if __name__ == "__main__":
    main()

