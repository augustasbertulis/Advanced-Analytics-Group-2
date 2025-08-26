# sensitivity_model_weights_csv.py
import numpy as np
import pandas as pd
from itertools import product
from paths import PROCESSED_DATA_DIR
from model import (
    run_kmeans_model,
    best_clusters_by_score,
    MODEL_CONFIGS,
    IMPORT_PATH,
    KPI_CSV_PATH,
    load_risk_data_from_kpis,
)

# ---------------- CONFIG ----------------
OUT_DIR = PROCESSED_DATA_DIR
OUT_DIR.mkdir(exist_ok=True)

WEIGHT_GRID = [-0.5, -0.25, 0, 0.25, 0.5]  # coarse grid for model feature weights

# ---------------- HELPERS ----------------
def generate_weight_combinations(features, grid=WEIGHT_GRID):
    """Generate all combinations of weights for given features."""
    for combo in product(grid, repeat=len(features)):
        yield dict(zip(features, combo))

# ---------------- MAIN ----------------
def run_model_weight_sensitivity():
    # ---------- Load and preprocess once ----------
    df = pd.read_csv(IMPORT_PATH, low_memory=False)
    df = load_risk_data_from_kpis(df, KPI_CSV_PATH)

    if "owners_avg" not in df.columns:
        df["owners_avg"] = (df["owners_min"] + df["owners_max"]) / 2
    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    today = pd.Timestamp("2025-08-25")
    df["days_since_release"] = (today - df["release_date"]).dt.days.replace(0, 1)
    if {"owners_max", "owners_min"}.issubset(df.columns):
        df["growth_potential"] = (df["owners_max"] - df["owners_min"]) / df["days_since_release"]

    fill_cols = ["revenue_proxy", "price_in_eur", "owners_avg", "positive_score", "positive",
                 "total", "active_engagement_score", "concurrent_users_yesterday", "growth_potential"]
    df[fill_cols] = df[fill_cols].fillna(0)

    MODEL_CONFIGS["model4"]["source_cols"] = ["publisher", "growth_potential", "owners_avg", "concurrent_users_yesterday"]

    # ---------- Cache clustering results ----------
    cluster_cache = {}
    for model_key, cfg in MODEL_CONFIGS.items():
        group_col = cfg["groupby"]
        feats = cfg["features"]
        k = cfg["k"]

        sub = df[cfg["source_cols"]].copy() if cfg.get("source_cols") else df[[group_col] + feats].copy()
        if cfg.get("fillna"):
            sub = sub.fillna(cfg["fillna"])
        grouped = sub.groupby(group_col, as_index=False).agg(cfg.get("agg", {f: "mean" for f in feats}))
        cluster_col = f"cluster_{model_key}"
        grouped = run_kmeans_model(grouped, feats, k, cluster_col)
        cluster_cache[model_key] = (grouped, cluster_col, feats, group_col)

    # ---------- Sensitivity analysis ----------
    best_weights_records = []

    for model_key, (grouped, cluster_col, feats, group_col) in cluster_cache.items():
        print(f"Running sensitivity analysis for {model_key} with features {feats}")
        best_score = -np.inf
        best_weight = None

        for weights in generate_weight_combinations(feats):
            top_clusters, summary = best_clusters_by_score(grouped, cluster_col, feats, weights, top_n=1)

            # Extract numeric score (assumes summary is a DataFrame with 'score' column)
            if isinstance(summary, pd.DataFrame) and 'score' in summary.columns:
                score = summary['score'].iloc[0]
            else:
                score = float(summary)  # fallback in case summary is a number

            if score > best_score:
                best_score = score
                best_weight = weights

        best_weights_records.append({
            "model": model_key,
            **best_weight
        })
        print(f"Best weights for {model_key}: {best_weight} with score {best_score}")

    # ---------- Save results to CSV ----------
    best_weights_df = pd.DataFrame(best_weights_records)
    best_weights_df.to_csv(OUT_DIR / "best_model_weights.csv", index=False)
    print("\nSaved best weights for all models to CSV.")

if __name__ == "__main__":
    run_model_weight_sensitivity()
