# pseudo_markowitz.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from paths import PROCESSED_DATA_DIR
from model import run_all_kmeans, load_risk_data_from_kpis, KPI_CSV_PATH, IMPORT_PATH

def build_pseudo_markowitz(
    ranked_df: pd.DataFrame,
    full_df: pd.DataFrame,
    growth_col: str = "growth_potential",
    risk_factors: list = None,
    vote_col: str = "votes",
    idiosyncratic_var: float = 1e-4,
    gamma: float = 1.0,
    min_weight: float = 0.0,
    normalize_returns: bool = True,
):
    """
    Build a pseudo-Markowitz portfolio using snapshot data.
    Expected return = growth_potential
    Risk factors = model5 exposures
    Selection = publishers with votes > 0
    """
    if risk_factors is None:
        risk_factors = ["num_developers", "num_languages", "num_games"]

    # --- filter publishers by votes ---
    sel = ranked_df[ranked_df[vote_col].notnull() & (ranked_df[vote_col] > 0)].copy()
    publishers = sel["publisher"].unique().tolist()

    # --- aggregate features at publisher level ---
    pub_feats = (
        full_df[["publisher"] + [growth_col] + risk_factors]
        .groupby("publisher")
        .mean()
        .loc[publishers]
    )
    pub_feats = pub_feats[pub_feats[growth_col].notnull()]

    if pub_feats.empty:
        raise ValueError("No publishers found with votes > 0 and valid growth_potential.")

    # --- expected returns (mu) ---
    mu = pub_feats[growth_col].values.astype(float)
    if normalize_returns:
        mu = MinMaxScaler().fit_transform(mu.reshape(-1, 1)).ravel()

    # --- factor exposures ---
    X = pub_feats[risk_factors].fillna(0.0).values.astype(float)
    X = StandardScaler().fit_transform(X)

    # --- covariance construction ---
    cov_f = np.cov(X, rowvar=False)
    asset_cov = X @ cov_f @ X.T
    asset_cov += np.eye(asset_cov.shape[0]) * idiosyncratic_var
    vol_proxy = np.sqrt(np.diag(asset_cov))

    # --- unconstrained Markowitz solution ---
    try:
        inv_cov = np.linalg.inv(asset_cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(asset_cov)
    w_uncon = inv_cov @ mu / gamma

    # --- enforce non-negativity + normalize ---
    w_clip = np.clip(w_uncon, min_weight, None)
    if w_clip.sum() <= 0:
        w = np.repeat(1.0 / len(w_clip), len(w_clip))
    else:
        w = w_clip / w_clip.sum()

    # --- portfolio stats ---
    port_return = w.dot(mu)
    port_var = w.T @ asset_cov @ w
    port_vol = np.sqrt(port_var)
    sharpe_proxy = port_return / (port_vol + 1e-9)

    # --- output DataFrame ---
    out = pub_feats.copy()
    out["mu"] = mu
    out["vol_proxy"] = vol_proxy
    out["opt_weight"] = w
    out = out.reset_index().rename(columns={"index": "publisher"})

    stats = {
        "n_assets": len(w),
        "portfolio_return": float(port_return),
        "portfolio_vol": float(port_vol),
        "sharpe_proxy": float(sharpe_proxy),
        "gamma": gamma,
        "min_weight": min_weight,
    }

    return out, stats


def run_pseudo_markowitz():
    # Run consensus ranking first
    ranked, per_model_summaries = run_all_kmeans(
        import_path=IMPORT_PATH,
        kpis_csv_path=KPI_CSV_PATH,
    )

    # Load full raw data again for feature access
    df = pd.read_csv(IMPORT_PATH, low_memory=False)
    df = load_risk_data_from_kpis(df, KPI_CSV_PATH)

    # add growth_potential if not already there
    if "owners_avg" not in df.columns and {"owners_min", "owners_max"}.issubset(df.columns):
        df["owners_avg"] = (df["owners_min"] + df["owners_max"]) / 2
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    else:
        df["release_date"] = pd.NaT
    today = pd.Timestamp.today()
    df["days_since_release"] = (today - df["release_date"]).dt.days.replace(0, 1)
    if {"owners_max", "owners_min"}.issubset(df.columns):
        df["growth_potential"] = (df["owners_max"] - df["owners_min"]) / df["days_since_release"]
    else:
        df["growth_potential"] = np.nan
    df["growth_potential"] = df["growth_potential"].fillna(0)

    # Build pseudo-Markowitz
    portfolio_df, stats = build_pseudo_markowitz(ranked, df, vote_col="votes")

    # Save outputs
    out_csv = PROCESSED_DATA_DIR / "publisher_markowitz_portfolio.csv"
    out_xlsx = PROCESSED_DATA_DIR / "publisher_markowitz_portfolio.xlsx"
    out_json = PROCESSED_DATA_DIR / "portfolio_stats.json"

    portfolio_df.to_csv(out_csv, index=False)
    portfolio_df.to_excel(out_xlsx, index=False)
    pd.Series(stats).to_json(out_json)

    print("Pseudo-Markowitz portfolio saved.")
    print(stats)


if __name__ == "__main__":
    run_pseudo_markowitz()
