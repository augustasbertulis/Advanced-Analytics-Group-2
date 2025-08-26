# pseudo_markowitz.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from paths import PROCESSED_DATA_DIR
from src.processing.model import run_all_kmeans, load_risk_data_from_kpis, KPI_CSV_PATH, IMPORT_PATH


def build_pseudo_markowitz(
        ranked_df: pd.DataFrame,
        full_df: pd.DataFrame,
        growth_col: str = "growth_potential",
        risk_factors: list = None,
        vote_col: str = "votes",
        gamma: float = 1.0,
        min_weight: float = 0.0,
        normalize_returns: bool = True,
):
    if risk_factors is None:
        risk_factors = ["num_developers", "num_languages", "num_games"]

    # Filter publishers with votes
    sel = ranked_df[ranked_df[vote_col].notnull() & (ranked_df[vote_col] > 0)].copy()
    publishers = sel["publisher"].unique().tolist()

    pub_feats = (
        full_df[["publisher"] + [growth_col] + risk_factors]
        .groupby("publisher")
        .mean()
        .loc[publishers]
    )
    pub_feats = pub_feats[pub_feats[growth_col].notnull()]
    if pub_feats.empty:
        raise ValueError("No publishers found with votes > 0 and valid growth_potential.")

    # Expected growth
    mu = pub_feats[growth_col].values.astype(float)
    if normalize_returns:
        mu = MinMaxScaler().fit_transform(mu.reshape(-1, 1)).ravel()

    # Standardize risk factors
    X = pub_feats[risk_factors].fillna(0.0).values.astype(float)
    X_std = StandardScaler().fit_transform(X)

    # Compute risk proxy: higher X_std means lower risk, so invert
    risk_proxy_raw = np.sqrt(np.sum(X_std ** 2, axis=1))
    risk_proxy = 1.0 - MinMaxScaler().fit_transform(risk_proxy_raw.reshape(-1, 1)).ravel()

    # Adjust returns by risk (higher risk_proxy → higher effective return)
    adjusted_mu = mu * (1 + gamma * risk_proxy)

    # Compute unconstrained weights
    w_uncon = adjusted_mu
    w_clip = np.clip(w_uncon, min_weight, None)
    if w_clip.sum() <= 0:
        w = np.repeat(1.0 / len(w_clip), len(w_clip))
    else:
        w = w_clip / w_clip.sum()

    # Portfolio metrics
    port_return = w.dot(mu)
    port_risk = np.sqrt(np.sum((w * (1 - risk_proxy)) ** 2))  # inverted back for risk calculation
    sharpe_proxy = port_return / (port_risk + 1e-9)

    out = pub_feats.copy()
    out["mu"] = mu
    out["risk_proxy"] = risk_proxy
    out["opt_weight"] = w
    out = out.reset_index().rename(columns={"index": "publisher"})

    stats = {
        "n_assets": len(w),
        "portfolio_return": float(port_return),
        "portfolio_risk": float(port_risk),
        "sharpe_proxy": float(sharpe_proxy),
        "gamma": gamma,
        "min_weight": min_weight,
    }

    return out, stats


def plot_perturbed_frontier(portfolio_df, risk_factors, n_perturb=200, perturb_scale=0.05):
    """
    Plots pseudo-optimal portfolio and perturbed portfolios around it.

    :param n_perturb: number of perturbed portfolios to generate
    :param perturb_scale: maximum fraction of perturbation per weight
    """
    # Standardize risk factors
    X = portfolio_df[risk_factors].fillna(0.0).values.astype(float)
    X_std = StandardScaler().fit_transform(X)
    portfolio_df["risk_agg"] = np.sqrt(np.sum(X_std ** 2, axis=1))

    plt.figure(figsize=(10, 6))

    w_opt = portfolio_df["opt_weight"].values
    n_assets = len(w_opt)
    perturbed_returns = []
    perturbed_risks = []

    np.random.seed(42)

    for _ in range(n_perturb):
        # Perturb each weight slightly
        perturbation = (np.random.rand(n_assets) - 0.5) * 2 * perturb_scale
        w_new = w_opt + perturbation
        w_new = np.clip(w_new, 0.0, None)
        if w_new.sum() == 0:
            w_new = np.repeat(1.0 / n_assets, n_assets)
        else:
            w_new /= w_new.sum()

        # Compute portfolio metrics
        port_ret = (portfolio_df["mu"] * w_new).sum()
        port_risk = np.sqrt(w_new.T @ np.diag(portfolio_df["risk_agg"] ** 2) @ w_new)
        perturbed_returns.append(port_ret)
        perturbed_risks.append(port_risk)

    # Plot perturbed portfolios (swap x and y)
    plt.scatter(perturbed_risks, perturbed_returns, color="gray", alpha=0.4, s=30, label="Perturbed Portfolios")

    # Plot pseudo-Markowitz portfolio (swap x and y)
    port_return = (portfolio_df["mu"] * w_opt).sum()
    port_risk = np.sqrt(w_opt.T @ np.diag(portfolio_df["risk_agg"] ** 2) @ w_opt)
    plt.scatter(port_risk, port_return, color="red", marker="*", s=200, label="Pseudo-Markowitz Portfolio")

    # Set axes limits
    plt.xlim(0.15, 0.40)  # x-axis: Aggregate Risk
    plt.ylim(0.4, 0.65)   # y-axis: Growth Potential

    plt.xlabel("Aggregate Risk")
    plt.ylabel("Growth Potential")
    plt.title("Pseudo-Markowitz Portfolio vs Nearby Perturbed Portfolios")
    plt.legend()
    plt.grid(True)
    plt.show()




def run_pseudo_markowitz():
    ranked, per_model_summaries = run_all_kmeans(
        import_path=IMPORT_PATH,
        kpis_csv_path=KPI_CSV_PATH,
    )

    df = pd.read_csv(IMPORT_PATH, low_memory=False)
    df = load_risk_data_from_kpis(df, KPI_CSV_PATH)

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

    portfolio_df, stats = build_pseudo_markowitz(ranked, df, vote_col="votes")

    out_csv = PROCESSED_DATA_DIR / "publisher_markowitz_portfolio.csv"
    out_xlsx = PROCESSED_DATA_DIR / "publisher_markowitz_portfolio.xlsx"
    out_json = PROCESSED_DATA_DIR / "portfolio_stats.json"

    portfolio_df.to_csv(out_csv, index=False)
    portfolio_df.to_excel(out_xlsx, index=False)
    pd.Series(stats).to_json(out_json)

    print("Pseudo-Markowitz portfolio saved.")
    print(stats)

    # Plot perturbed portfolios
    risk_factors = ["num_developers", "num_languages", "num_games"]
    plot_perturbed_frontier(portfolio_df, risk_factors, n_perturb=200, perturb_scale=0.05)


if __name__ == "__main__":
    run_pseudo_markowitz()
