import pandas as pd
import matplotlib.pyplot as plt
from paths import PROCESSED_DATA_DIR


def main():
    # Load the aggregated publisher table
    df = pd.read_csv(PROCESSED_DATA_DIR / "attributes.csv")

    # --- Clean numeric columns (remove commas, force float) ---
    for col in [
        "Recommendations", "Price [€]", "Owner Avg", "Revenue Proxy",
        "Positive Reviews", "Total Reviews", "Positive Score",
        "Concurrent Users Yesterday", "Active Engagement Score", "Growth Potential"
    ]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Scatter plot: Growth Potential vs Price ---
    plt.figure(figsize=(12, 7))

    plt.scatter(
        df["Growth Potential"],
        df["Price [€]"],
        s=df["Recommendations"] / 200,   # bubble size
        alpha=0.6,
        color="orange",
        edgecolor="k"
    )

    # Annotate only publishers with strong metrics (to avoid clutter)
    for i, row in df.iterrows():
        if row["Recommendations"] > 20000 or row["Growth Potential"] > 5000:
            plt.text(row["Growth Potential"], row["Price [€]"], row["Publisher"], fontsize=8)

    plt.xscale("log")   # Growth Potential is often very skewed
    plt.xlabel("Growth Potential (log scale)")
    plt.ylabel("Average Price (€)")
    plt.title("Publishers: Growth Potential vs Price (Bubble = Recommendations)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.show()


if __name__ == "__main__":
    main()
