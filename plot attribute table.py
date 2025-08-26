import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from paths import PROCESSED_DATA_DIR


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric columns: remove commas, convert to numeric."""
    for col in [
        "Recommendations", "Price [€]", "Owner Avg", "Revenue Proxy",
        "Positive Reviews", "Total Reviews", "Positive Score",
        "Concurrent Users Yesterday", "Active Engagement Score", "Growth Potential"
    ]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def plot_scatter(df: pd.DataFrame):
    """Scatter plot: Growth Potential vs Price with forced publisher labels."""
    plt.figure(figsize=(12, 7))
    plt.scatter(
        df["Growth Potential"],
        df["Price [€]"],
        s=df["Recommendations"] / 200,
        alpha=0.6,
        color="orange"
    )

    # Publishers to always show
    force_labels = {
        "Game Science",
        "Eleventh Hour Games",
        "Amazon Games",
        "Smartly Dressed Games",
        "Bungie",
        "Endnight Games Ltd",
        "Kinetic Games",
        "KRAFTON, Inc.",
        "Curve Animation",
        "OPNeon Games"
    }

    # Add labels
    texts = []
    for i, row in df.iterrows():
        if row["Publisher"] in force_labels:
            texts.append(
                plt.text(
                    row["Growth Potential"], row["Price [€]"],
                    row["Publisher"], fontsize=8, color="black"
                )
            )

    # Adjust labels to avoid overlap
    adjust_text(
        texts,
        force_points=1.5,
        force_text=2.5,
        expand_points=(1.2, 1.6),
        expand_text=(1.2, 1.6),
        lim=1000
    )

    # Axes
    plt.xscale("log")
    plt.xlabel("Growth Potential (log scale)")
    plt.ylabel("Average Price (€)")
    plt.title("Publishers: Growth Potential vs Price (Bubble = Recommendations)")

    # Bubble size legend
    for size in [20000, 100000, 300000]:
        plt.scatter([], [], s=size/200, color="gray", alpha=0.4, label=f"{size:,} Recs")
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Bubble = Recommendations")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


def plot_bar(df: pd.DataFrame):
    """Grouped bar chart: Recommendations, Revenue Proxy, Positive Score."""
    metrics = ["Recommendations", "Revenue Proxy", "Positive Score"]

    # Normalize
    df_plot = df.copy()
    for col in metrics:
        df_plot[col] = df_plot[col] / df_plot[col].max()

    # First 10 publishers by ID
    df_plot = df_plot.sort_values(by="publisher_id").head(10)

    # Plot
    colors = ["steelblue", "darkorange", "seagreen"]
    df_plot.set_index("Publisher")[metrics].plot(
        kind="barh",
        figsize=(12, 7),
        color=colors,
        width=0.7
    )

    plt.title("Top 10 Publishers (by ID): Popularity, Revenue, Reputation (normalized)", fontsize=14)
    plt.xlabel("Normalized Score (0–1)")
    plt.ylabel("Publisher")

    plt.legend(metrics, title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    df = pd.read_csv(PROCESSED_DATA_DIR / "attributes.csv")

    # Clean
    df = clean_data(df)

    # Plots
    plot_scatter(df)
    plot_bar(df)


if __name__ == "__main__":
    main()
