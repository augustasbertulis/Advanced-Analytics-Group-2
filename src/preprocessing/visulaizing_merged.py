import pandas as pd
import matplotlib.pyplot as plt
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

def main():
    # -------------------- File paths --------------------
    import_path = PROCESSED_DATA_DIR / "combined.csv"
    # Load dataset
    df = pd.read_csv(import_path, low_memory=False)

    # --- Calculate NaN percentage per column ---
    nan_pct = df.isna().mean().sort_values(ascending=False) * 100

    # --- Print summary ---
    print(df.info())
    print(nan_pct)

    # --- Plot NaN percentages ---
    plt.figure(figsize=(14, 6))
    nan_pct.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("NaN Percentage per Column", fontsize=14)
    plt.ylabel("Percentage of NaN Values (%)")
    plt.xlabel("Columns")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
