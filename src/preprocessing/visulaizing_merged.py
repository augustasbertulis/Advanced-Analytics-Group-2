import pandas as pd
import matplotlib.pyplot as plt
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

#-------------------- File paths (adjust if needed) --------------------
import_path = PROCESSED_DATA_DIR / "combined.csv"
# Load your dataset
# import_path = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/combined.csv"
df = pd.read_csv(import_path, low_memory=False)

# --- Calculate NaN percentage per column ---
nan_pct = df.isna().mean().sort_values(ascending=False) * 100

# --- Print summary (like left side of your screenshot) ---
print(df.info())
print(nan_pct)

# --- Plot (like right side of your screenshot) ---
plt.figure(figsize=(14,6))
nan_pct.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("NaN Percentage per Column", fontsize=14)
plt.ylabel("Percentage of NaN Values (%)")
plt.xlabel("Columns")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
