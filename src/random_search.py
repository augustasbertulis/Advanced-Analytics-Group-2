# kmeans_weight_sensitivity.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # progress bar
from model import run_all_kmeans, MODEL_CONFIGS

# ------------------------ Random search settings ------------------------
NUM_SAMPLES = 10  # number of random weight combinations
TOP_N = 50  # number of top publishers to evaluate metric

# ------------------------ Helper functions ------------------------
def random_weights(num_models):
    """Generate random weights that sum to 1."""
    weights = np.random.rand(num_models)
    return weights / weights.sum()

def evaluate_weights(weights_dict):
    """Run all KMeans with given weights and return top-N metric."""
    ranked, _ = run_all_kmeans(vote_weights=weights_dict)
    # metric: mean weighted_votes of top N publishers
    if "weighted_votes" in ranked.columns:
        metric = ranked["weighted_votes"].head(TOP_N).mean()
    else:
        metric = ranked["votes"].head(TOP_N).mean()
    return metric

# ------------------------ Random search ------------------------
results = []
model_keys = list(MODEL_CONFIGS.keys())
num_models = len(model_keys)

print(f"Running random search over {NUM_SAMPLES} weight combinations...")
for i in tqdm(range(NUM_SAMPLES), desc="Random Search Progress"):
    w = random_weights(num_models)
    weights_dict = dict(zip(model_keys, w))
    metric = evaluate_weights(weights_dict)
    results.append({**weights_dict, "metric": metric})

results_df = pd.DataFrame(results)

# ------------------------ Identify best weights ------------------------
best_idx = results_df["metric"].idxmax()
best_weights = results_df.loc[best_idx, model_keys].to_dict()
best_metric = results_df.loc[best_idx, "metric"]

print("\nBest weights found:", best_weights)
print("Best metric:", best_metric)

# ------------------------ Visualizations ------------------------
# 1. Distribution of metrics
plt.figure(figsize=(12, 6))
sns.histplot(results_df["metric"], bins=30, kde=True)
plt.title("Distribution of Top-N Metric Across Random Weight Combinations")
plt.xlabel("Metric (Mean Weighted Votes)")
plt.ylabel("Frequency")
plt.show()

# 2. Scatter plots of weights vs metric
sns.pairplot(results_df, x_vars=model_keys, y_vars="metric", kind="scatter", height=3)
plt.suptitle("Weights vs Metric Scatter Plots", y=1.02)
plt.show()

# 3. Heatmap: correlation between weights and metric
plt.figure(figsize=(10, 5))
sns.heatmap(results_df.corr()[["metric"]].drop("metric"), annot=True, cmap="coolwarm")
plt.title("Correlation of Model Weights with Metric")
plt.show()
