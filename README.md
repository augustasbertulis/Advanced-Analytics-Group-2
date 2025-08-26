## Advanced-Analytics Group 2
# AI Disclosure
This project contains content generated or assisted by AI.

# Project Overview
This project clusters game publishers based on performance KPIs such as **revenue, reviews, and engagement** using **K-Means clustering** to identify **top-performing publishers**.  

During cleaning and preprocessing, we stripped away a lot of noise:
- Removed unnecessary descriptive data files (e.g., `descriptions.csv`, `promotional.csv`) that consumed too much memory.  
- Cleaned string-based fields.  
- Aggregated KPIs to the **publisher level**.  
- Slimmed the dataset down to only essential numeric fields required for clustering.  

To avoid arbitrarily selecting k, we evaluated cluster quality using:
- SSE (Sum of Squared Errors): Measures within-cluster compactness. Lower SSE = tighter clusters.
- Elbow Curve: Plots SSE against k to identify the “elbow point,” where additional clusters stop giving big improvements.
- Silhouette Score: Evaluates how well a publisher fits into its assigned cluster compared to other clusters. Scores range from -1 (bad) to +1 (excellent).
- We combined these metrics to make an informed choice for the number of clusters.

After clustering:
- Explored how changing the importance of different KPIs affects clustering results and identified the most influential features through a sensitivity analysis.
- Created a risk-adjusted portfolio using a pseudo markowitz model of publishers using the clustering output, showing how top performers could be weighted to maximize growth while managing risk.

---

# Project Structure
- `data/raw data/` : Contains the raw CSV files.  
- `processed data/` : Folder where all outputs are stored (cluster assignments, summaries, visualizations, cleaned data).  
- `main.py` : Script that runs the analysis.  
- `README.md` : Documentation for the project.
- `paths.py` : A small utility script that defines project data paths and ensures required directories exist.

---
# How to Run
1. Place your raw data CSV in `data/raw data`.
2. Run the script:
   main.py
3. Outputs will be saved in the `processed data/` folder.

# ⚙️ Requirements
- Python 3.10+  
- pandas  
- scikit-learn  
- matplotlib  

Install with:
```bash
pip install -r requirements.txt
