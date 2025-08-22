import pandas as pd
from functools import reduce
import glob
import os

#input_folder = "data/clean data"  # folder containing multiple CSVs
#output_file = "data/clean data/combined_output.csv"


input_folder = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data"
output_file = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/combined.csv"
# Get a list of all CSV files in the folder
des= pd.read_csv(input_folder+r"\descriptions_clean.csv")
df=des.copy()
games= pd.read_csv(input_folder+r"\games_clean.csv")
genres= pd.read_csv(input_folder+r"\genres_clean.csv")
promotional= pd.read_csv(input_folder+r"\promotional_clean.csv")
steamspy = pd.read_csv(input_folder+r"\steamspy_insights.csv")
tags= pd.read_csv(input_folder+r"\tags_clean.csv")
reviews= pd.read_csv(input_folder+r"\reviews_clean.csv")

# Inspect the first few rows
df=df.merge(games, on="app_id", how="left")
df=df.merge(genres, on="app_id", how="left")
df=df.merge(promotional, on="app_id", how="left")
df=df.merge(steamspy, on="app_id", how="left")
df=df.merge(tags, on="app_id", how="left")
df=df.merge(reviews, on="app_id", how="left")
# Save the combined DataFrame to a new CSV

print(df.head())

df.to_csv(output_file, index=False)