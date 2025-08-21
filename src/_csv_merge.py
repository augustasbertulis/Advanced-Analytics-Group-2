import pandas as pd
from functools import reduce
import glob
import os

input_folder = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\processed data"  # folder containing multiple CSVs
output_file = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\processed data\combined_output.csv"
des_path =r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\processed data\descriptions_clean.csv"
# Get a list of all CSV files in the folder
des= pd.read_csv(input_folder+"\descriptions_clean.csv")
df=des.copy()
games= pd.read_csv(input_folder+"\games_cleaned.csv")
genres= pd.read_csv(input_folder+"\genres_clean.csv")
promotional= pd.read_csv(input_folder+"\promotional_clean.csv")
steamspy = pd.read_csv(input_folder+"\steamspy_insights_cleaned.csv")
tags= pd.read_csv(input_folder+r"\tags_clean.csv")

# Inspect the first few rows
df=df.merge(games, on="app_id", how="left")
df=df.merge(genres, on="app_id", how="left")
df=df.merge(promotional, on="app_id", how="left")
df=df.merge(steamspy, on="app_id", how="left")
df=df.merge(tags, on="app_id", how="left")

# Save the combined DataFrame to a new CSV

print(df.head())

df.to_csv(output_file, index=False)