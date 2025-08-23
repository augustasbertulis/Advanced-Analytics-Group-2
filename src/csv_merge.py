import pandas as pd
from functools import reduce
import glob
import os

input_folder = "data/clean data"  # folder containing multiple CSVs
output_file = "data/clean data/combined_output.csv"


#input_folder = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data"
#output_file = r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/combined.csv"
# Get a list of all CSV files in the folder
#des= pd.read_csv(input_folder+r"\descriptions_clean.csv")
#gam= pd.read_csv(input_folder+r"\games_clean.csv")
#gen= pd.read_csv(input_folder+r"\genres_clean.csv")
#pro= pd.read_csv(input_folder+r"\promotional_clean.csv")
#ste= pd.read_csv(input_folder+r"\steamspy_insights.csv")
#tag= pd.read_csv(input_folder+r"\tags_clean.csv")
#rev= pd.read_csv(input_folder+r"\reviews_clean.csv")
des= pd.read_csv(input_folder+"/descriptions_clean.csv")
gam= pd.read_csv(input_folder+"/games_clean.csv")
gen= pd.read_csv(input_folder+"/genres_clean.csv")
pro= pd.read_csv(input_folder+"/promotional_clean.csv")
ste= pd.read_csv(input_folder+"/steamspy_insights_clean.csv")
tag= pd.read_csv(input_folder+"/tags_clean.csv")
rev= pd.read_csv(input_folder+"/reviews_clean.csv")
df=gam.copy()

# Inspect the first few rows
df=df.merge(des, on="app_id", how="left")
df=df.merge(gen, on="app_id", how="left")
df=df.merge(pro, on="app_id", how="left")
df=df.merge(ste, on="app_id", how="left")
df=df.merge(tag, on="app_id", how="left")
df=df.merge(rev, on="app_id", how="left")
# Save the combined DataFrame to a new CSV

print(df.shape)

df.to_csv(output_file, index=False)