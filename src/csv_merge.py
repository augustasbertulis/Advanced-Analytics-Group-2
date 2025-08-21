import pandas as pd
import glob
import os

input_folder = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\processed data"  # folder containing multiple CSVs
output_file = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\processed data\combined_output.csv"

# Get a list of all CSV files in the folder
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# Check if any CSV files were found
if not csv_files:
    print("No CSV files found in the folder!")
    exit()

# Read and store each CSV in a list
dataframes = [pd.read_csv(file) for file in csv_files]

# Combine all CSVs into one DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Inspect the first few rows
print(combined_df.head())

# Save the combined DataFrame to a new CSV
combined_df.to_csv(output_file, index=False)

print(f"Successfully combined {len(csv_files)} CSV files into {output_file}")
