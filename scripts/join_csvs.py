import pandas as pd
import os

# Set the directory where the CSV files are located



def merge_csvs():
    key = "Transformer"
    dir_path = f'./{key}'
    # Create an empty DataFrame to store the merged data
    merged_data = pd.DataFrame()

    # Loop through all CSV files in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv') and key in filename:
            # Load the CSV file into a DataFrame
            csv_path = os.path.join(dir_path, filename)
            data = pd.read_csv(csv_path, index_col=0)
            
            # Merge the DataFrame with the merged_data DataFrame
            merged_data = pd.concat([merged_data, data], axis=1, sort=False)

    # Write the merged data to a new CSV file
    if not os.path.exists('./res'):
        os.mkdir("./res")
    merged_data.to_csv(f'./res/{key}_merged.csv')