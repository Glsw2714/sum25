
#!/usr/bin/env python3
"""
script to read truth and reco CSVs and format the data into
specific NumPy tensors for x and q2 variables.
"""

import pandas as pd
import numpy as np
import glob
import os

# Base directory for input CSV files
data_base_dir = "/scratch/gregory/allcsv"

def concat_csvs_unique_event(pattern, key_column='evt'):
    """
    Read all CSV files matching the naming pattern and ensure unique event IDs.
    """
    dfs = []
    offset = 0
    files_found = sorted(glob.glob(pattern))
    if not files_found:
        print(f"No files found matching pattern: {pattern}")
        return pd.DataFrame()
    for file in files_found:
        try:
            df = pd.read_csv(file)
            df.columns = [col.strip().strip(',') for col in df.columns]
            if df.empty:
                print(f"Skipping empty file: {file}")
                continue
            if key_column not in df.columns:
                print(f"Skipping malformed file (missing key_column '{key_column}'): {file}")
                continue
            df[key_column] = pd.to_numeric(df[key_column], errors="coerce")
            df[key_column] += offset
            max_evt = df[key_column].max()
            if pd.notna(max_evt):
                offset = int(max_evt) + 1
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Define the specific variables and methods to be extracted
variables_to_extract = ['x', 'q2']
methods_to_extract = ['da', 'electron', 'jb', 'esigma', 'sigma']

# Map the internal variable names to the column names in the CSV files
truth_var_mapping = {'x': 'xbj', 'q2': 'q2'}

beam_energies = ['5x41', '10x100', '18x275']

for beam_energy in beam_energies:
    print(f"--- Processing Beam Energy: {beam_energy} ---")

    # Load data
    print("  Loading CSV data...")
    mc_dis_df = concat_csvs_unique_event(os.path.join(data_base_dir, f"*{beam_energy}*mc_dis.csv"), key_column='evt')
    reco_df   = concat_csvs_unique_event(os.path.join(data_base_dir, f"*{beam_energy}*reco_dis.csv"), key_column='evt')
    
    if mc_dis_df.empty:
        print(f"  Skipping {beam_energy}: Truth dataframe (mc_dis_df) is empty.")
        continue
    
    
    # If reco event is missing its value will be NaN
    merged_df = pd.merge(mc_dis_df, reco_df, on='evt', how='left')
    

    # Define num_events based on the number of rows in the merged DataFrame
    num_events = len(merged_df)
    print(f"  Total number of truth events to process: {num_events}")
    
    
    # Shape: (variables, methods, events)
    reco_tensor = np.zeros((len(variables_to_extract), len(methods_to_extract), num_events))
    
    # Shape: (variables, 1 for truth, events)
    truth_tensor = np.zeros((len(variables_to_extract), 1, num_events))
    
    print(f"  Initialized reco_tensor with shape: {reco_tensor.shape}")
    print(f"  Initialized truth_tensor with shape: {truth_tensor.shape}")

    # -Populate the Tensors 
    print("  Populating tensors")

    # Loop through the variables ('x', 'q2')
    for i, var_key in enumerate(variables_to_extract):
        
        # 1. Populate the TRUTH tensor for this variable
        truth_col_name = truth_var_mapping.get(var_key)
        if truth_col_name and truth_col_name in merged_df.columns:
            # Fill NaN with 0 just in case, then convert to numpy array
            truth_values = merged_df[truth_col_name].fillna(0).to_numpy()
            truth_tensor[i, 0, :] = truth_values
        else:
            print(f"Warning: Truth column for '{var_key}' not found. Leaving as zeros.")
            
        # 2. Populate the RECO tensor for this variable across all methods
        for j, method in enumerate(methods_to_extract):
            reco_col_name = f"{method}_{var_key}"
            
            if reco_col_name in merged_df.columns:
                # The left merge created NaNs for missing reco events. Fill these with 0.
                reco_values = merged_df[reco_col_name].fillna(0).to_numpy()
                reco_tensor[i, j, :] = reco_values
            else:
                # If a method's column doesn't exist at all, it remains zeros by default
                print(f"  Warning: Column '{reco_col_name}' not found. Data will be all zeros for this slice.")

    
    print(f"\n  --- Verification for {beam_energy} ---")
    print(f"  Final shape of reco_tensor: {reco_tensor.shape}")
    print(f"  Final shape of truth_tensor: {truth_tensor.shape}")
    
    
    # np.save(f'reco_tensor_{beam_energy}.npy', reco_tensor)
    # np.save(f'truth_tensor_{beam_energy}.npy', truth_tensor)

