#!/usr/bin/env python3
"""
Analyzes reconstructed Far-Forward Lambda particles by comparing reco and truth CSVs.
Includes plots for hit locations (1D and 2D), momentum comparison, and resolutions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import os

# --- Core Settings ---

# Base directory for input CSV files
data_base_dir = "/scratch/gregory/allcsv4" 

# Base directory for all output histograms
base_output_dir = "/home/gregory/sum25/eic/analysis/ff_lambda_plots"
os.makedirs(base_output_dir, exist_ok=True)
print(f"Base output directory set to: {base_output_dir}")

# --- Helper Functions ---

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
                if 'event' in df.columns and key_column == 'evt':
                     df = df.rename(columns={'event': 'evt'})
                else:
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

def compute_range(arr, use_percentiles=True, lower=1, upper=99):
    """
    Compute a plotting range, defaulting to 1st and 99th percentiles.
    """
    mask = np.isfinite(arr)
    if not np.any(mask):
        return (0, 1)
    if use_percentiles:
        lo, hi = np.percentile(arr[mask], [lower, upper])
    else:
        lo, hi = np.nanmin(arr[mask]), np.nanmax(arr[mask])
    if lo == hi:
        hi = lo + 1e-6
    return (lo, hi)

def save_2dhist(x, y, xlabel, ylabel, title, filename, target_dir, bins=100):
    """
    Save a 2D histogram.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        print(f"  Skipping 2D histogram {filename}: no finite data.")
        return
    plt.figure(figsize=(8, 6))
    plt.hist2d(x[mask], y[mask], bins=bins, cmap='viridis', norm=LogNorm())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="Counts")
    plt.title(title, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    full_path = os.path.join(target_dir, filename)
    plt.savefig(full_path)
    plt.close()
    print(f"  Saved plot: {filename}")

def save_1dhist(data, xlabel, title, filename, target_dir, bins=100, plot_range=None):
    """
    Save a 1D histogram.
    """
    mask = np.isfinite(data)
    if not mask.any():
        print(f"  Skipping 1D histogram {filename}: no finite data.")
        return
    plt.figure(figsize=(8, 6))
    # Use dynamic range if none is provided
    if plot_range is None:
        plot_range = compute_range(data[mask])
        
    plt.hist(data[mask], bins=bins, range=plot_range, histtype='stepfilled', color='teal', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.title(title, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    full_path = os.path.join(target_dir, filename)
    plt.savefig(full_path)
    plt.close()
    print(f"  Saved plot: {filename}")


# --- Main Script Logic ---

beam_energies = ['5x41', '10x100', '18x275']
particles_to_check = ['lam', 'neut', 'gam1', 'gam2']
momenta_vars = ['px', 'py', 'pz', 'energy']
position_vars = ['ref_x', 'ref_y', 'ref_z']

for beam_energy in beam_energies:
    print(f"\n--- Processing Beam Energy: {beam_energy} ---")

    current_beam_energy_output_dir = os.path.join(base_output_dir, beam_energy)
    os.makedirs(current_beam_energy_output_dir, exist_ok=True)
    
    print("  Loading CSV data...")
    reco_ff_df = concat_csvs_unique_event(os.path.join(data_base_dir, f"*{beam_energy}*reco_ff_lambda.csv"), key_column='evt')
    mcpart_df = concat_csvs_unique_event(os.path.join(data_base_dir, f"*{beam_energy}*mcpart_lambda.csv"), key_column='evt')
    
    if reco_ff_df.empty or mcpart_df.empty:
        print(f"  Skipping {beam_energy}: A required dataframe is empty.")
        continue
        
    merged_df = pd.merge(reco_ff_df, mcpart_df, on='evt', suffixes=('_reco', '_truth'))
    if merged_df.empty:
        print(f"  Skipping {beam_energy}: Merged DataFrame is empty (no common events found).")
        continue

    print(f"  Successfully loaded and merged data. Found {len(merged_df)} common events.")

    # --- 1. Plot 2D Reconstructed Hit Locations ---
    print("\n  1. Plotting 2D Reconstructed Hit Locations (Y vs X)...")
    hit_loc_dir = os.path.join(current_beam_energy_output_dir, "hit_locations")
    os.makedirs(hit_loc_dir, exist_ok=True)
    for particle in particles_to_check:
        x_col_reco, y_col_reco = f"{particle}_ref_x_reco", f"{particle}_ref_y_reco"
        x_col_nosuffix, y_col_nosuffix = f"{particle}_ref_x", f"{particle}_ref_y"
        x_col = x_col_reco if x_col_reco in merged_df.columns else x_col_nosuffix
        y_col = y_col_reco if y_col_reco in merged_df.columns else y_col_nosuffix
        
        if x_col in merged_df.columns and y_col in merged_df.columns:
            print(f"    Found 2D hit location columns for '{particle}': ('{x_col}', '{y_col}')")
            save_2dhist(
                x=merged_df[x_col], y=merged_df[y_col],
                xlabel=f"Reconstructed {particle} X [cm]", ylabel=f"Reconstructed {particle} Y [cm]",
                title=f"Reconstructed Hit Location for {particle.capitalize()} ({beam_energy})",
                filename=f"hit_location_2d_{particle}.png",
                target_dir=hit_loc_dir
            )
        else:
            print(f"    Skipping 2D hit location plot for '{particle}': Columns not found.")
            
    # --- 2. Plot 1D Hit Location Distributions ---
    print("\n  2. Plotting 1D Reconstructed Hit Location Distributions...")
    for particle in particles_to_check:
        for var in position_vars:
            col_reco = f"{particle}_{var}_reco"
            col_nosuffix = f"{particle}_{var}"
            col_to_plot = col_reco if col_reco in merged_df.columns else col_nosuffix
            
            if col_to_plot in merged_df.columns:
                print(f"    Found 1D hit location column for '{particle}': '{col_to_plot}'")
                save_1dhist(
                    data=merged_df[col_to_plot],
                    xlabel=f"Reconstructed {particle} {var.replace('_', ' ')} [cm]",
                    title=f"1D Distribution of Reco {particle.capitalize()} {var.replace('_', ' ')} ({beam_energy})",
                    filename=f"hit_location_1d_{particle}_{var}.png",
                    target_dir=hit_loc_dir
                )
            else:
                print(f"    Skipping 1D hit location plot for '{particle} {var}': Column not found.")

    # --- 3. Plot Reconstructed vs. Truth Momenta ---
    print("\n  3. Plotting Reconstructed vs. Truth Momenta...")
    mom_comp_dir = os.path.join(current_beam_energy_output_dir, "momentum_comparison")
    os.makedirs(mom_comp_dir, exist_ok=True)
    for particle in particles_to_check:
        for var in momenta_vars:
            reco_col, truth_col = f"{particle}_{var}_reco", f"{particle}_{var}_truth"
            if reco_col in merged_df.columns and truth_col in merged_df.columns:
                save_2dhist(
                    x=merged_df[truth_col], y=merged_df[reco_col],
                    xlabel=f"Truth {particle} {var} [GeV]", ylabel=f"Reco {particle} {var} [GeV]",
                    title=f"Reco vs. Truth for {particle.capitalize()} {var} ({beam_energy})",
                    filename=f"reco_vs_truth_{particle}_{var}.png", target_dir=mom_comp_dir
                )

    # --- 4. Plot Agreement (Resolution) of Kinematic Variables ---
    print("\n  4. Plotting Agreement (Resolution) for Position and Momenta...")
    agreement_dir = os.path.join(current_beam_energy_output_dir, "agreement_plots")
    os.makedirs(agreement_dir, exist_ok=True)
    vars_for_resolution = position_vars + momenta_vars
    for particle in particles_to_check:
        for var in vars_for_resolution:
            reco_col, truth_col = f"{particle}_{var}_reco", f"{particle}_{var}_truth"
            if reco_col in merged_df.columns and truth_col in merged_df.columns:
                valid_mask = (merged_df[truth_col] != 0) & merged_df[reco_col].notna()
                if not valid_mask.any(): continue
                
                resolution = (merged_df.loc[valid_mask, reco_col] - merged_df.loc[valid_mask, truth_col]) / merged_df.loc[valid_mask, truth_col]
                
                save_1dhist(
                    data=resolution,
                    xlabel=f"({particle} {var} Reco - Truth) / Truth",
                    title=f"Agreement for {particle.capitalize()} {var} ({beam_energy})",
                    filename=f"agreement_{particle}_{var}.png",
                    target_dir=agreement_dir
                )

print("\n--- All processing complete. ---")
