
#!/usr/bin/env python3
"""
A simple analysis script to load a trained model and evaluate its performance
on the test dataset by plotting predictions vs. truth and resolutions.
Generates both full-range and truth-focused plots.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm

# ==============================================================================
# 1. CONFIGURATION (Should match your training script)
# ==============================================================================

# --- Set the directory where your split data is located ---
TEST_DATA_DIR = "/scratch/gregory/allcsv2/test"

# --- Set the base directory where your model runs are saved ---
BASE_MODEL_DIR = "./model_runs"

# --- Data and Model Definitions (MUST MATCH the training script) ---
VARIABLES_TO_EXTRACT = ['x', 'q2']
METHODS_TO_EXTRACT = ['da', 'electron', 'jb', 'esigma', 'sigma']
TRUTH_VAR_MAPPING = {'x': 'xbj', 'q2': 'q2'}
BEAM_ENERGIES = ['5x41', '10x100', '18x275']


# ==============================================================================
# 2. HELPER FUNCTIONS AND CLASS DEFINITIONS (Copied from training script)
# ==============================================================================

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
                continue
            if key_column not in df.columns:
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

# NEW: Added compute_range function to calculate focused axis ranges
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

class MLP(torch.nn.Module):
    """
    A simple Multi-Layer Perceptron model.
    """
    def __init__(self, input_size=10, output_size=2):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.layers(x)

def prepare_tensors_for_set(data_dir, beam_energy, variables, methods, truth_mapping):
    """
    Loads data for a specific beam energy from a specific directory.
    """
    print(f"  Loading test data for '{beam_energy}' from '{data_dir}'...")
    mc_dis_path = os.path.join(data_dir, f"*{beam_energy}*mc_dis.csv")
    reco_path = os.path.join(data_dir, f"*{beam_energy}*reco_dis.csv")
    
    mc_dis_df = concat_csvs_unique_event(mc_dis_path, key_column='evt')
    reco_df   = concat_csvs_unique_event(reco_path, key_column='evt')
    
    if mc_dis_df.empty:
        print(f"    Warning: Truth dataframe is empty for this set. Returning empty tensors.")
        return np.array([]), np.array([])
    
    merged_df = pd.merge(mc_dis_df, reco_df, on='evt', how='left')
    num_events = len(merged_df)
    if num_events == 0:
        return np.array([]), np.array([])

    reco_tensor = np.zeros((len(variables), len(methods), num_events))
    truth_tensor = np.zeros((len(variables), 1, num_events))

    for i, var_key in enumerate(variables):
        truth_col_name = truth_mapping.get(var_key)
        if truth_col_name and truth_col_name in merged_df.columns:
            truth_tensor[i, 0, :] = merged_df[truth_col_name].fillna(0).to_numpy()
        
        for j, method in enumerate(methods):
            reco_col_name = f"{method}_{var_key}"
            if reco_col_name in merged_df.columns:
                reco_tensor[i, j, :] = merged_df[reco_col_name].fillna(0).to_numpy()
            
    reco_tensor = np.transpose(reco_tensor, (2, 0, 1))
    truth_tensor = np.transpose(truth_tensor, (2, 0, 1))
    
    print(f"    Loaded {num_events} test events.")
    return reco_tensor, truth_tensor

# MODIFIED: Added custom_range parameter
def save_2d_analysis_hist(x, y, xlabel, ylabel, title, filename, target_dir, custom_range=None):
    """ Saves a 2D histogram for analysis. Can accept a custom plot range. """
    plt.figure(figsize=(8, 6))
    
    # Use the provided range for the histogram, otherwise auto-range
    plt.hist2d(x, y, bins=100, cmap='viridis', norm=LogNorm(), range=custom_range)
    
    plt.colorbar(label="Counts")
    
    # Determine limits for the y=x line
    if custom_range:
        lims = [custom_range[0][0], custom_range[0][1]]
        plt.xlim(lims)
        plt.ylim(lims)
    else:
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]

    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=1, label='Ideal (y=x)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, filename))
    plt.close()
    print(f"    Saved plot: {filename}")


def save_1d_resolution_hist(res_data, var_name, title, filename, target_dir):
    """ Saves a 1D resolution histogram. """
    plt.figure(figsize=(8, 6))
    # Filter out NaNs and infinite values for stats calculation
    res_data_finite = res_data[np.isfinite(res_data)]
    mean, std = res_data_finite.mean(), res_data_finite.std()
    
    plt.hist(res_data_finite, bins=100, range=(-1, 1), histtype='step', lw=2, label=f'Mean: {mean:.3f}\nStd Dev: {std:.3f}')
    plt.xlabel(f"({var_name} Predicted - Truth) / Truth")
    plt.ylabel("Counts")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, filename))
    plt.close()
    print(f"    Saved plot: {filename}")


# ==============================================================================
# 4. MAIN ANALYSIS SCRIPT
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for beam_energy in BEAM_ENERGIES:
        print(f"\n{'='*80}")
        print(f"STARTING ANALYSIS FOR BEAM ENERGY: {beam_energy}")
        print(f"{'='*80}")

        # --- 1. Find the latest trained model ---
        model_dirs = sorted(glob.glob(os.path.join(BASE_MODEL_DIR, beam_energy, "run_*")))
        if not model_dirs:
            print(f"  ERROR: No trained models found for {beam_energy}. Skipping.")
            continue
        
        latest_model_dir = model_dirs[-1]
        model_path = os.path.join(latest_model_dir, "model.pth")
        if not os.path.exists(model_path):
            print(f"  ERROR: Model file not found at {model_path}. Skipping.")
            continue
            
        print(f"  Found latest model: {model_path}")

        # --- 2. Load the test data ---
        test_inputs, test_truths = prepare_tensors_for_set(
            TEST_DATA_DIR, beam_energy, VARIABLES_TO_EXTRACT, METHODS_TO_EXTRACT, TRUTH_VAR_MAPPING
        )
        if len(test_inputs) == 0:
            print(f"  ERROR: No test data found for {beam_energy}. Skipping analysis.")
            continue
            
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_inputs).float(), torch.from_numpy(test_truths).float())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096)

        # --- 3. Perform inference ---
        input_size = len(VARIABLES_TO_EXTRACT) * len(METHODS_TO_EXTRACT)
        output_size = len(VARIABLES_TO_EXTRACT)
        model = MLP(input_size, output_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                all_predictions.append(outputs.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)

        # --- 4. Process results and generate plots ---
        analysis_output_dir = os.path.join(latest_model_dir, "analysis_plots")
        os.makedirs(analysis_output_dir, exist_ok=True)
        print(f"  Saving analysis plots to: {analysis_output_dir}")

        truths_flat = test_truths.reshape(test_truths.shape[0], -1)

        for i, var_key in enumerate(VARIABLES_TO_EXTRACT):
            pred_vals = predictions[:, i]
            truth_vals = truths_flat[:, i]
            
            # --- Plot 1: Full Range Predicted vs Truth ---
            save_2d_analysis_hist(
                x=truth_vals, y=pred_vals,
                xlabel=f"Truth {var_key}", ylabel=f"MLP Predicted {var_key}",
                title=f"MLP Performance for {var_key} ({beam_energy})",
                filename=f"pred_vs_truth_{var_key}_full_range.png",
                target_dir=analysis_output_dir
            )

            # --- Plot 2: Focused Range Predicted vs Truth ---
            # Define the focused plot range based on the truth data distribution
            focused_range = compute_range(truth_vals)
            plot_range_2d = [focused_range, focused_range] # Use same range for x and y
            
            save_2d_analysis_hist(
                x=truth_vals, y=pred_vals,
                xlabel=f"Truth {var_key}", ylabel=f"MLP Predicted {var_key}",
                title=f"MLP Performance for {var_key} ({beam_energy}) - Focused",
                filename=f"pred_vs_truth_{var_key}_focused.png",
                target_dir=analysis_output_dir,
                custom_range=plot_range_2d
            )

            # --- Plot 3: 1D Resolution ---
            valid_truth_mask = truth_vals != 0
            resolution = (pred_vals[valid_truth_mask] - truth_vals[valid_truth_mask]) / truth_vals[valid_truth_mask]
            save_1d_resolution_hist(
                res_data=resolution,
                var_name=var_key,
                title=f"MLP Resolution for {var_key} ({beam_energy})",
                filename=f"resolution_{var_key}.png",
                target_dir=analysis_output_dir
            )

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE.")
    print(f"{'='*80}")
