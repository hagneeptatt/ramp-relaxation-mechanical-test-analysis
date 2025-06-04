## Imports
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import seaborn as sns
import pathlib
from pathlib import Path
import math
import scipy
from scipy.signal import savgol_filter, find_peaks
import os
import glob
from datetime import datetime

############### Main Script ###################

def process_file(file_path, thickness, diameter):
    """Process a single mechanical test file and return calculated parameters"""
    
    # extract df from csv file
    raw_df = pd.read_csv(file_path, skiprows=48, usecols=range(1,4))
    
    # drop row containing units
    raw_df = raw_df.drop(0)
    
    # reset index after dropping
    raw_df = raw_df.reset_index(drop=True)
    
    # convert data to float 
    raw_df = raw_df.astype(float)
    
    # calculate area from diameter
    area = math.pi * (diameter / 2)**2 
    
    # calculate stress and strain 
    raw_df['Stress'] = - raw_df['Load 3'] / area
    raw_df['Strain'] = - raw_df['Disp'] / thickness
    
    # split ramp relaxation and dynamic data
    # first need to calculate relaxation and dynamic start times
    strain_rate = 0.01 # mm/s 
    t_relax_start = (thickness * 0.2) / strain_rate
    t_dyn_start = t_relax_start + 600
    
    # create filtered df with ramp-relaxation and dynamic data
    ramp_rel_df = raw_df[(raw_df['Elapsed Time'] <= t_dyn_start)].copy()
    dyn_df = raw_df[(raw_df['Elapsed Time'] >= t_dyn_start) & (raw_df['Elapsed Time'] <= t_dyn_start + 10)].copy()
    
    # smooth ramp data 
    ramp_rel_df['Stress_smooth'] = savgol_filter(ramp_rel_df['Stress'], 200, 2)
    
    ########### Ramp Relaxation Analysis ############
    ## find peak stress
    peak_stress = max(ramp_rel_df['Stress']) if not ramp_rel_df.empty else np.nan
    
    ## find secant modulus from 5% to 15%
    strain_low = 0.05
    strain_high = 0.15
    
    secant_modulus = np.nan
    if not ramp_rel_df.empty:
        # find idx for above strains
        idx_low = (ramp_rel_df['Strain'] - strain_low).abs().idxmin()
        idx_high = (ramp_rel_df['Strain'] - strain_high).abs().idxmin()
        
        # find corresponding stress values
        stress_low = ramp_rel_df.loc[idx_low, 'Stress_smooth']
        stress_high = ramp_rel_df.loc[idx_high, 'Stress_smooth']
        
        # calculate secant modulus
        secant_modulus = (stress_high - stress_low) / (strain_high - strain_low)
    
    ## find equilibrium modulus
    equil_stress = np.nan
    equil_modulus = np.nan
    
    if not ramp_rel_df.empty:
        # extract last 30 seconds of relaxation phase as equilibrium region
        equil_region_df = ramp_rel_df[(ramp_rel_df['Elapsed Time'] >= t_dyn_start-60) & 
                                      (ramp_rel_df['Elapsed Time'] <= t_dyn_start)].copy()
        
        if not equil_region_df.empty:
            # find average stress over equilibrium region
            equil_stress = equil_region_df['Stress'].mean()
            equil_modulus = equil_stress / 0.2
    
    ## find relaxation time
    t_relax = np.nan
    
    if not ramp_rel_df.empty:
        # extract relaxation region
        relax_region_df = ramp_rel_df[(ramp_rel_df['Elapsed Time'] <= t_dyn_start) & 
                                      (ramp_rel_df['Elapsed Time'] >= t_relax_start)].copy()
        
        if not relax_region_df.empty:
            # define 50% stress 
            stress_50 = peak_stress * 0.5
            
            # create boolean mask where stress is less than 50% of peak
            mask_50 = relax_region_df['Stress_smooth'] <= stress_50
            
            if any(mask_50):
                # find max idx of mask to get first instance stress goes below 50%
                idx_50 = np.argmax(mask_50.values)
                
                # find time from peak stress to 50% peak stress
                t_50 = relax_region_df.iloc[idx_50]['Elapsed Time']
                t_relax = t_50 - t_relax_start
    
    ####### Dynamic Phase Analysis ######
    dyn_modulus = np.nan
    
    if not dyn_df.empty:
        if len(dyn_df) > 15:  # Ensure there's enough data for peak finding
            dyn_peaks_idx, _ = find_peaks(dyn_df['Stress'], distance=15)
            dyn_troughs_idx, _ = find_peaks(-dyn_df['Stress'], distance=15)
            
            if len(dyn_peaks_idx) > 0 and len(dyn_troughs_idx) > 0:
                dyn_peaks = dyn_df['Stress'].iloc[dyn_peaks_idx]
                dyn_troughs = dyn_df['Stress'].iloc[dyn_troughs_idx]
                
                dyn_peak_mean = dyn_peaks.mean()
                dyn_trough_mean = dyn_troughs.mean()
                
                dyn_modulus = (dyn_peak_mean - dyn_trough_mean) / 0.2
    
    # Compile results
    results = {
        'Filename': os.path.basename(file_path),
        'Thickness_mm': thickness,
        'Diameter_mm': diameter,
        'Area_mm2': area,
        'Peak_Stress': peak_stress,
        'Secant_Modulus': secant_modulus,
        'Equilibrium_Stress': equil_stress,
        'Equilibrium_Modulus': equil_modulus,
        'Relaxation_Time_s': t_relax,
        'Dynamic_Modulus': dyn_modulus
    }
    
    return results

def main():
    # Set up folder paths
    folder_path = r'C:\Users\mbgm4fs3\OneDrive - The University of Manchester\PhD\Experimental\Data\5. Mechanical Stimulation\Primary\Mechanical testing\raw_data'
    dimensions_path = Path(folder_path) / 'dimensions' / 'sample_dimensions_config.csv'
    
    # Create output path for results
    output_folder = Path(folder_path) / 'results'
    output_folder.mkdir(exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f'mechanical_test_results_{timestamp}.xlsx'
    
    # Read dimensions file
    try:
        dimensions_df = pd.read_csv(dimensions_path)
        print(f"Successfully loaded dimensions data with {len(dimensions_df)} samples")
    except Exception as e:
        print(f"Error loading dimensions file: {e}")
        print("Using default dimensions from original code")
        dimensions_df = None
    
    # Find all CSV files in the directory
    csv_files = glob.glob(str(Path(folder_path) / '*.CSV')) + glob.glob(str(Path(folder_path) / '*.csv'))
    
    if not csv_files:
        print("No CSV files found in the specified directory")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file and collect results
    all_results = []
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing file: {filename}")
        
        # Get dimensions for this file from the dimensions dataframe
        if dimensions_df is not None and filename in dimensions_df['Filename'].values:
            sample_info = dimensions_df[dimensions_df['Filename'] == filename].iloc[0]
            thickness = sample_info['Thickness_mm']
            diameter = sample_info['Diameter_mm']
            print(f"  Using dimensions from file: Thickness={thickness}mm, Diameter={diameter}mm")
        else:
            # Default dimensions if not found
            thickness = 1.81  # mm
            diameter = 5.08  # mm
            print(f"  Using default dimensions: Thickness={thickness}mm, Diameter={diameter}mm")
        
        try:
            # Process the file and get results
            results = process_file(file_path, thickness, diameter)
            all_results.append(results)
            print(f"  Processing completed successfully")
        except Exception as e:
            print(f"  Error processing file {filename}: {e}")
    
    if all_results:
        # Combine all results into a single dataframe
        results_df = pd.DataFrame(all_results)
        
        # Save results to Excel
        results_df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results were successfully processed")

if __name__ == "__main__":
    main()