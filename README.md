# Mechanical test processing - ramp/relaxation + dynamic modulus

This script batch-processes mechanical testing CSV files (ramp-relaxation followed by a short dynamic phase) and outputs:

- A single Excel file summarising mechanical parameters for each sample  
- Per-sample CSV exports of the raw dynamic segment  
- Per-sample CSV exports of a combined trace (smoothed ramp-relaxation + raw dynamic)

It is designed for TA instruments TA 5500 exports where the first 48 rows contain metadata and the measured data are in columns 2–4.

---

## What the script does

For each `*.CSV` file:

1. Loads time, load and displacement data  
2. Computes sample **area** from diameter and converts to:  
   - **Stress** = -Load / area  
   - **Strain** = -Displacement / thickness  
3. Splits the signal into:
   - Ramp-relaxation phase  
   - Dynamic phase (10 s window)  
4. Applies Savitzky-Golay smoothing to ramp-relaxation stress  
5. Calculates:

- Peak stress  
- Secant modulus (5–15% strain)  
- Equilibrium stress & modulus  
- Relaxation time to 50% peak stress  
- Dynamic modulus from cyclic loading

---

## Folder structure

```
gel_raw/
 ├─ *.CSV
 ├─ dimensions/
 │   └─ sample_dimensions_config.csv
 ├─ results/        (auto-created)
 ├─ dynamic_data/   (auto-created)
 └─ combined_data/  (auto-created)
```

---

## Dimensions file format

`dimensions/sample_dimensions_config.csv` must include:

| Column | Description |
|--------|------------|
| Filename | CSV filename (e.g. Sample1.CSV) |
| Thickness_mm | Sample thickness |
| Diameter_mm | Sample diameter |

Files without matching dimensions are skipped.

---

## Outputs

### Summary Excel file
Saved to:

```
results/mechanical_test_results_TIMESTAMP.xlsx
```

Includes:

- Peak stress
- Secant modulus
- Equilibrium modulus
- Relaxation time
- Dynamic modulus

### Raw dynamic data
```
dynamic_data/<filename>_dynamic.csv
```

### Combined data (smoothed ramp + raw dynamic)
```
combined_data/<filename>_combined.csv
```

---

## Key assumptions

### Geometry
Area assumes a circular sample:
```
area = π * (diameter / 2)^2
```

### Timing
The script assumes:

- Ramp to **20% strain**
- Straining rate = **0.01 mm/s**
- Dynamic loading begins **600 s after ramp**

Modify these values in `process_file()` if your protocol differs.

### Smoothing
Savitzky-Golay filter:
- window length = 200
- polynomial order = 2

Reduce the window length if your dataset is small.

---

## Requirements

Python 3 with:

```
pandas
numpy
scipy
matplotlib
```

Install:

```
pip install pandas numpy scipy matplotlib
```

---

## Running the script

1. Update the `folder_path` in `main()`.
2. Ensure the dimensions CSV is correct.
3. Run:

```
python script_name.py
```

---

## Troubleshooting

**No CSV files found**
- Check path and file extension `.CSV`

**Column name errors**
- Confirm CSV export format and update column names if needed

**NaN dynamic modulus**
- Ensure dynamic section contains enough cycles and data points

---

## Reproducibility notes

For thesis/publication use:

- Save dependency versions (`pip freeze > requirements.txt`)
- Archive the repository and generate a DOI (e.g. Zenodo)
- Tag the thesis analysis version
