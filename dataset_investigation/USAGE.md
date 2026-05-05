# CSV Investigation and Preprocessing Usage

## Environment

From project root:

```bash
cd ~/aiotlab/htien/TFM
conda activate tien_tfm
```

If your `tien_tfm` environment does not include `pandas`, use:

```bash
/home/user01/miniconda3/bin/python
```

in place of `python` for the commands below.

## 1) Investigate Source CSV Datasets

```bash
python dataset_investigation/scripts/investigate_csv_datasets.py
```

Outputs:

- `dataset_investigation/reports/eICU_cardiacArrest_report.json`
- `dataset_investigation/reports/eICU_sepsis_report.json`
- `dataset_investigation/reports/MIMIC_gib_report.json`
- `dataset_investigation/reports/combined_report.json`

## 2) Explore Template `data/toy_data.pkl`

```bash
python dataset_investigation/scripts/explore_toy_template.py
```

Output:

- `dataset_investigation/reports/toy_template_schema.json`

## 3) Preprocess CSV -> PKL (Per Config Profile)

### `eICU` profile (from sepsis CSV)

```bash
python dataset_investigation/scripts/preprocess_csv_to_pkl.py \
  --csv tfm-data/eICU_sepsis_physionet.csv \
  --profile eICU \
  --output dataset_investigation/outputs/eicu_sepsis_eicu.pkl
```

### `eICU_ablated` profile (from sepsis CSV)

```bash
python dataset_investigation/scripts/preprocess_csv_to_pkl.py \
  --csv tfm-data/eICU_sepsis_physionet.csv \
  --profile eICU_ablated \
  --output dataset_investigation/outputs/eicu_sepsis_eicu_ablated.pkl
```

### `eICU_multdim` profile (from cardiac-arrest CSV)

```bash
python dataset_investigation/scripts/preprocess_csv_to_pkl.py \
  --csv tfm-data/eICU_cardiacArrest_physionet.csv \
  --profile eICU_multdim \
  --output dataset_investigation/outputs/eicu_cardiac_multdim.pkl
```

### `mimic_liver` profile (from MIMIC GIB CSV)

```bash
python dataset_investigation/scripts/preprocess_csv_to_pkl.py \
  --csv tfm-data/MIMIC_gib_physionet.csv \
  --profile mimic_liver \
  --output dataset_investigation/outputs/mimic_gib_liver.pkl
```

## 4) Validate Profile Compatibility

```bash
python dataset_investigation/scripts/validate_profile_coverage.py
```

Output:

- `dataset_investigation/reports/profile_validation_report.json`

## 5) Quick Output Sanity Check

```bash
python - <<'PY'
import pandas as pd
for path in [
    "dataset_investigation/outputs/eicu_sepsis_eicu.pkl",
    "dataset_investigation/outputs/eicu_sepsis_eicu_ablated.pkl",
    "dataset_investigation/outputs/eicu_cardiac_multdim.pkl",
    "dataset_investigation/outputs/mimic_gib_liver.pkl",
]:
    data = pd.read_pickle(path)
    print(path, list(data.keys()), {k: v.shape for k, v in data.items()})
PY
```

## Notes

- Split assignment is sourced from the CSV `label` column and requires `train`, `val`, `test` labels.
- `time_scaled_v1` is created from existing `time_scaled`.
- Profile-specific aliases are applied where needed:
  - `eICU_multdim`: creates `*_scaled` columns from available normalized columns and maps `age_normalized` to `AGE_AT_ADM_normalized`.
  - `mimic_liver`: creates `hr_normalized_scaled` from `hr_normalized`, normalizes map as `map_normalized` (supports `map`/`MAP` inputs), and maps `prbc` -> `prbc_outcome`.
