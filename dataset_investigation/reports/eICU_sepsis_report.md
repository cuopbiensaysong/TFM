# Dataset Report: eICU_sepsis

- Rows: `147212`
- Columns: `11`
- Unique patients (`HADM_ID`): `3362`

## Label counts
- `train`: `117855`
- `test`: `14872`
- `val`: `14485`

## Time stats (`time_scaled`)
- min: `0.0`
- max: `1.0`
- mean: `0.4695872798337696`
- is_nonnegative: `True`

## Profile coverage
- `eICU` present: HADM_ID, apache_outcome_prob, hr_normalized, map_normalized, norepi_inf_scaled
- `eICU` missing: time_scaled_v1
- `eICU_ablated` present: HADM_ID, apache_outcome_prob, hr_normalized, map_normalized
- `eICU_ablated` missing: time_scaled_v1
- `eICU_multdim` present: HADM_ID
- `eICU_multdim` missing: AGE_AT_ADM_normalized, dbp_normalized_scaled, hr_normalized_scaled, rr_normalized_scaled, time_scaled_v1
- `mimic_liver` present: HADM_ID
- `mimic_liver` missing: 1, MAP, bloodprod, prbc_outcome, pressor, severe_liver, time_scaled_v1
