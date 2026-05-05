# Dataset Report: eICU_cardiacArrest

- Rows: `1650378`
- Columns: `11`
- Unique patients (`HADM_ID`): `64589`

## Label counts
- `train`: `1322663`
- `val`: `164522`
- `test`: `163193`

## Time stats (`time_scaled`)
- min: `0.0`
- max: `1.0`
- mean: `0.4508898647427708`
- is_nonnegative: `True`

## Profile coverage
- `eICU` present: HADM_ID, hr_normalized
- `eICU` missing: apache_outcome_prob, map_normalized, norepi_inf_scaled, time_scaled_v1
- `eICU_ablated` present: HADM_ID, hr_normalized
- `eICU_ablated` missing: apache_outcome_prob, map_normalized, time_scaled_v1
- `eICU_multdim` present: HADM_ID
- `eICU_multdim` missing: AGE_AT_ADM_normalized, dbp_normalized_scaled, hr_normalized_scaled, rr_normalized_scaled, time_scaled_v1
- `mimic_liver` present: HADM_ID
- `mimic_liver` missing: bloodprod, hr_normalized_scaled, map_normalized, prbc_outcome, pressor, severe_liver, time_scaled_v1
