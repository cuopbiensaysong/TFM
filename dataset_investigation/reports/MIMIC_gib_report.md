# Dataset Report: MIMIC_gib

- Rows: `61741`
- Columns: `13`
- Unique patients (`HADM_ID`): `2602`

## Label counts
- `train`: `49403`
- `val`: `6314`
- `test`: `6024`

## Time stats (`time_scaled`)
- min: `0.0`
- max: `0.9993055555555554`
- mean: `0.48399992846460754`
- is_nonnegative: `True`

## Profile coverage
- `eICU` present: HADM_ID, hr_normalized, map_normalized
- `eICU` missing: apache_outcome_prob, norepi_inf_scaled, time_scaled_v1
- `eICU_ablated` present: HADM_ID, hr_normalized, map_normalized
- `eICU_ablated` missing: apache_outcome_prob, time_scaled_v1
- `eICU_multdim` present: HADM_ID
- `eICU_multdim` missing: AGE_AT_ADM_normalized, dbp_normalized_scaled, hr_normalized_scaled, rr_normalized_scaled, time_scaled_v1
- `mimic_liver` present: HADM_ID, bloodprod, pressor, severe_liver
- `mimic_liver` missing: 1, MAP, prbc_outcome, time_scaled_v1
