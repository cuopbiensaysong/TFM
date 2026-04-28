import unittest

import pandas as pd

from dataset_investigation.scripts.preprocess_csv_to_pkl import (
    PROFILE_SPECS,
    convert_dataframe_for_profile,
    split_by_label,
    validate_profile_contract,
)


class TestPreprocessCsvToPkl(unittest.TestCase):
    def test_split_by_label_requires_three_splits(self):
        df = pd.DataFrame({"label": ["train", "train", "val"]})
        with self.assertRaises(ValueError):
            split_by_label(df)

    def test_convert_multdim_applies_aliases(self):
        df = pd.DataFrame(
            {
                "HADM_ID": [1, 1],
                "time_scaled": [0.1, 0.2],
                "hr_normalized": [0.0, 0.1],
                "dbp_normalized": [0.0, -0.2],
                "rr_normalized": [0.3, 0.4],
                "age_normalized": [0.8, 0.8],
                "label": ["train", "val"],
            }
        )
        out = convert_dataframe_for_profile(df, "eICU_multdim")
        self.assertIn("hr_normalized_scaled", out.columns)
        self.assertIn("dbp_normalized_scaled", out.columns)
        self.assertIn("rr_normalized_scaled", out.columns)
        self.assertIn("AGE_AT_ADM_normalized", out.columns)
        self.assertIn("time_scaled_v1", out.columns)

    def test_convert_mimic_creates_required_columns(self):
        df = pd.DataFrame(
            {
                "HADM_ID": [1, 2, 3],
                "time_scaled": [0.1, 0.2, 0.3],
                "map_normalized": [0.4, 0.5, 0.6],
                "prbc": [1, 0, 1],
                "pressor": [1, 0, 1],
                "bloodprod": [0, 1, 1],
                "severe_liver": [0, 1, 0],
                "label": ["train", "val", "test"],
            }
        )
        out = convert_dataframe_for_profile(df, "mimic_liver")
        self.assertIn("MAP", out.columns)
        self.assertIn("prbc_outcome", out.columns)
        self.assertIn("1", out.columns)
        self.assertTrue((out["1"] == 1).all())

    def test_validate_profile_contract_raises_when_missing_columns(self):
        df = pd.DataFrame({"HADM_ID": [1], "time_scaled_v1": [0.1], "label": ["train"]})
        with self.assertRaises(ValueError):
            validate_profile_contract(df, PROFILE_SPECS["eICU"])


if __name__ == "__main__":
    unittest.main()
