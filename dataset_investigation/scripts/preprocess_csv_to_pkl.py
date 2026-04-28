#!/usr/bin/env python3
"""Convert source CSV files to framework-compatible pkl datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROFILE_SPECS: Dict[str, Dict[str, List[str]]] = {
    "eICU": {
        "required": ["HADM_ID", "time_scaled_v1", "hr_normalized", "map_normalized", "apache_outcome_prob", "norepi_inf_scaled", "label"],
        "ordered_output": ["HADM_ID", "time_scaled_v1", "hr_normalized", "map_normalized", "apache_outcome_prob", "norepi_inf_scaled", "label"],
    },
    "eICU_ablated": {
        "required": ["HADM_ID", "time_scaled_v1", "hr_normalized", "map_normalized", "apache_outcome_prob", "label"],
        "ordered_output": ["HADM_ID", "time_scaled_v1", "hr_normalized", "map_normalized", "apache_outcome_prob", "label"],
    },
    "eICU_multdim": {
        "required": [
            "HADM_ID",
            "time_scaled_v1",
            "hr_normalized_scaled",
            "dbp_normalized_scaled",
            "rr_normalized_scaled",
            "AGE_AT_ADM_normalized",
            "label",
        ],
        "ordered_output": [
            "HADM_ID",
            "time_scaled_v1",
            "hr_normalized_scaled",
            "dbp_normalized_scaled",
            "rr_normalized_scaled",
            "AGE_AT_ADM_normalized",
            "label",
        ],
    },
    "mimic_liver": {
        "required": ["HADM_ID", "time_scaled_v1", "1", "MAP", "prbc_outcome", "pressor", "bloodprod", "severe_liver", "label"],
        "ordered_output": ["HADM_ID", "time_scaled_v1", "1", "MAP", "prbc_outcome", "pressor", "bloodprod", "severe_liver", "label"],
    },
}


def normalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time_scaled_v1" not in df.columns:
        if "time_scaled" not in df.columns:
            raise ValueError("Missing both `time_scaled` and `time_scaled_v1`.")
        df["time_scaled_v1"] = pd.to_numeric(df["time_scaled"], errors="coerce")

    if "label" not in df.columns:
        raise ValueError("Missing `label` column required for split assignment.")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df


def apply_profile_aliases(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    df = df.copy()
    if profile == "eICU_multdim":
        aliases = {
            "hr_normalized_scaled": "hr_normalized",
            "dbp_normalized_scaled": "dbp_normalized",
            "rr_normalized_scaled": "rr_normalized",
            "AGE_AT_ADM_normalized": "age_normalized",
        }
        for target, src in aliases.items():
            if target not in df.columns and src in df.columns:
                df[target] = pd.to_numeric(df[src], errors="coerce")

    if profile == "mimic_liver":
        if "MAP" not in df.columns:
            if "map_normalized" in df.columns:
                df["MAP"] = pd.to_numeric(df["map_normalized"], errors="coerce")
            elif "map" in df.columns:
                df["MAP"] = pd.to_numeric(df["map"], errors="coerce")

        if "prbc_outcome" not in df.columns and "prbc" in df.columns:
            df["prbc_outcome"] = pd.to_numeric(df["prbc"], errors="coerce")

        if "1" not in df.columns:
            df["1"] = 1.0

    return df


def validate_profile_contract(df: pd.DataFrame, spec: Dict[str, List[str]]) -> None:
    missing = [c for c in spec["required"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for profile: {missing}")


def split_by_label(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    split_values = set(df["label"].dropna().unique().tolist())
    needed = {"train", "val", "test"}
    if not needed.issubset(split_values):
        raise ValueError(f"`label` must include train/val/test; found {sorted(split_values)}")

    out = {}
    for split in ["train", "val", "test"]:
        split_df = df[df["label"] == split].copy()
        split_df = split_df.sort_values(["HADM_ID", "time_scaled_v1"]).reset_index(drop=True)
        out[split] = split_df
    return out


def convert_dataframe_for_profile(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    if profile not in PROFILE_SPECS:
        raise ValueError(f"Unknown profile: {profile}")
    df = normalize_common_columns(df)
    df = apply_profile_aliases(df, profile)
    validate_profile_contract(df, PROFILE_SPECS[profile])
    ordered = PROFILE_SPECS[profile]["ordered_output"]
    return df[ordered].copy()


def write_pkl_from_csv(csv_path: Path, output_path: Path, profile: str) -> None:
    df = pd.read_csv(csv_path)
    converted = convert_dataframe_for_profile(df, profile)
    split_dict = split_by_label(converted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(split_dict, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV to framework-ready pkl.")
    parser.add_argument("--csv", required=True, help="Input CSV path.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILE_SPECS.keys()), help="Profile name.")
    parser.add_argument("--output", required=True, help="Output pkl path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_pkl_from_csv(Path(args.csv), Path(args.output), args.profile)
    print(f"Wrote {args.output} ({args.profile})")


if __name__ == "__main__":
    main()
