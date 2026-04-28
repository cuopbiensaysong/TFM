#!/usr/bin/env python3
"""Validate source CSV compatibility for each configured profile."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dataset_investigation.scripts.preprocess_csv_to_pkl import (
    PROFILE_SPECS,
    convert_dataframe_for_profile,
)


CSV_BY_PROFILE = {
    "eICU": "tfm-data/eICU_sepsis_physionet.csv",
    "eICU_ablated": "tfm-data/eICU_sepsis_physionet.csv",
    "eICU_multdim": "tfm-data/eICU_cardiacArrest_physionet.csv",
    "mimic_liver": "tfm-data/MIMIC_gib_physionet.csv",
}


def main() -> None:
    report = {}
    for profile, csv_path in CSV_BY_PROFILE.items():
        raw = pd.read_csv(csv_path, nrows=2000)
        try:
            converted = convert_dataframe_for_profile(raw, profile)
            report[profile] = {
                "ok": True,
                "source_csv": csv_path,
                "rows_checked": int(len(raw)),
                "output_columns": list(converted.columns),
                "required_columns": PROFILE_SPECS[profile]["required"],
            }
        except Exception as exc:  # noqa: BLE001
            report[profile] = {
                "ok": False,
                "source_csv": csv_path,
                "error": str(exc),
                "required_columns": PROFILE_SPECS[profile]["required"],
            }

    out_path = Path("dataset_investigation/reports/profile_validation_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
