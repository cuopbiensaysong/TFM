#!/usr/bin/env python3
"""Inspect toy template pkl schema and write report artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def split_report(df: pd.DataFrame, split_name: str) -> Dict[str, object]:
    req_cols = ["HADM_ID", "time_scaled_v1", "hr_normalized", "map_normalized"]
    present = [c for c in req_cols if c in df.columns]
    missing = [c for c in req_cols if c not in df.columns]

    sorted_ok = True
    if "HADM_ID" in df.columns and "time_scaled_v1" in df.columns:
        sorted_ok = bool(
            df.sort_values(["HADM_ID", "time_scaled_v1"]).index.equals(df.index)
        )

    return {
        "split": split_name,
        "rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "null_rate_top10": {
            k: float(v) for k, v in df.isna().mean().sort_values(ascending=False).head(10).to_dict().items()
        },
        "required_columns_present": present,
        "required_columns_missing": missing,
        "is_sorted_by_hadm_time_scaled_v1": sorted_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore toy_data.pkl schema.")
    parser.add_argument("--pkl", default="data/toy_data.pkl", help="Path to toy pkl.")
    parser.add_argument("--output-dir", default="dataset_investigation/reports", help="Output report directory.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obj = pd.read_pickle(args.pkl)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict pkl object, got {type(obj)}")

    report: Dict[str, object] = {
        "object_type": str(type(obj)),
        "keys": list(obj.keys()),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        if split not in obj:
            report["splits"][split] = {"error": "missing split"}
            continue
        split_df = obj[split]
        if not isinstance(split_df, pd.DataFrame):
            report["splits"][split] = {"error": f"not a DataFrame: {type(split_df)}"}
            continue
        report["splits"][split] = split_report(split_df, split)

    (output_dir / "toy_template_schema.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote toy template report: {output_dir / 'toy_template_schema.json'}")


if __name__ == "__main__":
    main()
