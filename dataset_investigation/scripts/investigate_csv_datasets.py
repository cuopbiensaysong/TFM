#!/usr/bin/env python3
"""Investigate source CSV datasets and emit profile reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_DATASETS = {
    "eICU_cardiacArrest": "tfm-data/eICU_cardiacArrest_physionet.csv",
    "eICU_sepsis": "tfm-data/eICU_sepsis_physionet.csv",
    "MIMIC_gib": "tfm-data/MIMIC_gib_physionet.csv",
}

PROFILE_COLUMNS = {
    "eICU": ["HADM_ID", "time_scaled_v1", "hr_normalized", "map_normalized", "apache_outcome_prob", "norepi_inf_scaled"],
    "eICU_ablated": ["HADM_ID", "time_scaled_v1", "hr_normalized", "map_normalized", "apache_outcome_prob"],
    "eICU_multdim": ["HADM_ID", "time_scaled_v1", "hr_normalized_scaled", "dbp_normalized_scaled", "rr_normalized_scaled", "AGE_AT_ADM_normalized"],
    "mimic_liver": [
        "HADM_ID",
        "time_scaled_v1",
        "hr_normalized_scaled",
        "map_normalized",
        "prbc_outcome",
        "pressor",
        "bloodprod",
        "severe_liver",
    ],
}


def parse_dictionary_variables(md_path: Path) -> List[str]:
    """Parse variable names from first markdown table column."""
    variables: List[str] = []
    lines = md_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().split("|")]
        if len(cells) < 3:
            continue
        first = cells[1]
        if first in {"Variable Name", "---------------------"}:
            continue
        if first:
            variables.append(first.replace("\\_", "_"))
    return variables


def summarize_dataset(df: pd.DataFrame, dictionary_vars: List[str]) -> Dict[str, object]:
    cols = list(df.columns)
    label_counts = df["label"].value_counts(dropna=False).to_dict() if "label" in df.columns else {}
    unique_patients = int(df["HADM_ID"].nunique()) if "HADM_ID" in df.columns else 0
    time_col = "time_scaled" if "time_scaled" in df.columns else None
    time_stats = {}
    if time_col:
        time_stats = {
            "min": float(df[time_col].min()),
            "max": float(df[time_col].max()),
            "mean": float(df[time_col].mean()),
            "is_nonnegative": bool((df[time_col] >= 0).all()),
        }

    key_missing = df.isna().mean().sort_values(ascending=False).head(10).to_dict()
    dictionary_overlap = sorted(set(cols).intersection(dictionary_vars))
    dictionary_missing = sorted(set(dictionary_vars).difference(cols))

    return {
        "rows": int(len(df)),
        "columns": cols,
        "num_columns": int(len(cols)),
        "unique_patients": unique_patients,
        "label_counts": label_counts,
        "time_stats": time_stats,
        "top_missingness": {k: float(v) for k, v in key_missing.items()},
        "dictionary_overlap_count": int(len(dictionary_overlap)),
        "dictionary_overlap": dictionary_overlap,
        "dictionary_missing_count": int(len(dictionary_missing)),
        "profile_column_coverage": {
            profile: {
                "present": sorted([c for c in needed if c in cols]),
                "missing": sorted([c for c in needed if c not in cols]),
            }
            for profile, needed in PROFILE_COLUMNS.items()
        },
    }


def to_markdown(name: str, report: Dict[str, object]) -> str:
    lines = [
        f"# Dataset Report: {name}",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Columns: `{report['num_columns']}`",
        f"- Unique patients (`HADM_ID`): `{report['unique_patients']}`",
        "",
        "## Label counts",
    ]
    label_counts = report.get("label_counts", {})
    if label_counts:
        for k, v in label_counts.items():
            lines.append(f"- `{k}`: `{v}`")
    else:
        lines.append("- `label` column missing")

    lines += ["", "## Time stats (`time_scaled`)"]
    if report.get("time_stats"):
        for k, v in report["time_stats"].items():
            lines.append(f"- {k}: `{v}`")
    else:
        lines.append("- `time_scaled` column missing")

    lines += ["", "## Profile coverage"]
    profile_cov = report["profile_column_coverage"]
    for profile, cov in profile_cov.items():
        lines.append(f"- `{profile}` present: {', '.join(cov['present']) if cov['present'] else '(none)'}")
        lines.append(f"- `{profile}` missing: {', '.join(cov['missing']) if cov['missing'] else '(none)'}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate source CSV datasets.")
    parser.add_argument("--dictionary", default="tfm-data/data_dictionary.md", help="Path to data dictionary markdown file.")
    parser.add_argument("--output-dir", default="dataset_investigation/reports", help="Directory to write reports.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_vars = parse_dictionary_variables(Path(args.dictionary))

    combined = {}
    for name, dataset_path in DEFAULT_DATASETS.items():
        df = pd.read_csv(dataset_path)
        report = summarize_dataset(df, dictionary_vars)
        combined[name] = report
        (output_dir / f"{name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (output_dir / f"{name}_report.md").write_text(to_markdown(name, report), encoding="utf-8")

    (output_dir / "combined_report.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Wrote reports to: {output_dir}")


if __name__ == "__main__":
    main()
