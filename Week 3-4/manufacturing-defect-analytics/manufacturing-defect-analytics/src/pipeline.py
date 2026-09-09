"""
pipeline.py
-----------
OOP pipeline for the Kaggle "Predicting Manufacturing Defects Dataset"
(https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset).

Design notes:
- `ManufacturingDataset` wraps the raw CSV and exposes chainable methods:
  .load().clean().engineer_features() so a caller can do:
      ds = ManufacturingDataset().load().clean().engineer_features()
- List/dict comprehensions are used deliberately in a couple of spots
  (per the "Python essentials" syllabus item) instead of explicit loops.
- Even though this dataset ships clean (no nulls/dupes upstream), clean()
  still performs real defensive checks -- that's what you'd do with any
  dataset you didn't generate yourself, real-world or not.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class CleaningReport:
    """Small value object recording what clean() actually found/did, so the
    notebook can print a human-readable audit trail instead of silently
    trusting the input file."""
    rows_before: int = 0
    rows_after: int = 0
    duplicates_removed: int = 0
    missing_before: dict = field(default_factory=dict)
    rows_out_of_range: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Rows before cleaning     : {self.rows_before}",
            f"Duplicate rows removed   : {self.duplicates_removed}",
            f"Rows after cleaning      : {self.rows_after}",
        ]
        any_missing = any(v > 0 for v in self.missing_before.values())
        if any_missing:
            lines += [f"Missing values in '{c}': {n}"
                      for c, n in self.missing_before.items() if n > 0]
        else:
            lines.append("Missing values found     : none")

        any_oor = any(v > 0 for v in self.rows_out_of_range.values())
        if any_oor:
            lines += [f"Out-of-range rows in '{c}': {n}"
                      for c, n in self.rows_out_of_range.items() if n > 0]
        else:
            lines.append("Out-of-range values found: none")
        return "\n".join(lines)


class ManufacturingDataset:
    """Owns the raw + cleaned/engineered table for one analysis run."""

    RAW_FEATURES = [
        "ProductionVolume", "ProductionCost", "SupplierQuality", "DeliveryDelay",
        "DefectRate", "QualityScore", "MaintenanceHours", "DowntimePercentage",
        "InventoryTurnover", "StockoutRate", "WorkerProductivity", "SafetyIncidents",
        "EnergyConsumption", "EnergyEfficiency", "AdditiveProcessTime", "AdditiveMaterialCost",
    ]
    TARGET = "DefectStatus"

    # Known plausible ranges for a couple of percentage/rate-like columns,
    # used only to flag obviously corrupt rows (e.g. a negative rate) --
    # not to silently drop legitimate outliers.
    RANGE_CHECKS = {
        "SupplierQuality": (0, 100),
        "QualityScore": (0, 100),
        "WorkerProductivity": (0, 100),
        "StockoutRate": (0, 1),
    }

    def __init__(self, csv_path: str = "../data/manufacturing_defect_dataset.csv"):
        self.csv_path = csv_path
        self.raw: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.report = CleaningReport()

    # ------------------------------------------------------------------
    def load(self) -> "ManufacturingDataset":
        self.raw = pd.read_csv(self.csv_path)
        return self

    # ------------------------------------------------------------------
    def clean(self) -> "ManufacturingDataset":
        df = self.raw.copy()
        self.report.rows_before = len(df)

        # Missing-value audit (dict comprehension over columns)
        self.report.missing_before = {col: int(df[col].isna().sum()) for col in df.columns}
        # Any numeric column's missing values get filled with the median --
        # defensive, even though this particular file has none.
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # Duplicate rows
        before = len(df)
        df = df.drop_duplicates()
        self.report.duplicates_removed = before - len(df)

        # Range sanity checks -- flag, then clip rather than silently drop,
        # so a genuine data-entry error doesn't quietly remove a whole row.
        for col, (lo, hi) in self.RANGE_CHECKS.items():
            out_of_range = int(((df[col] < lo) | (df[col] > hi)).sum())
            self.report.rows_out_of_range[col] = out_of_range
            df[col] = df[col].clip(lo, hi)

        self.report.rows_after = len(df)
        self.df = df.reset_index(drop=True)
        return self

    # ------------------------------------------------------------------
    def engineer_features(self) -> "ManufacturingDataset":
        df = self.df.copy()

        # Composite "process health" score - simple weighted combination of
        # quality/efficiency-positive metrics vs. cost/risk-negative ones.
        df["process_health_score"] = (
            df["QualityScore"] * 0.35
            + df["SupplierQuality"] * 0.25
            + df["WorkerProductivity"] * 0.20
            + df["EnergyEfficiency"] * 100 * 0.20
        ).round(2)

        # list comprehension: bucket maintenance hours into a readable label
        df["maintenance_band"] = [
            "low" if h <= 7 else "medium" if h <= 15 else "high"
            for h in df["MaintenanceHours"]
        ]

        # dict comprehension: human-readable label for the target
        status_labels = {0: "Good", 1: "Defective"}
        df["defect_label"] = df["DefectStatus"].map(status_labels)

        df["cost_per_unit"] = (df["ProductionCost"] / df["ProductionVolume"]).round(2)
        df["energy_per_unit"] = (df["EnergyConsumption"] / df["ProductionVolume"]).round(3)

        self.df = df
        return self

    # ------------------------------------------------------------------
    def save(self, path: str = "../data/manufacturing_clean.csv") -> "ManufacturingDataset":
        self.df.to_csv(path, index=False)
        return self


if __name__ == "__main__":
    ds = ManufacturingDataset(csv_path="../data/manufacturing_defect_dataset.csv") \
            .load().clean().engineer_features()
    print(ds.report.summary())
    ds.save()
    print(f"\nFinal shape: {ds.df.shape}")
