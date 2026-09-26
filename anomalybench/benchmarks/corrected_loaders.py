"""Correctness-focused benchmark dataset loaders.

This module contains loaders whose semantics intentionally differ from the
legacy implementations in :mod:`anomalybench.benchmarks.load_datasets`. Keeping the
correction isolated makes the behavioural change explicit and easy to test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dataexcept import DataValidationError

from anomalybench._io import load_csv as _read_csv

_BENCHMARK_DIR = Path(__file__).resolve().parent


def load_cardio_corrected() -> tuple[pd.DataFrame, list[str], str, str]:
    """Load Cardiotocography with pathological records as anomalies.

    The UCI ``NSP`` target uses ``1=normal``, ``2=suspect`` and
    ``3=pathological``. The benchmark contract discards suspect observations,
    keeps normal and pathological observations, and encodes pathological cases
    as anomaly label ``0`` to preserve the repository's historical anomaly
    label convention for this dataset family.
    """

    df = _read_csv(_BENCHMARK_DIR / "ctg.csv")
    df = df.dropna(axis=1, how="all").dropna()

    # Keep only normal and pathological cases. Suspect cases are deliberately
    # excluded from the anomaly-detection benchmark.
    df = df[df["NSP"].isin([1, 3])].copy()
    df["Class"] = df["NSP"].map({1: 1, 3: 0})
    df = df.drop(columns=["NSP"])

    if df["Class"].isna().any():
        raise DataValidationError(
            "Class", None, "cardiotocography class mapping produced missing labels"
        )
    if set(df["Class"].unique()) != {0, 1}:
        raise DataValidationError(
            "Class",
            sorted(df["Class"].unique()),
            "cardiotocography benchmark must contain both classes",
        )

    features = [column for column in df.columns if column != "Class"]
    return df, features, "Class", "cardio"
