"""Reproducibility helpers for benchmark runs."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import tomllib
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from dataexcept import DataLoadingError, DataValidationError, ResourceNotFoundError

from anomalybench._io import write_text
from anomalybench.benchmarks.catalog import list_available_datasets, load_catalog

REPORT_SCHEMA_VERSION = "benchmark-report-v1"
MANIFEST_SCHEMA_VERSION = "benchmark-manifest-v1"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BENCHMARK_DIR = Path(__file__).resolve().parent

SEED_PARAMETER_BY_DETECTOR = {
    "isolation_forest": "random_state",
    "elliptic_envelope": "random_state",
    "gaussian_mixture": "random_state",
    "kmeans": "random_state",
    "graph_isolation_forest": "random_state",
    "random_network_distillation": "random_state",
    "random_feature_isolation_forest": "random_state",
    "half_space_trees": "seed",
    "online_isolation_forest": "seed",
}


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> str:
    """Serialize *value* as stable JSON."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any, length: int = 16) -> str:
    """Return a short SHA-256 hash for a JSON-serializable value."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]


def package_version() -> str:
    """Return the project version without requiring an installed package."""

    pyproject_path = _PROJECT_ROOT / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with pyproject_path.open("rb") as fh:
                return str(tomllib.load(fh)["project"]["version"])
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise DataLoadingError(str(pyproject_path), exc) from exc
    try:
        return metadata.version("anomalybench")
    except metadata.PackageNotFoundError:
        return "unknown"


def normalize_run_id(run_id: str | None, timestamp: str, config_hash: str) -> str:
    """Return a user-supplied or deterministic benchmark run identifier."""

    if run_id:
        return run_id
    compact_time = (
        timestamp.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace("T", "-")
    )
    return f"run-{compact_time}-{config_hash[:8]}"


def seed_runtime(seed: int | None) -> None:
    """Seed common Python and NumPy RNGs when a benchmark seed is provided."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def apply_seed_to_detector_entries(
    detector_entries: list[dict[str, Any]],
    seed: int | None,
) -> list[dict[str, Any]]:
    """Add supported seed parameters to detector entries when absent."""

    seeded_entries: list[dict[str, Any]] = []
    for entry in detector_entries:
        copied = dict(entry)
        params = dict(copied.get("params") or {})
        seed_param = SEED_PARAMETER_BY_DETECTOR.get(str(copied.get("name")))
        if seed is not None and seed_param and seed_param not in params:
            params[seed_param] = seed
        copied["params"] = params
        seeded_entries.append(copied)
    return seeded_entries


def benchmark_config_hash(
    dataset_keys: list[str],
    detector_entries: list[dict[str, Any]],
    random_seed: int | None,
    n_jobs: int | None,
    metric_config: dict[str, Any] | None = None,
) -> str:
    """Return the stable hash for the effective benchmark configuration."""

    return stable_hash(
        {
            "dataset_keys": dataset_keys,
            "detectors": detector_entries,
            "random_seed": random_seed,
            "n_jobs": n_jobs,
            "metrics": metric_config,
        }
    )


def _manifest_detector_entries(
    detector_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand effective temporal window defaults for reproducibility."""

    records: list[dict[str, Any]] = []
    for entry in detector_entries:
        copied = dict(entry)
        params = dict(copied.get("params") or {})
        copied["params"] = params
        if "window_length" in params:
            copied["windowing"] = {
                "window_length": int(params["window_length"]),
                "stride": int(params.get("stride", 1)),
                "horizon": int(params.get("horizon", 0)),
                "aggregation": "window",
                "label_alignment": "window_end",
            }
        records.append(copied)
    return records


def build_manifest(
    *,
    run_id: str,
    timestamp: str,
    config_hash: str,
    dataset_keys: list[str],
    detector_entries: list[dict[str, Any]],
    random_seed: int | None,
    n_jobs: int | None,
    output_directory: str | None,
    dataset_integrity: list[dict[str, Any]],
    metric_config: dict[str, Any] | None = None,
    dataset_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the reproducibility manifest for a benchmark run."""

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "run_timestamp_utc": timestamp,
        "package_version": package_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "dataset_keys": dataset_keys,
        "detectors": _manifest_detector_entries(detector_entries),
        "random_seed": random_seed,
        "n_jobs": n_jobs,
        "config_hash": config_hash,
        "output_directory": output_directory,
        "dataset_integrity": dataset_integrity,
        "dataset_metadata": dataset_metadata or [],
        "metrics": metric_config or {},
    }


def build_report(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a versioned benchmark report payload."""

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest": manifest,
        "results": rows,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a reproducibility payload as pretty JSON."""

    # Serialize before touching the file so invalid payloads retain their error.
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def collect_dataset_integrity(names: list[str] | None) -> list[dict[str, Any]]:
    """Collect integrity metadata for bundled files used by selected datasets."""

    catalog = load_catalog()
    selected = names if names is not None else list_available_datasets()
    records: list[dict[str, Any]] = []
    for dataset_key in selected:
        metadata_entry = catalog.get(dataset_key, {})
        files = metadata_entry.get("files") or []
        if not isinstance(files, list):
            continue
        for relative_file in files:
            relative_path = Path(str(relative_file))
            file_path = _BENCHMARK_DIR / relative_path
            if not file_path.exists():
                raise ResourceNotFoundError(
                    "bundled benchmark file", str(relative_path)
                )
            try:
                size = file_path.stat().st_size
            except OSError as exc:
                raise DataLoadingError(str(file_path), exc) from exc
            if size <= 0:
                raise DataValidationError(
                    str(relative_path),
                    0,
                    "bundled benchmark file is empty",
                )
            try:
                digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            except OSError as exc:
                raise DataLoadingError(str(file_path), exc) from exc
            records.append(
                {
                    "dataset_key": dataset_key,
                    "file": str(relative_path).replace("\\", "/"),
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
    return records
