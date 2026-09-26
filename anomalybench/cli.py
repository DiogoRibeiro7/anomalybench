"""Command line interface to run anomaly detection benchmarks.

Use ``--summary`` to display information about the available datasets instead
of running the detectors.
"""

import argparse
import csv
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path
from typing import Any

from anomalybench.analytics.runtime import ensure_supported_python

ensure_supported_python()

import networkx as nx
import pandas as pd
from dataexcept import (
    ConfigurationError,
    DataFormatError,
    DataValidationError,
    DependencyError,
    FileWriteError,
    HyperparameterError,
)

from anomalybench._io import ensure_directory
from anomalybench.analytics.detectors import (
    DETECTOR_REGISTRY,
    get_detector_class,
)
from anomalybench.analytics.exceptions import DetectorNotFittedError
from anomalybench.benchmarks.catalog import resolve_dataset_names
from anomalybench.benchmarks.load_all_datasets import load_all_datasets
from anomalybench.benchmarks.metrics import (
    DEFAULT_SCORE_ORIENTATION,
    MetricConfig,
    evaluate_metrics,
    resolve_metric_config,
)
from anomalybench.benchmarks.reproducibility import (
    apply_seed_to_detector_entries,
    benchmark_config_hash,
    build_manifest,
    build_report,
    collect_dataset_integrity,
    normalize_run_id,
    seed_runtime,
    utc_timestamp,
    write_json,
)

logger = logging.getLogger(__name__)
LEADERBOARD_HEADER = [
    "run_timestamp_utc",
    "run_id",
    "config_hash",
    "dataset_name",
    "dataset_key",
    "detector_name",
    "detector_label",
    "detector_params",
    "score_orientation",
    "random_seed",
    "runtime_seconds",
    "failure_category",
    "metrics",
    "auc",
    "error",
]


def summarize_datasets(datasets=None):
    """Print basic information about the available datasets."""

    resolved = resolve_dataset_names(datasets)
    try:
        dataset_entries = load_all_datasets(resolved)
    except KeyError as exc:  # pragma: no cover - defensive logging
        logger.error("%s", exc)
        raise

    for ds in dataset_entries:
        name = ds["name"]
        df = ds["dataframe"]
        features = ds["feature_cols"]
        labels = ds["label_col"]
        metadata = ds.get("metadata") or {}
        print(f"Dataset: {name}")
        if isinstance(df, pd.DataFrame):
            counts = df[labels].value_counts().to_dict()
            print(f"  samples: {len(df)}")
            print(f"  features: {len(features or [])}")
            print(f"  label distribution: {counts}")
        else:
            counts = Counter(nx.get_node_attributes(df, labels).values())
            print(f"  nodes: {df.number_of_nodes()}")
            print(f"  edges: {df.number_of_edges()}")
            print(f"  label distribution: {dict(counts)}")
        tags = metadata.get("tags")
        if isinstance(tags, (list, tuple)) and tags:
            print(f"  tags: {', '.join(str(tag) for tag in tags)}")
        source = metadata.get("source")
        if source:
            print(f"  source: {source}")
        task = metadata.get("task")
        if task:
            print(f"  task: {task}")
        modality = metadata.get("modality")
        if modality:
            print(f"  modality: {modality}")
        label_type = metadata.get("label_type")
        if label_type:
            print(f"  label type: {label_type}")
        print()


# Detector instances are created on demand to avoid importing optional
# dependencies unless required.


ALLOWED_PLUGIN_PREFIX = "plugins."


def load_plugins(modules):
    """Import plugin modules while restricting the allowed namespace."""
    for mod in modules:
        if not mod.startswith(ALLOWED_PLUGIN_PREFIX):
            raise ConfigurationError(
                "plugins",
                f"plugin '{mod}' is not allowed; "
                f"must start with '{ALLOWED_PLUGIN_PREFIX}'",
            )
        import_module(mod)


def _resolve_worker_count(total_tasks, requested):
    """Return the number of worker threads to use for the benchmark run."""

    if total_tasks <= 0:
        return 1
    if requested is None:
        return 1
    if requested <= 0:
        return min(total_tasks, os.cpu_count() or 1)
    return min(total_tasks, requested)


def _format_detector_display(entry: dict[str, object]) -> str:
    label = entry.get("label") or entry["name"]
    params = entry.get("params") or {}
    if not isinstance(params, dict) or not params:
        return str(label)
    param_bits = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"{label} ({param_bits})"


def _format_metric_values(metrics: dict[str, float | None]) -> str:
    if not metrics:
        return "metrics unavailable"
    bits: list[str] = []
    for name, value in metrics.items():
        display_name = "AUC" if name == "roc_auc" else name
        if value is None:
            bits.append(f"{display_name}=n/a")
        else:
            bits.append(f"{display_name}={value:.3f}")
    return ", ".join(bits)


def _metric_args_from_cli(args):
    if (
        args.metrics is None
        and args.metric_k is None
        and args.metric_threshold is None
        and args.positive_label is None
    ):
        return None
    metric_config: dict[str, object] = {}
    if args.metrics is not None:
        metric_config["include"] = args.metrics or ["roc_auc"]
    if args.metric_k is not None:
        metric_config["k"] = args.metric_k
    if args.metric_threshold is not None:
        metric_config["threshold"] = args.metric_threshold
    if args.positive_label is not None:
        metric_config["positive_label"] = _parse_label_value(args.positive_label)
    return metric_config


def _parse_label_value(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def _resolve_detector_entries(selection):
    if selection is None:
        return [
            {"name": name, "params": {}, "label": name} for name in DETECTOR_REGISTRY
        ]

    if isinstance(selection, dict) and {"include", "exclude", "defaults"} & set(
        selection
    ):
        defaults = selection.get("defaults", {})
        default_params = {}
        if isinstance(defaults, dict):
            default_params = dict(defaults.get("params", {}))
        include_spec = selection.get("include")
        if include_spec is None:
            include_entries = [
                {"name": name, "params": {}, "label": name}
                for name in DETECTOR_REGISTRY
            ]
        else:
            include_entries = _resolve_detector_entries(include_spec)
        exclude_spec = selection.get("exclude")
        excluded = (
            {entry["name"] for entry in _resolve_detector_entries(exclude_spec)}
            if exclude_spec
            else set()
        )
        resolved: list[dict[str, object]] = []
        for entry in include_entries:
            name = entry["name"]
            if name not in DETECTOR_REGISTRY or name in excluded:
                continue
            params = dict(default_params)
            entry_params = entry.get("params") or {}
            if isinstance(entry_params, dict):
                params.update(entry_params)
            resolved.append(
                {
                    "name": name,
                    "params": params,
                    "label": entry.get("label") or entry.get("name", name),
                }
            )
        return resolved

    if isinstance(selection, (list, tuple)):
        resolved = []
        for item in selection:
            resolved.extend(_resolve_detector_entries(item))
        return resolved

    if isinstance(selection, str):
        if selection not in DETECTOR_REGISTRY:
            return []
        return [{"name": selection, "params": {}, "label": selection}]

    if isinstance(selection, dict) and "name" in selection:
        selected_name = selection.get("name")
        if selected_name not in DETECTOR_REGISTRY:
            return []
        params = dict(selection.get("params", {}))
        label = selection.get("label") or selected_name
        return [{"name": selected_name, "params": params, "label": label}]

    return []


def _evaluated_score_orientation(scores) -> str:
    """Return the orientation benchmark evaluation actually applies to *scores*.

    ``BaseDetector.detect_anomalies`` returns ``OrientedScores`` carrying the
    detector's own orientation. A detector that overrides it may return a plain
    array, which is evaluated as ``DEFAULT_SCORE_ORIENTATION``. Reporting the
    class attribute in that case would disagree with how the metrics were
    computed, so the applied orientation is what gets recorded.
    """

    return getattr(scores, "score_orientation", DEFAULT_SCORE_ORIENTATION)


def _classify_failure(exc: Exception | None) -> str:
    """Categorise a detector failure for the benchmark report.

    Both the DataExcept types this package raises and the builtin types that
    third-party detectors still raise are recognised, so the category does not
    depend on which library produced the error.
    """

    if exc is None:
        return ""
    if isinstance(exc, (DependencyError, ImportError, ModuleNotFoundError)):
        return "missing_dependency"
    if isinstance(
        exc,
        (
            ConfigurationError,
            DataFormatError,
            DataValidationError,
            HyperparameterError,
            TypeError,
            ValueError,
        ),
    ):
        return "invalid_input_or_parameter"
    if isinstance(exc, (DetectorNotFittedError, RuntimeError)):
        return "runtime_error"
    return "detector_error"


def _result_rows(
    *,
    timestamp: str,
    run_id: str,
    config_hash: str,
    random_seed: int | None,
    metric_config: MetricConfig,
    datasets_to_run,
    detector_entries,
    results,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_spec in datasets_to_run:
        dataset_name = dataset_spec["name"]
        dataset_key = dataset_spec.get("key") or dataset_name
        for entry_index, det_spec in enumerate(detector_entries):
            label = det_spec.get("label") or det_spec["name"]
            outcome = results.get(dataset_name, {}).get(entry_index)
            if not outcome:
                continue
            params = det_spec.get("params") or {}
            rows.append(
                {
                    "run_timestamp_utc": timestamp,
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "dataset_name": dataset_name,
                    "dataset_key": dataset_key,
                    "detector_name": det_spec["name"],
                    "detector_label": label,
                    "detector_params": params,
                    "score_orientation": outcome["score_orientation"],
                    "random_seed": random_seed,
                    "runtime_seconds": outcome["runtime_seconds"],
                    "failure_category": outcome["failure_category"],
                    "metrics": outcome["metrics"],
                    "auc": outcome["auc"],
                    "error": outcome["error"] or "",
                }
            )
    return rows


def _manifest_dataset_metadata(datasets_to_run) -> list[dict[str, object]]:
    metadata_records: list[dict[str, object]] = []
    for dataset_spec in datasets_to_run:
        metadata = dict(dataset_spec.get("metadata") or {})
        metadata_records.append(
            {
                "dataset_key": dataset_spec.get("key") or dataset_spec["name"],
                "dataset_name": dataset_spec["name"],
                "modality": metadata.get("modality"),
                "task": metadata.get("task"),
                "label_type": metadata.get("label_type"),
                "positive_label": metadata.get("positive_label"),
                "label_semantics": metadata.get("label_semantics"),
                "tags": metadata.get("tags") or [],
            }
        )
    return metadata_records


def _append_leaderboard_rows(
    leaderboard_path: str | Path, rows: list[dict[str, Any]]
) -> None:
    """Append benchmark rows, reporting filesystem errors with the CSV path."""

    destination = Path(leaderboard_path)
    ensure_directory(destination.parent)
    try:
        write_header = not destination.exists() or destination.stat().st_size == 0
        with destination.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LEADERBOARD_HEADER)
            if write_header:
                writer.writeheader()

            for row in rows:
                csv_row = dict(row)
                csv_row["detector_params"] = json.dumps(
                    csv_row["detector_params"], sort_keys=True
                )
                csv_row["metrics"] = json.dumps(csv_row["metrics"], sort_keys=True)
                csv_row["random_seed"] = (
                    "" if csv_row["random_seed"] is None else csv_row["random_seed"]
                )
                csv_row["auc"] = "" if csv_row["auc"] is None else csv_row["auc"]
                writer.writerow(csv_row)
    except OSError as exc:
        raise FileWriteError(str(destination), exc) from exc


def run_benchmarks(
    datasets=None,
    detectors=None,
    leaderboard=None,
    n_jobs=None,
    output_dir=None,
    json_report=None,
    run_id=None,
    random_seed=None,
    metrics=None,
):
    """Run benchmarks for the specified datasets and detectors.

    Parameters
    ----------
    datasets: Any
        Dataset selectors to evaluate. ``None`` evaluates all available datasets.
        Selectors may be dataset names, ``tag:<name>`` expressions, or dictionaries
        with ``include``/``exclude``/``limit`` keys.
    detectors: Any
        Detector selectors to evaluate. ``None`` evaluates all registered
        detectors. Detector selectors may be detector names, dictionaries with a
        ``name``/``params`` pair, or configuration dictionaries containing
        ``include``/``exclude``/``defaults``.
    leaderboard: str | None
        Optional path to a CSV file where results are appended.
    n_jobs: int | None
        Number of worker threads to use. ``None`` or ``1`` runs sequentially.
        Non-positive values use the available CPU count.
    output_dir: str | None
        Optional directory where manifest and default report JSON files are
        written.
    json_report: str | None
        Optional explicit path for the JSON benchmark report.
    run_id: str | None
        Optional deterministic run identifier.
    random_seed: int | None
        Optional random seed applied to supported detectors and common RNGs.
    metrics: Any
        Optional metric selector. Supports a string, list of strings, or mapping
        with ``include``, ``positive_label``, ``k``, and ``threshold`` keys.
    """
    metric_config = resolve_metric_config(metrics)
    detector_entries = _resolve_detector_entries(detectors)
    if not detector_entries:
        detector_entries = [
            {"name": name, "params": {}, "label": name} for name in DETECTOR_REGISTRY
        ]
    detector_entries = apply_seed_to_detector_entries(detector_entries, random_seed)

    resolved_datasets = resolve_dataset_names(datasets)
    dataset_integrity = collect_dataset_integrity(resolved_datasets)
    try:
        datasets_to_run = list(load_all_datasets(resolved_datasets))
    except KeyError as exc:
        logger.error("%s", exc)
        raise

    if not datasets_to_run:
        return

    seed_runtime(random_seed)
    dataset_keys = [str(ds.get("key") or ds["name"]) for ds in datasets_to_run]
    config_hash = benchmark_config_hash(
        dataset_keys,
        detector_entries,
        random_seed,
        n_jobs,
        metric_config.as_dict(),
    )
    timestamp = utc_timestamp()
    effective_run_id = normalize_run_id(run_id, timestamp, config_hash)

    total_tasks = len(datasets_to_run) * len(detector_entries)
    worker_count = _resolve_worker_count(total_tasks, n_jobs)

    results: dict[Any, dict[int, dict[str, Any]]] = {
        ds["name"]: {} for ds in datasets_to_run
    }

    def _evaluate(dataset_spec, det_spec, entry_index):
        name = dataset_spec["name"]
        df = dataset_spec["dataframe"]
        features = dataset_spec["feature_cols"]
        labels = dataset_spec["label_col"]
        detector_cls = get_detector_class(det_spec["name"])
        score_orientation = getattr(
            detector_cls, "score_orientation", "estimator_defined"
        )
        params = dict(det_spec.get("params") or {})
        fit_params: dict[str, object] = {}
        try:
            detector = detector_cls(**params)
        except TypeError:
            detector = detector_cls()
            fit_params = params
        started = time.perf_counter()
        try:
            if isinstance(df, pd.DataFrame):
                scores = detector.detect_anomalies(df[features], **fit_params)
                y_true = df[labels]
            else:
                scores = detector.detect_anomalies(df, **fit_params)
                y_true = [data[labels] for _, data in df.nodes(data=True)]
            evaluated_orientation = _evaluated_score_orientation(scores)
            runtime_seconds = round(time.perf_counter() - started, 6)
            metric_values = evaluate_metrics(
                y_true,
                scores,
                runtime_seconds=runtime_seconds,
                config=metric_config,
            )
            auc = metric_values.get("roc_auc")
            return (
                name,
                det_spec,
                entry_index,
                auc,
                metric_values,
                None,
                "",
                runtime_seconds,
                evaluated_orientation,
            )
        except Exception as exc:  # pragma: no cover - benchmarking utility
            runtime_seconds = round(time.perf_counter() - started, 6)
            return (
                name,
                det_spec,
                entry_index,
                None,
                {},
                str(exc),
                _classify_failure(exc),
                runtime_seconds,
                score_orientation,
            )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_evaluate, dataset_spec, det_spec, entry_index)
            for dataset_spec in datasets_to_run
            for entry_index, det_spec in enumerate(detector_entries)
        ]
        for future in as_completed(futures):
            (
                dataset_name,
                det_spec,
                entry_index,
                auc,
                metric_values,
                error,
                failure_category,
                runtime_seconds,
                score_orientation,
            ) = future.result()
            results[dataset_name][entry_index] = {
                "auc": auc,
                "metrics": metric_values,
                "error": error,
                "failure_category": failure_category,
                "runtime_seconds": runtime_seconds,
                "score_orientation": score_orientation,
                "detector": det_spec,
            }

    for dataset_spec in datasets_to_run:
        name = dataset_spec["name"]
        print(f"Dataset: {name}")
        dataset_results = results.get(name, {})
        for entry_index, det_spec in enumerate(detector_entries):
            label = det_spec.get("label") or det_spec["name"]
            outcome = dataset_results.get(entry_index)
            if not outcome:
                print(f"  {label}: skipped (not evaluated)")
                continue
            if outcome["error"] is None:
                display = _format_detector_display(det_spec)
                metric_bits = _format_metric_values(outcome["metrics"])
                print(f"  {display}: {metric_bits}")
            else:
                print(f"  {label}: skipped ({outcome['error']})")
        print()

    rows = _result_rows(
        timestamp=timestamp,
        run_id=effective_run_id,
        config_hash=config_hash,
        random_seed=random_seed,
        metric_config=metric_config,
        datasets_to_run=datasets_to_run,
        detector_entries=detector_entries,
        results=results,
    )
    manifest = build_manifest(
        run_id=effective_run_id,
        timestamp=timestamp,
        config_hash=config_hash,
        dataset_keys=dataset_keys,
        detector_entries=detector_entries,
        random_seed=random_seed,
        n_jobs=n_jobs,
        output_directory=str(output_dir) if output_dir else None,
        dataset_integrity=dataset_integrity,
        metric_config=metric_config.as_dict(),
        dataset_metadata=_manifest_dataset_metadata(datasets_to_run),
    )
    report = build_report(manifest, rows)

    if leaderboard:
        _append_leaderboard_rows(leaderboard, rows)

    output_path = Path(output_dir) if output_dir else None
    if output_path:
        ensure_directory(output_path)
        write_json(output_path / f"{effective_run_id}-manifest.json", manifest)
    if json_report:
        write_json(json_report, report)
    elif output_path:
        write_json(output_path / f"{effective_run_id}-report.json", report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection benchmarks")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Names of datasets to benchmark. Defaults to all",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show dataset summaries instead of running benchmarks",
    )
    parser.add_argument(
        "--detectors",
        nargs="*",
        help="Detector names to run. Defaults to all",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML configuration specifying datasets and detectors",
    )
    parser.add_argument(
        "--plugins",
        nargs="*",
        help="Plugin modules providing additional detectors",
    )
    parser.add_argument(
        "--leaderboard",
        help="CSV file path to append benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for reproducibility manifest and default JSON report files",
    )
    parser.add_argument(
        "--json-report",
        help="Path to write a versioned JSON benchmark report",
    )
    parser.add_argument(
        "--run-id",
        help="Stable identifier to store in manifests, reports, and leaderboards",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Seed supported detectors and common Python/NumPy RNGs",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        help=(
            "Metrics to compute. Supported values include roc_auc, "
            "average_precision, precision_at_k, recall_at_k, "
            "f1_at_threshold, best_f1, and runtime."
        ),
    )
    parser.add_argument(
        "--metric-k",
        type=int,
        help="Top-k value for precision_at_k and recall_at_k metrics",
    )
    parser.add_argument(
        "--metric-threshold",
        type=float,
        help="Score threshold for f1_at_threshold",
    )
    parser.add_argument(
        "--positive-label",
        default=None,
        help="Label value treated as positive when computing metrics",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Number of worker threads for detector execution. "
            "Use a non-positive value to leverage all available CPUs."
        ),
    )
    args = parser.parse_args()
    if args.plugins:
        load_plugins(args.plugins)
    if args.config:
        from anomalybench.benchmarks.config_benchmark import run_from_config

        run_from_config(
            args.config,
            leaderboard=args.leaderboard,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
            json_report=args.json_report,
            run_id=args.run_id,
            random_seed=args.random_seed,
            metrics=_metric_args_from_cli(args),
        )
    elif args.summary:
        summarize_datasets(args.datasets)
    else:
        run_benchmarks(
            args.datasets,
            args.detectors,
            leaderboard=args.leaderboard,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
            json_report=args.json_report,
            run_id=args.run_id,
            random_seed=args.random_seed,
            metrics=_metric_args_from_cli(args),
        )


if __name__ == "__main__":
    main()
