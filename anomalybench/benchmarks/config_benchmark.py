"""Utilities for running benchmarks from a YAML configuration file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

from dataexcept import ConfigurationError

from anomalybench._io import load_yaml


class ConfigValidationError(ConfigurationError):
    """Raised when a benchmark YAML configuration is invalid.

    Inherits from :class:`dataexcept.ConfigurationError`, so the offending
    key is available as ``option`` in addition to the formatted message.
    """


def _fail(path: str, expected: str, value: Any) -> NoReturn:
    value_type = type(value).__name__
    raise ConfigValidationError(
        path,
        f"Invalid config at '{path}': expected {expected}, got {value_type}.",
    )


_load_plugins = None
_run_benchmarks = None


def _validate_dataset_selector(selector: Any, path: str) -> None:
    if selector is None or isinstance(selector, str):
        return
    if isinstance(selector, (list, tuple)):
        for idx, item in enumerate(selector):
            _validate_dataset_selector(item, f"{path}[{idx}]")
        return
    if isinstance(selector, dict):
        keys = set(selector)
        if {"include", "exclude", "limit", "modality", "task", "label_type"} & keys:
            if "include" in selector:
                _validate_dataset_selector(selector.get("include"), f"{path}.include")
            if "exclude" in selector:
                _validate_dataset_selector(selector.get("exclude"), f"{path}.exclude")
            if "limit" in selector:
                limit = selector.get("limit")
                if not isinstance(limit, int):
                    _fail(f"{path}.limit", "an integer", limit)
            for metadata_key in ("modality", "task", "label_type"):
                value = selector.get(metadata_key)
                if value is None:
                    continue
                if isinstance(value, str):
                    continue
                if isinstance(value, (list, tuple)) and all(
                    isinstance(item, str) for item in value
                ):
                    continue
                _fail(f"{path}.{metadata_key}", "a string or list of strings", value)
            return
        if "name" in selector:
            name = selector.get("name")
            if not isinstance(name, str):
                _fail(f"{path}.name", "a string", name)
            return
        if "tag" in selector:
            tag = selector.get("tag")
            if not isinstance(tag, str):
                _fail(f"{path}.tag", "a string", tag)
            return
        _fail(path, "a dataset selector dictionary", selector)
    _fail(path, "a dataset selector", selector)


def _validate_detector_selector(selector: Any, path: str) -> None:
    if selector is None or isinstance(selector, str):
        return
    if isinstance(selector, (list, tuple)):
        for idx, item in enumerate(selector):
            _validate_detector_selector(item, f"{path}[{idx}]")
        return
    if isinstance(selector, dict):
        keys = set(selector)
        if {"include", "exclude", "defaults"} & keys:
            if "include" in selector:
                _validate_detector_selector(selector.get("include"), f"{path}.include")
            if "exclude" in selector:
                _validate_detector_selector(selector.get("exclude"), f"{path}.exclude")
            if "defaults" in selector:
                defaults = selector.get("defaults")
                if not isinstance(defaults, dict):
                    _fail(f"{path}.defaults", "a mapping", defaults)
                params = defaults.get("params")
                if params is not None and not isinstance(params, dict):
                    _fail(f"{path}.defaults.params", "a mapping", params)
            return
        if "name" in selector:
            name = selector.get("name")
            if not isinstance(name, str):
                _fail(f"{path}.name", "a string", name)
            label = selector.get("label")
            if label is not None and not isinstance(label, str):
                _fail(f"{path}.label", "a string", label)
            params = selector.get("params")
            if params is not None and not isinstance(params, dict):
                _fail(f"{path}.params", "a mapping", params)
            return
        _fail(path, "a detector selector dictionary", selector)
    _fail(path, "a detector selector", selector)


def _validate_metric_selector(selector: Any, path: str) -> None:
    if selector is None or isinstance(selector, str):
        return
    if isinstance(selector, (list, tuple)):
        for idx, item in enumerate(selector):
            if not isinstance(item, str):
                _fail(f"{path}[{idx}]", "a string", item)
        return
    if isinstance(selector, dict):
        allowed_keys = {"include", "names", "positive_label", "k", "threshold"}
        unknown = set(selector) - allowed_keys
        if unknown:
            unknown_keys = ", ".join(sorted(unknown))
            raise ConfigValidationError(
                path,
                f"Invalid config at '{path}': unknown key(s): {unknown_keys}.",
            )
        names = selector.get("include", selector.get("names"))
        if names is not None:
            _validate_metric_selector(names, f"{path}.include")
        k = selector.get("k")
        if k is not None and not isinstance(k, int):
            _fail(f"{path}.k", "an integer", k)
        threshold = selector.get("threshold")
        if threshold is not None and not isinstance(threshold, (int, float)):
            _fail(f"{path}.threshold", "a number", threshold)
        positive_label = selector.get("positive_label")
        if positive_label is not None and not isinstance(positive_label, (int, str)):
            _fail(f"{path}.positive_label", "an integer or string", positive_label)
        return
    _fail(path, "a metric selector", selector)


def _validate_config(config: Any) -> None:
    if not isinstance(config, dict):
        _fail("root", "a mapping", config)

    allowed_keys = {
        "datasets",
        "detectors",
        "n_jobs",
        "leaderboard",
        "plugins",
        "output_dir",
        "json_report",
        "run_id",
        "random_seed",
        "metrics",
    }
    unknown = set(config) - allowed_keys
    if unknown:
        unknown_keys = ", ".join(sorted(unknown))
        raise ConfigValidationError(
            "root",
            f"Invalid config at 'root': unknown key(s): {unknown_keys}.",
        )

    _validate_dataset_selector(config.get("datasets"), "datasets")
    _validate_detector_selector(config.get("detectors"), "detectors")
    _validate_metric_selector(config.get("metrics"), "metrics")

    n_jobs = config.get("n_jobs")
    if n_jobs is not None and not isinstance(n_jobs, int):
        _fail("n_jobs", "an integer", n_jobs)

    leaderboard = config.get("leaderboard")
    if leaderboard is not None and not isinstance(leaderboard, str):
        _fail("leaderboard", "a string", leaderboard)

    output_dir = config.get("output_dir")
    if output_dir is not None and not isinstance(output_dir, str):
        _fail("output_dir", "a string", output_dir)

    json_report = config.get("json_report")
    if json_report is not None and not isinstance(json_report, str):
        _fail("json_report", "a string", json_report)

    run_id = config.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        _fail("run_id", "a string", run_id)

    random_seed = config.get("random_seed")
    if random_seed is not None and not isinstance(random_seed, int):
        _fail("random_seed", "an integer", random_seed)

    plugins = config.get("plugins")
    if plugins is not None:
        if not isinstance(plugins, (list, tuple)):
            _fail("plugins", "a list of strings", plugins)
        for idx, plugin in enumerate(plugins):
            if not isinstance(plugin, str):
                _fail(f"plugins[{idx}]", "a string", plugin)


def run_from_config(
    path: str | Path,
    leaderboard: str | None = None,
    n_jobs: int | None = None,
    output_dir: str | None = None,
    json_report: str | None = None,
    run_id: str | None = None,
    random_seed: int | None = None,
    metrics: Any = None,
) -> None:
    """Execute benchmarks defined in a YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML file containing ``datasets`` and ``detectors`` selectors.
    """
    config_path = Path(path)
    config = load_yaml(config_path) or {}
    _validate_config(config)
    global _load_plugins, _run_benchmarks
    if _load_plugins is None or _run_benchmarks is None:
        from anomalybench.cli import (
            load_plugins as _cli_load_plugins,
        )
        from anomalybench.cli import (
            run_benchmarks as _cli_run_benchmarks,
        )

        _load_plugins = _cli_load_plugins
        _run_benchmarks = _cli_run_benchmarks

    datasets: Any = config.get("datasets")
    detectors: Any = config.get("detectors")
    configured_jobs = config.get("n_jobs")
    effective_jobs = n_jobs if n_jobs is not None else configured_jobs
    configured_leaderboard = config.get("leaderboard")
    effective_leaderboard = (
        leaderboard if leaderboard is not None else configured_leaderboard
    )
    effective_output_dir = (
        output_dir if output_dir is not None else config.get("output_dir")
    )
    effective_json_report = (
        json_report if json_report is not None else config.get("json_report")
    )
    effective_run_id = run_id if run_id is not None else config.get("run_id")
    effective_random_seed = (
        random_seed if random_seed is not None else config.get("random_seed")
    )
    effective_metrics = metrics if metrics is not None else config.get("metrics")

    config_plugins = config.get("plugins")
    if config_plugins:
        _load_plugins(config_plugins)

    _run_benchmarks(
        datasets,
        detectors,
        leaderboard=effective_leaderboard,
        n_jobs=effective_jobs,
        output_dir=effective_output_dir,
        json_report=effective_json_report,
        run_id=effective_run_id,
        random_seed=effective_random_seed,
        metrics=effective_metrics,
    )
