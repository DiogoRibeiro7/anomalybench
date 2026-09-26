"""Dataset catalog metadata and selection utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache
from inspect import getmembers, isfunction
from pathlib import Path
from typing import Any, TypedDict

from dataexcept import SchemaMismatchError

from anomalybench._io import load_yaml
from anomalybench.benchmarks import load_datasets
from anomalybench.benchmarks.corrected_loaders import load_cardio_corrected

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetLoader",
    "DatasetRegistry",
    "DatasetSpec",
    "get_dataset_functions",
    "load_catalog",
    "get_dataset_metadata",
    "list_available_datasets",
    "resolve_dataset_names",
]

_CATALOG_PATH = Path(__file__).with_name("datasets.yml")

type DatasetLoader = Callable[[], tuple[Any, list[str] | None, str, str]]
type DatasetRegistry = dict[str, DatasetLoader]


class DatasetSpec(TypedDict):
    """Structured dataset entry returned by ``load_all_datasets``."""

    dataframe: Any
    feature_cols: list[str] | None
    label_col: str
    name: str
    key: str
    metadata: dict[str, object]


@lru_cache(maxsize=1)
def get_dataset_functions() -> DatasetRegistry:
    """Return mapping of canonical dataset names to loader callables."""

    functions: dict[str, DatasetLoader] = {}
    for func_name, func in getmembers(load_datasets, isfunction):
        if not func_name.startswith("load_"):
            continue
        key = func_name.replace("load_", "")
        functions[key] = func

    # The legacy Cardiotocography loader accidentally discarded pathological
    # observations. Override that single registry entry with the corrected,
    # contract-tested implementation while preserving all public selector keys.
    functions["cardio"] = load_cardio_corrected
    return functions


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, dict[str, object]]:
    """Load the dataset metadata catalog from disk."""

    if not _CATALOG_PATH.exists():
        return {}
    data = load_yaml(_CATALOG_PATH) or {}
    if not isinstance(data, dict):
        raise SchemaMismatchError("catalog mapping", type(data).__name__)
    raw_catalog = data.get("datasets", {})
    catalog: dict[str, dict[str, object]] = {}
    for name, details in raw_catalog.items():
        meta = dict(details or {})
        aliases = meta.get("aliases") or []
        if isinstance(aliases, (list, tuple)):
            meta["aliases"] = [str(alias) for alias in aliases if alias]
        else:
            meta["aliases"] = []
        catalog[str(name)] = meta
    return catalog


def get_dataset_metadata(name: str) -> dict[str, object]:
    """Return metadata for *name* from the catalog."""

    return load_catalog().get(name, {})


def list_available_datasets() -> list[str]:
    """Return the canonical dataset names discovered in the loaders."""

    return list(get_dataset_functions().keys())


def resolve_dataset_names(selectors: Any) -> list[str] | None:
    """Resolve *selectors* into canonical dataset names.

    The *selectors* argument accepts strings (dataset names or ``tag:<name>``),
    dictionaries with ``include``/``exclude``/``limit`` keys, metadata filters,
    or iterables of those forms. ``None`` returns ``None`` to signal "all
    datasets".
    """

    available = list_available_datasets()
    alias_map = _build_alias_map()
    if selectors is None:
        return None

    if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes)):
        combined: list[str] = []
        for entry in selectors:
            names = resolve_dataset_names(entry)
            if names is None:
                names = available.copy()
            combined.extend(names)
        return _dedupe(combined)

    if isinstance(selectors, Mapping):
        keys = set(selectors)
        if {"include", "exclude", "limit", "modality", "task", "label_type"} & keys:
            include_spec = selectors.get("include")
            if include_spec is None:
                included = available.copy()
            else:
                included = resolve_dataset_names(include_spec) or []
            exclude_spec = selectors.get("exclude")
            excluded = set(resolve_dataset_names(exclude_spec) or [])
            resolved = [name for name in included if name not in excluded]
            resolved = _filter_by_metadata(resolved, selectors)
            limit = selectors.get("limit")
            if isinstance(limit, int) and limit > 0:
                resolved = resolved[:limit]
            return _dedupe(resolved)
        if "name" in selectors:
            return resolve_dataset_names(selectors["name"]) or []
        if "tag" in selectors:
            return _expand_tag(selectors["tag"], available)
        return []

    if isinstance(selectors, str):
        if selectors.startswith("tag:"):
            return _expand_tag(selectors.split(":", 1)[1], available)
        normalized = selectors.replace("load_", "")
        canonical = _lookup_alias(normalized, alias_map)
        if canonical:
            return [canonical]
        if normalized not in available:
            logger.warning("Unknown dataset selector '%s'", selectors)
        return [normalized]

    return []


def _expand_tag(tag: str, available: Iterable[str]) -> list[str]:
    matches = []
    for name, meta in load_catalog().items():
        tags = meta.get("tags") or []
        if isinstance(tags, (list, tuple)) and tag in tags:
            matches.append(name)
    return [name for name in matches if name in available]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _filter_by_metadata(names: list[str], selectors: Mapping[str, Any]) -> list[str]:
    filtered = names
    for key in ("modality", "task", "label_type"):
        expected = selectors.get(key)
        if expected is None:
            continue
        expected_values = (
            {str(value) for value in expected}
            if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes))
            else {str(expected)}
        )
        filtered = [
            name
            for name in filtered
            if str(get_dataset_metadata(name).get(key)) in expected_values
        ]
    return filtered


@lru_cache(maxsize=1)
def _build_alias_map() -> dict[str, str]:
    """Return mapping of known dataset aliases to canonical loader keys."""

    alias_map: dict[str, str] = {}
    for canonical in list_available_datasets():
        alias_map[canonical] = canonical
        alias_map[canonical.lower()] = canonical

    catalog = load_catalog()
    for canonical, meta in catalog.items():
        display_name = meta.get("display_name")
        if isinstance(display_name, str) and display_name:
            alias_map.setdefault(display_name, canonical)
            alias_map.setdefault(display_name.lower(), canonical)
            alias_map.setdefault(display_name.replace(" ", "_").lower(), canonical)
        aliases = meta.get("aliases") or []
        if isinstance(aliases, Iterable):
            for alias in aliases:
                if not isinstance(alias, str) or not alias:
                    continue
                alias_map.setdefault(alias, canonical)
                alias_map.setdefault(alias.lower(), canonical)
    return alias_map


def _lookup_alias(name: str, alias_map: dict[str, str]) -> str | None:
    if not name:
        return None
    return alias_map.get(name) or alias_map.get(name.lower())
