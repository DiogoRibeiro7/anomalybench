"""Path and cause contracts for benchmark input and output failures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from dataexcept import DataLoadingError, FileWriteError, SchemaMismatchError

from anomalybench._io import load_csv
from anomalybench.benchmarks import catalog, config_benchmark, reproducibility
from anomalybench.benchmarks.corrected_loaders import load_cardio_corrected
from anomalybench.benchmarks.load_datasets import load_lympho
from anomalybench.cli import _append_leaderboard_rows


def test_missing_dataset_retains_its_source_and_cause(tmp_path: Path) -> None:
    source = tmp_path / "absent.csv"

    with pytest.raises(DataLoadingError) as error:
        load_csv(source)

    assert error.value.source == str(source)
    assert isinstance(error.value.original, FileNotFoundError)
    assert error.value.__cause__ is error.value.original


@pytest.mark.parametrize("loader", [load_lympho, load_cardio_corrected])
def test_bundled_dataset_loaders_preserve_read_failures(
    monkeypatch: pytest.MonkeyPatch, loader: Callable[[], object]
) -> None:
    def denied(*args: object, **kwargs: object) -> pd.DataFrame:
        raise PermissionError("dataset access denied")

    monkeypatch.setattr(pd, "read_csv", denied)
    with pytest.raises(DataLoadingError) as error:
        loader()
    assert error.value.source is not None
    assert error.value.source.endswith(("lymphography.dat", "ctg.csv"))
    assert isinstance(error.value.original, PermissionError)


def test_malformed_config_retains_its_source_and_cause(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.yaml"
    source.write_text("datasets: [unclosed", encoding="utf-8")

    with pytest.raises(DataLoadingError) as error:
        config_benchmark.run_from_config(source)

    assert error.value.source == str(source)
    assert error.value.__cause__ is error.value.original


def test_unreadable_catalog_surfaces_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "datasets.yml"
    source.write_text("datasets: [unclosed", encoding="utf-8")
    monkeypatch.setattr(catalog, "_CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        with pytest.raises(DataLoadingError) as error:
            catalog.load_catalog()
        assert error.value.source == str(source)
    finally:
        catalog.load_catalog.cache_clear()


def test_catalog_rejects_a_non_mapping_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "datasets.yml"
    source.write_text("- not a catalog\n", encoding="utf-8")
    monkeypatch.setattr(catalog, "_CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        with pytest.raises(SchemaMismatchError, match="catalog mapping"):
            catalog.load_catalog()
    finally:
        catalog.load_catalog.cache_clear()


@pytest.mark.parametrize("writer", ["json", "leaderboard"])
def test_artifact_writers_retain_destination_and_cause(
    tmp_path: Path, writer: str
) -> None:
    destination = tmp_path / "occupied"
    destination.mkdir()

    with pytest.raises(FileWriteError) as error:
        if writer == "json":
            reproducibility.write_json(destination, {"valid": True})
        else:
            _append_leaderboard_rows(destination, [])

    assert error.value.path == str(destination)
    assert isinstance(error.value.original, OSError)
    assert error.value.__cause__ is error.value.original
