"""File I/O boundaries for datasets, configuration, and benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dataexcept import DataLoadingError, FileWriteError


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV and retain the path and cause of read or parsing failures."""

    source = Path(path)
    try:
        return pd.read_csv(source, **kwargs)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        raise DataLoadingError(str(source), exc) from exc


def load_yaml(path: str | Path) -> Any:
    """Read a YAML document, leaving its schema validation to the caller."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DataLoadingError(str(source), exc) from exc


def ensure_directory(path: str | Path) -> Path:
    """Create an output directory or report the failed destination."""

    destination = Path(path)
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FileWriteError(str(destination), exc) from exc
    return destination


def write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text, preserving errors from the filesystem."""

    destination = Path(path)
    ensure_directory(destination.parent)
    try:
        destination.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise FileWriteError(str(destination), exc) from exc
