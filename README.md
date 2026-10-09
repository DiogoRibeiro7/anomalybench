<p align="center">
  <img src="assets/project-logo.png" alt="anomalybench project logo" width="160" height="160">
</p>

# AnomalyBench

[![CI](https://github.com/DiogoRibeiro7/anomalybench/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DiogoRibeiro7/anomalybench/actions/workflows/ci.yml) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21496904.svg)](https://doi.org/10.5281/zenodo.21496904) [![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://diogoribeiro7.github.io/anomalybench/)

AnomalyBench provides anomaly detection algorithms, benchmark dataset
loaders, and a command line interface for comparing detectors on standard
datasets under one reproducible evaluation protocol.

**Documentation: <https://diogoribeiro7.github.io/anomalybench/>**

## Python Support

This project supports Python `3.12` only. Python `3.13` is intentionally blocked
until the project runtime and dependency stack are validated there.

Install the project and development tooling with Poetry:

```bash
poetry install
poetry run pre-commit install
```

Poetry manages dependencies through `pyproject.toml`.

## Optional Extras

The base install supports classical detectors, ARIMA forecasting, graph
detectors, the CLI, and benchmark workflows. Optional detector stacks can be
enabled with Poetry extras:

```bash
# Deep learning detectors: PyTorch and TensorFlow
poetry install -E deep

# Streaming detectors: River
poetry install -E streaming

# Prophet detector support
poetry install -E forecasting

# Enable all optional detector stacks
poetry install -E all-detectors
```

## Usage

Run all benchmarks:

```bash
poetry run benchmark-cli
```

Select specific datasets by name:

```bash
poetry run benchmark-cli wisconsin_breast_cancer cardio
```

The suite includes tabular, image, time-series, and graph datasets:

```bash
poetry run benchmark-cli iris digits fashion_mnist_sample nab_art_daily_small_noise nab_machine_temperature synthetic_timeseries karate_club_graph
```

Run selected detectors:

```bash
poetry run benchmark-cli --detectors knn hbos
```

Run from a YAML configuration:

```bash
poetry run benchmark-cli --config anomalybench/benchmarks/benchmark_config.yml
```

Run a reproducible smoke benchmark with a manifest, JSON report, and enriched
leaderboard:

```bash
poetry run benchmark-cli --config anomalybench/benchmarks/benchmark_config.v0.3.0-smoke.yml
```

Run a multi-metric benchmark using dataset metadata selectors:

```bash
poetry run benchmark-cli --config anomalybench/benchmarks/benchmark_config.v0.4.0-metrics.yml
```

Run a modern tabular detector smoke benchmark:

```bash
poetry run benchmark-cli --config anomalybench/benchmarks/benchmark_config.v0.5.0-modern-tabular.yml
```

Show dataset summaries:

```bash
poetry run benchmark-cli --summary
```

Legacy display names such as `wisconsinBreast` remain supported, but canonical
loader keys such as `wisconsin_breast_cancer` should be preferred for scripts
and configuration files.

Available detectors include Isolation Forest, Stochastic Outlier Selection,
K-Nearest Neighbors, Histogram-Based Outlier Score, One-Class SVM, DBSCAN,
Elliptic Envelope, Gaussian Mixture, Sklearn LOF, KMeans, PCA Reconstruction,
Mahalanobis distance, Kernel Density, Autoencoder, Denoising Autoencoder,
Variational Autoencoder, LSTM Autoencoder, Transformer, COPOD, Feature Bagging,
LODA, ABOD, Half-Space Trees, Online Isolation Forest, AnoGAN, MAD-GAN, Degree
Centrality, Graph Isolation Forest, ECOD, Random Feature Isolation Forest,
Random Network Distillation, ARIMA, and Prophet.

## Exceptions

Errors raised by this package come from
[DataExcept](https://pypi.org/project/DataExcept/), which provides structured
exceptions carrying the offending field, value, or dependency rather than a
bare message. Every one inherits from `dataexcept.DataExceptError`:

```python
from dataexcept import DataExceptError, HyperparameterError

from anomalybench.analytics.time_series import WindowSpec

try:
    WindowSpec(window_length=1)
except HyperparameterError as exc:
    print(exc.param, exc.value)  # window_length 1
except DataExceptError:  # catches anything this package raises
    raise
```

These exceptions **do not inherit from** `ValueError`, `KeyError`, `TypeError`,
`RuntimeError`, or `ImportError`. Code written against earlier versions that
caught those must be updated; see the mapping table in
[CHANGELOG.md](CHANGELOG.md).

Cases DataExcept has no direct equivalent for are defined in
[anomalybench/analytics/exceptions.py](anomalybench/analytics/exceptions.py) and still inherit from it:
`DetectorNotFittedError`, `UnknownDetectorError`, and `UnknownDatasetError`.

## Plugins

External detector packages can register themselves through plugin modules whose
names start with `plugins.`:

```bash
poetry run benchmark-cli --plugins plugins.my_module --detectors my_custom_detector
```

The module should call `anomalybench.analytics.detectors.register_detector` during
import.
Detector keys are protected against accidental collisions by default. To
replace an existing detector intentionally, pass `allow_override=True` to
`register_detector`.

## Detector API

All built-in detectors follow the same lifecycle:

- `fit(data, **params)` trains the detector and marks it as fitted.
- `score(data)` returns detector-specific anomaly scores and raises
  `DetectorNotFittedError` if called before `fit`.
- `detect_anomalies(data, **params)` is the fit-and-score convenience path used
  by benchmark workflows.

Detectors expose an `is_fitted` property and a `score_orientation` value. Score
orientation is one of `higher_is_more_anomalous`, `lower_is_more_anomalous`,
`binary_anomaly`, or `estimator_defined`. Current score values are preserved for
backwards compatibility; use `score_orientation` when comparing detectors that
produce different score semantics.

## Modern Tabular Detectors

The `v0.5.0` detector pack adds CPU-friendly modern tabular methods:

- `ecod` wraps PyOD's empirical-CDF detector as an adapter.
- `random_feature_isolation_forest` applies Isolation Forest to random
  nonlinear feature representations.
- `random_network_distillation` trains a compact predictor against a fixed
  random representation and scores prediction error.

These methods are useful when classical distance, density, or covariance
baselines are too rigid for nonlinear feature interactions. Classical methods
remain preferable for small datasets, tight latency budgets, easy
interpretability, or when their score semantics are already validated for a
workflow. Benchmark reports include each detector's score orientation and
runtime so modern and classical methods can be compared explicitly.

## Hyperparameter Search

Stratified cross-validation utilities are available in
`anomalybench.analytics.hyperparam`:

```python
from anomalybench.analytics.hyperparam import grid_search

best_params, score = grid_search(
    "isolation_forest",
    {"n_estimators": [50, 100]},
    X,
    y,
    cv=5,
)
```

## Leaderboards

Benchmark results can be appended to a CSV leaderboard:

```bash
poetry run benchmark-cli iris --detectors isolation_forest --leaderboard results.csv
```

The leaderboard CSV uses a structured schema with these columns:
`run_timestamp_utc`, `run_id`, `config_hash`, `dataset_name`, `dataset_key`,
`detector_name`, `detector_label`, `detector_params`, `random_seed`,
`runtime_seconds`, `failure_category`, `auc`, and `error`.

For reproducible runs, provide a stable run identifier, seed, and output path:

```bash
poetry run benchmark-cli iris \
  --detectors isolation_forest \
  --metrics roc_auc average_precision precision_at_k runtime \
  --metric-k 10 \
  --positive-label 1 \
  --random-seed 42 \
  --run-id paper-table-1 \
  --output-dir benchmark-results \
  --json-report benchmark-results/paper-table-1.json
```

The JSON report embeds a benchmark manifest with the package version, Python
version, selected dataset keys, detector keys and parameters, random seed,
metric configuration, configuration hash, timestamp, and bundled dataset file
integrity hashes.
Bundled dataset metadata in `anomalybench/benchmarks/datasets.yml` records source URLs,
license notes, modality, task type, label semantics, and local files used for
integrity checks.

Supported benchmark metrics are `roc_auc`, `average_precision`,
`precision_at_k`, `recall_at_k`, `f1_at_threshold`, `best_f1`, and `runtime`.
YAML configurations can select datasets by metadata fields such as `modality`,
`task`, and `label_type`.

## Quality Checks

Run the same core checks used by CI before opening a pull request:

```bash
poetry check
poetry build -f wheel
poetry run python -m pytest -q
poetry run pre-commit run --all-files
```

## Citation

Machine-readable citation metadata is maintained in
[CITATION.cff](CITATION.cff) and [.zenodo.json](.zenodo.json). Zenodo reads
`.zenodo.json` when it archives a GitHub release and mints the DOI for that
version, so both files are kept in sync with the version in
[pyproject.toml](pyproject.toml). `tests/test_citation_metadata.py` and the
release workflow fail if they drift apart.

Zenodo issues two kinds of DOI:

- The **concept DOI**
  [10.5281/zenodo.21496904](https://doi.org/10.5281/zenodo.21496904) always
  resolves to the latest archived version. Cite it when referring to the
  software in general.
- A **version DOI** is minted for each archived release. Cite it when the
  exact version matters for reproducing reported results, and pair it with the
  run manifest emitted by the benchmark CLI.

BibTeX for the concept DOI:

```bibtex
@software{ribeiro_anomalydetection,
  author    = {Ribeiro, Diogo},
  title     = {{AnomalyBench: a reproducible benchmarking suite for
               anomaly detection algorithms}},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21496904},
  url       = {https://doi.org/10.5281/zenodo.21496904}
}
```

Replace the DOI with the version DOI shown on the Zenodo record to cite a
specific release, and add the matching `version` and `year` fields.

## Project Roadmap

- Implement variable width binning for HBOS. Completed.
- Add caching to LOF calculations. Completed.
- Provide a CLI for running included benchmarks. Completed.
- Add dataset summary output to the CLI. Completed.
- Publish the package to PyPI. Completed.
- Publish a documentation site. Completed:
  <https://diogoribeiro7.github.io/anomalybench/>.
- Expand the detector library with additional algorithms and deep learning
  approaches. See [ROADMAP.md](ROADMAP.md).

## Contributing

Contributions to expand the detector library are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
