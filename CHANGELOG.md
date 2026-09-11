# Changelog

All notable changes to this project will be documented in this file.

The format follows Keep a Changelog, and this project uses semantic versioning.

## [0.6.2](https://github.com/DiogoRibeiro7/anomalybench/compare/v0.6.1...v0.6.2) (2026-09-11)


### Fixed

* **packaging:** publish the project URLs to PyPI ([7c08191](https://github.com/DiogoRibeiro7/anomalybench/commit/7c081919d408a7e14a669cc238d7a917db91305b))

## [0.6.1](https://github.com/DiogoRibeiro7/anomalybench/compare/v0.6.0...v0.6.1) (2026-09-04)


### Internal

* only ever publish a release tag ([5bf42cb](https://github.com/DiogoRibeiro7/anomalybench/commit/5bf42cb53f50d63b1f50961f6992abc040e921c7))
* publish to PyPI with trusted publishing ([e94f17d](https://github.com/DiogoRibeiro7/anomalybench/commit/e94f17d2843a80cccdaccbddc6632db84afd4301))

## [0.6.0](https://github.com/DiogoRibeiro7/anomalybench/compare/v0.5.1...v0.6.0) (2026-09-04)


### ⚠ BREAKING CHANGES

* imports move from analytics.* and benchmarks.* to anomalybench.analytics.* and anomalybench.benchmarks.*, and the distribution is renamed from anomaly-detection to anomalybench.

### Fixed

* address review findings and follow the repository rename ([7e490f7](https://github.com/DiogoRibeiro7/anomalybench/commit/7e490f78988171244092ef4b4ad9aeb8567898bf))


### Changed

* nest the packages under an anomalybench namespace ([5c30d4d](https://github.com/DiogoRibeiro7/anomalybench/commit/5c30d4d48a226e75dc30dca6e3b0e30c29aa72b4))

## [0.5.1](https://github.com/DiogoRibeiro7/AnomalyDetection/compare/v0.5.0...v0.5.1) (2026-09-04)


### Fixed

* stop pinning the version in the reproducibility test ([6b04e9c](https://github.com/DiogoRibeiro7/AnomalyDetection/commit/6b04e9c7910b5ef001084efb1fae1455caefc0e1))


### Internal

* automate releases with release-please ([204246a](https://github.com/DiogoRibeiro7/AnomalyDetection/commit/204246a5acd33c0dc67dc8164ee170b9c97cc6ff))
* collapse to a single long-lived branch ([7a36ed6](https://github.com/DiogoRibeiro7/AnomalyDetection/commit/7a36ed657ad94d53fffac4ceda9de6a010ee1356))
* drop the develop trigger entries left by the rename ([b44795b](https://github.com/DiogoRibeiro7/AnomalyDetection/commit/b44795bff8c790edc5c8f962493a2a8825e18c88))
* give the wheel upload a retry path ([e798945](https://github.com/DiogoRibeiro7/AnomalyDetection/commit/e798945140baee9daad35f3e1c478172adf6f18b))
* queue release-please runs and keep attaching the wheel ([9d3f61d](https://github.com/DiogoRibeiro7/AnomalyDetection/commit/9d3f61db540b78ba6c39dc0f35e9b66d62185107))

## v0.5.0 - 2026-09-03

### Breaking

- Errors raised by this package now come from the
  [DataExcept](https://pypi.org/project/DataExcept/) hierarchy and **no longer
  inherit from `ValueError`, `KeyError`, `TypeError`, `RuntimeError`, or
  `ImportError`**. Every error inherits from `dataexcept.DataExceptError`, so a
  caller can still catch all of them with one type. Code that caught the
  builtins must be updated:

  | Previously | Now |
  | --- | --- |
  | `ValueError` on detector or field validation | `DataValidationError` |
  | `ValueError` on a parameter value | `HyperparameterError` |
  | `ValueError` on configuration | `ConfigurationError` |
  | `ValueError` on input shape | `DataValidationError` |
  | `TypeError` on an unsupported container type | `DataFormatError` |
  | `KeyError` on a missing dataframe column | `MissingColumnError` |
  | `KeyError` or `ValueError` on an unknown detector | `UnknownDetectorError` |
  | `KeyError` on an unknown dataset | `UnknownDatasetError` |
  | `RuntimeError` on scoring before fit | `DetectorNotFittedError` |
  | `ImportError` on an optional dependency | `DependencyError` |

- `benchmarks.config_benchmark.ConfigValidationError` now inherits from
  `dataexcept.ConfigurationError` instead of `ValueError`. Its message is
  unchanged and the offending key is available as `option`.
- Benchmark reports and leaderboards record an undefined metric as `null`
  rather than `NaN`. A single-class dataset previously wrote a bare `NaN`,
  which strict JSON parsers reject.

### Added

- Modern tabular detectors: ECOD, Random Feature Isolation Forest, and Random
  Network Distillation, with seed support in manifests and CLI runs and a
  versioned `v0.5.0` smoke benchmark configuration.
- A CPU-friendly TCN encoder-decoder time-series reconstruction baseline.
- Explicit rolling-window semantics for time-series detectors: `WindowSpec`,
  windowed score alignment, and point-label alignment, so sequence detectors no
  longer collapse the time axis.
- An inductive DBSCAN detector that can score new data without refitting.
- Engineering confidence gates in CI covering Ruff, Mypy, and a coverage floor.
- Structured citation metadata: an expanded Zenodo record, a synchronized
  `CITATION.cff`, a DOI badge, a BibTeX entry, and tests plus a release check
  that fail when the version drifts between `pyproject.toml`, `.zenodo.json`,
  and `CITATION.cff`.

### Fixed

- Two benchmark entries for the same detector no longer overwrite each other.
  Results were keyed by label, so a second entry for one detector reported the
  first entry's AUC and runtime.
- `analytics.hyperparam.grid_search` no longer ranks lower-is-more-anomalous
  detectors upside down. It scored raw `score()` output, so Isolation Forest
  produced an inverted ROC AUC and the search selected the worst parameters.
- Benchmark anomaly labels are canonicalized to `1 = anomaly` at the loader
  boundary, and dataset metadata must declare its source anomaly label.
- The reported `score_orientation` now matches the orientation evaluation
  actually applied, instead of disagreeing for detectors that return a plain
  score array.
- Transformer reconstruction is order-aware, and temporal detectors preserve
  sequence length and input dimensionality.
- Point-stream window scores align to benchmark labels rather than being
  compared against unaligned label vectors.
- `analytics.detector` re-exports every detector submodule; ECOD,
  Random Feature Isolation Forest, Random Network Distillation, and
  `InductiveDBSCANDetector` were unreachable through it.
- The Wisconsin loader works on pandas 3.
- Detector registrations are validated, and duplicate keys are rejected unless
  an override is requested explicitly.

### Changed

- `analytics.detector` writes its re-exports explicitly instead of installing
  them into `globals()`, so type checkers and IDEs can resolve them.

### Internal

- The repository is Ruff-clean and Mypy-clean, and both gates check the whole
  tree rather than a hand-maintained file list that silently went stale.
- Test coverage rose from 60% to 74%, with a CI floor of 71%. The largest gains
  were `analytics/lof.py` (14% to 96%), `analytics/detectors/graph.py` (43% to
  100%), `analytics/detectors/classical.py` (68% to 94%),
  `benchmarks/config_benchmark.py` (72% to 97%), and `analytics/time_series.py`
  (71% to 98%).
- Ruff, Mypy, pytest-cov, and types-PyYAML are pinned in CI so linter results
  do not drift between a developer's machine and the pipeline.

## v0.4.0 - 2026-07-23

- Added configurable benchmark metrics, including ROC AUC, average precision,
  precision at k, recall at k, F1 at threshold, best F1, and runtime.
- Added metric configuration to benchmark manifests, JSON reports, and
  leaderboard rows.
- Added metadata-based dataset selectors for modality, task, and label type.
- Expanded dataset metadata with modality, label type, positive label, and
  label semantics fields.
- Added a versioned `v0.4.0` metrics smoke benchmark configuration.

## v0.3.0 - 2026-07-22

- Added reproducible benchmark manifests and versioned JSON reports.
- Added CLI and YAML config options for output directories, JSON reports,
  stable run IDs, and random seeds.
- Expanded leaderboard rows with run IDs, configuration hashes, runtime, random
  seed, and failure category fields.
- Added metadata-driven integrity checks for bundled benchmark dataset files.
- Added a versioned smoke benchmark configuration for `v0.3.0`.

## v0.2.0 - 2026-07-22

- Added detector lifecycle enforcement through `BaseDetector.is_fitted`.
- Added clear `score`-before-`fit` errors across built-in detectors.
- Added shared tabular input coercion for detector implementations.
- Added detector score-orientation metadata.
- Added detector API documentation and lifecycle tests.

## v0.1.0 - 2026-07-22

- Modernized project metadata and packaging configuration.
- Added repository hygiene files for licensing, editor defaults, and releases.
- Added citation metadata for GitHub and Zenodo archiving.
