# Exceptions

Every error raised by AnomalyBench inherits from `dataexcept.DataExceptError`,
so one `except` clause catches all of them:

```python
from dataexcept import DataExceptError

try:
    detector.score(data)
except DataExceptError as exc:
    ...
```

!!! warning "These are not `ValueError` subclasses"
    The exceptions do **not** inherit from `ValueError`, `KeyError`, or
    `TypeError`. Code written against an earlier version that caught those
    builtins will no longer catch these — catch the DataExcept types instead.

## What is raised where

[DataExcept] supplies structured exceptions for most data and model failures,
and they are used directly wherever one fits:

| Exception | Raised when |
| --- | --- |
| `ConfigurationError` | a config key is malformed, or a detector key collides |
| `DataValidationError` | inputs fail a shape or content check |
| `DataFormatError` | a dataset file cannot be parsed as expected |
| `DataLoadingError` | a bundled CSV, YAML configuration, or catalog cannot be read or parsed |
| `FileWriteError` | a leaderboard, report, manifest, or output directory cannot be written |
| `DependencyError` | an optional extra is missing, or Python is unsupported |
| `HyperparameterError` | a search grid or parameter value is invalid |

[DataExcept]: https://pypi.org/project/dataexcept/

Three cases have no direct DataExcept equivalent and are defined here:

| Exception | Raised when |
| --- | --- |
| `DetectorNotFittedError` | `score()` is called before `fit()` |
| `UnknownDetectorError` | a detector key is absent from the registry |
| `UnknownDatasetError` | a selector names something the catalog does not hold |

Full signatures are in the [API reference](../reference/exceptions.md).

File errors include the affected location in `.source` or `.path` and retain
the underlying exception in `.original` and `__cause__`. The optional catalog
still resolves to an empty mapping when its file is absent. Missing benchmark
files continue to raise `ResourceNotFoundError`, while invalid configuration
values continue to raise `ConfigValidationError`.

`UnknownDetectorError` and `UnknownDatasetError` carry an `available` attribute
and list the valid keys in the message, so a typo tells you what you meant
instead of just that you were wrong.
