# ChangePointLab interoperability: outliers and regime shifts

This optional bridge combines **point-level anomaly scores** from AnomalyBench
with **offline structural boundary events** from ChangePointLab. It keeps
the roles separate: an outlier is not a changepoint, and a persistent
anomaly run is not sufficient proof of a regime change.

Both projects may be installed together on Python 3.12. The bridge imports
only NumPy, so ChangePointLab is not a hard dependency of AnomalyBench.

```python
import numpy as np

from anomalybench.analytics.detectors.classical import IsolationForestDetector
from anomalybench.integrations.changepointlab import (
    partition_anomalies_by_changepoints,
)
from changepoint_lab import PELT
from changepoint_lab.algorithms.optimization.pelt import NormalMeanKnownVar

# Fit an anomaly detector and a changepoint detector to the same time order.
rng = np.random.default_rng(42)
values = np.r_[rng.normal(0, 1, 100), rng.normal(4, 1, 100)]
observations = values.reshape(-1, 1)

anomaly_detector = IsolationForestDetector().fit(
    observations, random_state=42
)
scores = np.asarray(anomaly_detector.score(observations), dtype=float)
changes = PELT(
    cost_fn=NormalMeanKnownVar(sigma2=1),
    penalty=8,
    min_seg_len=5,
).fit_predict(values)

# AnomalyBench Isolation Forest returns LOWER scores for stronger anomalies.
# The threshold here is illustrative, not statistically calibrated.
partition = partition_anomalies_by_changepoints(
    scores,
    changes,
    threshold=float(np.quantile(scores, 0.05)),
    score_orientation="lower_is_more_anomalous",
    boundary_radius=2,
    min_persistent_run=3,
)
print("Structural boundaries:", partition.changepoints)
print("Isolated anomaly indices:", np.flatnonzero(partition.isolated))
print("Boundary-adjacent indices:", np.flatnonzero(partition.boundary_adjacent))
print("Persistent anomaly indices:", np.flatnonzero(partition.persistent))
```

**Semantics:** ChangePointLab's right-exclusive boundary `k` lies between
`values[k-1]` and `values[k]`. With `boundary_radius=2`, the
transition neighborhood covers `k-2, k-1, k, k+1`. Zero disables that
classification. The three output masks are disjoint and partition all
threshold-exceeding scores. Runs are measured outside transition neighborhoods.

This is descriptive analysis only. It does not infer causality, correct
for changepoint detection uncertainty, or calibrate anomaly thresholds.
AnomalyBench's regular benchmarking pipeline and ChangePointLab's
Monte Carlo benchmark remain independent.
