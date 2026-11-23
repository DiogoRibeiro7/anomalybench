"""Dependency-light integration of point anomaly scores and changepoint events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from dataexcept import DataValidationError
from numpy.typing import NDArray

ScoreOrientation = Literal["higher_is_more_anomalous", "lower_is_more_anomalous"]


class RightExclusiveChangepointResult(Protocol):
    """Minimal contract for an offline ChangePointLab result."""

    @property
    def indices(self) -> NDArray[np.integer]:
        """Return interior right-exclusive boundaries."""
        ...

    @property
    def boundary_convention(self) -> str:
        """Return the boundary convention."""
        ...


@dataclass(frozen=True)
class AnomalyChangePointPartition:
    """Disjoint anomaly categories and independent structural boundaries."""

    changepoints: NDArray[np.int_]
    anomalous: NDArray[np.bool_]
    boundary_adjacent: NDArray[np.bool_]
    isolated: NDArray[np.bool_]
    persistent: NDArray[np.bool_]


def partition_anomalies_by_changepoints(
    scores: NDArray[np.floating] | list[float],
    changepoints: RightExclusiveChangepointResult,
    *,
    threshold: float,
    score_orientation: ScoreOrientation,
    boundary_radius: int = 1,
    min_persistent_run: int = 3,
) -> AnomalyChangePointPartition:
    """Separate anomalous scores by proximity and duration, not causality.

    Args:
        scores: One finite anomaly score per observation, in time order.
        changepoints: Offline result using right-exclusive boundaries.
        threshold: Inclusive anomaly cutoff in original score units.
        score_orientation: Direction in which scores become more anomalous.
        boundary_radius: Number of observations on either side of a boundary.
            Zero disables boundary-adjacent categorization.
        min_persistent_run: Minimum run length for persistent alerts outside
            changepoint neighborhoods.

    Returns:
        Three disjoint masks partitioning the thresholded anomalous positions:
        boundary-adjacent, isolated, and persistent. Persistent observations
        are not automatically structural changes.
    """
    values = np.asarray(scores)
    if values.ndim != 1 or values.size < 2:
        raise DataValidationError(
            field="changepointlab_bridge",
            value="invalid",
            message="scores must be a 1-D array with at least two observations",
        )
    if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise DataValidationError(
            field="changepointlab_bridge",
            value="invalid",
            message="scores must contain finite numerical values",
        )
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(threshold)
    ):
        raise DataValidationError(
            field="changepointlab_bridge",
            value="invalid",
            message="threshold must be finite",
        )
    if score_orientation not in (
        "higher_is_more_anomalous",
        "lower_is_more_anomalous",
    ):
        raise DataValidationError(
            field="changepointlab_bridge",
            value="invalid",
            message="invalid score_orientation",
        )
    for name, value, minimum in (
        ("boundary_radius", boundary_radius, 0),
        ("min_persistent_run", min_persistent_run, 2),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise DataValidationError(
                field="changepointlab_bridge",
                value="invalid",
                message=f"{name} must be an integer >= {minimum}",
            )
    if changepoints.boundary_convention != "right_exclusive":
        raise DataValidationError(
            field="changepointlab_bridge",
            value="invalid",
            message="changepoints must use right_exclusive boundaries",
        )
    cps = np.asarray(changepoints.indices)
    if cps.ndim != 1 or cps.dtype.kind not in "iu":
        raise DataValidationError(
            field="changepointlab_bridge",
            value="invalid",
            message="changepoints must be a 1-D integer array",
        )
    if (
        np.any(cps <= 0)
        or np.any(cps >= values.size)
        or np.any(np.diff(cps) <= 0)
    ):
        raise DataValidationError(
            field="changepointlab_bridge",
            value="invalid",
            message="changepoints must be sorted, unique interior boundaries",
        )

    anomalous = (
        values >= threshold
        if score_orientation == "higher_is_more_anomalous"
        else values <= threshold
    )
    neighborhood = np.zeros(values.size, dtype=bool)
    for cp in cps:
        # Boundary cp lies between observations cp-1 and cp.
        left = max(0, int(cp) - boundary_radius)
        right = min(int(values.size), int(cp) + boundary_radius)
        neighborhood[left:right] = True
    adjacent = anomalous & neighborhood
    residual = anomalous & ~neighborhood
    persistent = np.zeros(values.size, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, residual, False].astype(np.int8)))
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        if end - start >= min_persistent_run:
            persistent[start:end] = True
    return AnomalyChangePointPartition(
        changepoints=cps.astype(int, copy=True),
        anomalous=anomalous.astype(bool, copy=True),
        boundary_adjacent=adjacent,
        isolated=residual & ~persistent,
        persistent=persistent,
    )
