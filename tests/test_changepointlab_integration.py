"""Behavioral tests for the optional ChangePointLab bridge."""

from dataclasses import dataclass

import numpy as np
import pytest

from anomalybench.integrations.changepointlab import (
    partition_anomalies_by_changepoints,
)


@dataclass
class CPResult:
    indices: np.ndarray
    boundary_convention: str = "right_exclusive"


def test_disjoint_partition_and_boundary_semantics() -> None:
    scores = np.array([0, 5, 0, 0, 5, 5, 0, 5, 5, 5, 0], dtype=float)
    result = partition_anomalies_by_changepoints(
        scores, CPResult(np.array([5])), threshold=4,
        score_orientation="higher_is_more_anomalous", boundary_radius=1,
    )
    np.testing.assert_array_equal(np.flatnonzero(result.boundary_adjacent), [4, 5])
    np.testing.assert_array_equal(np.flatnonzero(result.isolated), [1])
    np.testing.assert_array_equal(np.flatnonzero(result.persistent), [7, 8, 9])
    np.testing.assert_array_equal(
        result.anomalous,
        result.boundary_adjacent | result.isolated | result.persistent,
    )
    assert not np.any(result.boundary_adjacent & result.persistent)


def test_lower_scores_are_more_anomalous_and_inputs_are_copied() -> None:
    scores = np.array([0.0, -5.0, 0.0, -6.0])
    boundaries = np.array([2])
    result = partition_anomalies_by_changepoints(
        scores, CPResult(boundaries), threshold=-4,
        score_orientation="lower_is_more_anomalous", boundary_radius=0,
    )
    np.testing.assert_array_equal(np.flatnonzero(result.isolated), [1, 3])
    result.changepoints[0] = 3
    np.testing.assert_array_equal(boundaries, [2])


def test_persistent_run_with_no_changepoints() -> None:
    result = partition_anomalies_by_changepoints(
        [6.0, 6.0, 6.0, 0.0], CPResult(np.array([], dtype=int)),
        threshold=5, score_orientation="higher_is_more_anomalous",
    )
    assert result.persistent.sum() == 3
    assert not result.boundary_adjacent.any()


@pytest.mark.parametrize("indices", [[0], [4], [2, 2], [3, 2], [1.5]])
def test_invalid_boundaries(indices: list[float]) -> None:
    with pytest.raises(ValueError):
        partition_anomalies_by_changepoints(
            [1.0, 2.0, 3.0, 4.0], CPResult(np.asarray(indices)),
            threshold=2, score_orientation="higher_is_more_anomalous",
        )


def test_online_convention_rejected() -> None:
    with pytest.raises(ValueError, match="right_exclusive"):
        partition_anomalies_by_changepoints(
            [1, 2, 3], CPResult(np.array([1]), "time_index"),
            threshold=2, score_orientation="higher_is_more_anomalous",
        )


@pytest.mark.parametrize("kwargs", [
    {"threshold": float("nan")},
    {"score_orientation": "descending"},
    {"boundary_radius": -1},
    {"boundary_radius": True},
    {"min_persistent_run": 1},
    {"min_persistent_run": 2.5},
])
def test_invalid_policy(kwargs: dict[str, object]) -> None:
    policy: dict[str, object] = dict(
        threshold=2, score_orientation="higher_is_more_anomalous"
    )
    policy.update(kwargs)
    with pytest.raises(ValueError):
        partition_anomalies_by_changepoints(
            [1, 2, 3], CPResult(np.array([1])), **policy
        )
