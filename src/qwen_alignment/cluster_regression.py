"""Small, deterministic OLS and one-way CR1 cluster-robust inference core.

The implementation is intentionally narrow: an intercept is always included,
observations retain their original weights, and clusters affect the covariance
matrix but never replace item-level rows.  Student-t inference uses ``G - 1``
degrees of freedom, where ``G`` is the number of distinct clusters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import mpmath
import numpy as np


class ClusterRegressionError(ValueError):
    """Raised when a regression input or numerical invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClusterRegressionError(message)


def student_t_survival(t_statistic: float, degrees_of_freedom: int) -> float:
    """Return ``P(T_df >= t)`` using the regularised incomplete-beta identity."""

    _require(degrees_of_freedom > 0, "Student-t degrees of freedom must be positive")
    _require(np.isfinite(t_statistic), "Student-t statistic must be finite")
    with mpmath.workdps(80):
        t_value = mpmath.mpf(float(t_statistic))
        df_value = mpmath.mpf(degrees_of_freedom)
        x_value = df_value / (df_value + t_value * t_value)
        half_beta = mpmath.betainc(
            df_value / 2,
            mpmath.mpf("0.5"),
            0,
            x_value,
            regularized=True,
        ) / 2
        survival = half_beta if t_value >= 0 else 1 - half_beta
        return float(survival)


def student_t_critical_two_sided_95(degrees_of_freedom: int) -> float:
    """Return the positive 97.5th percentile of Student's t distribution."""

    target_survival = 0.025
    lower = 0.0
    upper = 2.0
    while student_t_survival(upper, degrees_of_freedom) > target_survival:
        upper *= 2.0
    for _ in range(200):
        midpoint = (lower + upper) / 2.0
        if student_t_survival(midpoint, degrees_of_freedom) > target_survival:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _numeric_vector(values: Sequence[float], label: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ClusterRegressionError(f"{label} is not numeric") from exc
    _require(vector.ndim == 1, f"{label} must be one-dimensional")
    _require(vector.size > 0, f"{label} is empty")
    _require(bool(np.isfinite(vector).all()), f"{label} contains a non-finite value")
    return vector


def fit_ols_cr1(
    response: Sequence[float],
    predictors: Mapping[str, Sequence[float]],
    cluster_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit OLS and return a complete one-way CR1 sandwich analysis.

    ``predictors`` is insertion ordered.  The returned coefficient order is
    ``intercept`` followed by the supplied predictor names.  The CR1 covariance
    multiplier is ``G/(G-1) * (N-1)/(N-K)``; the second factor equals one in an
    intercept-only model, reproducing the project's locked RQ1 estimator.
    """

    y = _numeric_vector(response, "response")
    n_observations = int(y.size)
    _require(len(cluster_ids) == n_observations, "cluster vector length mismatch")
    _require(all(isinstance(value, str) and value for value in cluster_ids),
             "cluster IDs must be non-empty strings")

    predictor_names = list(predictors)
    _require(len(set(predictor_names)) == len(predictor_names),
             "predictor names are duplicated")
    _require(all(name and name != "intercept" for name in predictor_names),
             "predictor names must be non-empty and cannot be 'intercept'")

    columns = [np.ones(n_observations, dtype=np.float64)]
    for name in predictor_names:
        vector = _numeric_vector(predictors[name], f"predictor {name!r}")
        _require(vector.size == n_observations, f"predictor {name!r} length mismatch")
        columns.append(vector)
    design = np.column_stack(columns)
    n_parameters = int(design.shape[1])
    _require(n_observations > n_parameters,
             "OLS requires more observations than fitted parameters")
    rank = int(np.linalg.matrix_rank(design))
    _require(rank == n_parameters, "design matrix is not full rank")

    ordered_clusters: list[str] = []
    cluster_rows: dict[str, list[int]] = {}
    for row_index, cluster_id in enumerate(cluster_ids):
        if cluster_id not in cluster_rows:
            ordered_clusters.append(cluster_id)
            cluster_rows[cluster_id] = []
        cluster_rows[cluster_id].append(row_index)
    n_clusters = len(ordered_clusters)
    _require(n_clusters > 1, "CR1 requires at least two clusters")
    _require(n_clusters > n_parameters,
             "CR1 requires more clusters than fitted parameters")

    cross_product = design.T @ design
    bread = np.linalg.inv(cross_product)
    estimates = bread @ design.T @ y
    fitted = design @ estimates
    residuals = y - fitted

    meat = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    for cluster_id in ordered_clusters:
        indices = np.asarray(cluster_rows[cluster_id], dtype=np.int64)
        cluster_score = design[indices].T @ residuals[indices]
        meat += np.outer(cluster_score, cluster_score)

    correction = (
        (n_clusters / (n_clusters - 1))
        * ((n_observations - 1) / (n_observations - n_parameters))
    )
    covariance = correction * (bread @ meat @ bread)
    covariance = (covariance + covariance.T) / 2.0
    diagonal = np.diag(covariance)
    numerical_tolerance = np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(covariance)))
    ) * 100.0
    _require(bool(np.all(diagonal >= -numerical_tolerance)),
             "CR1 covariance has a materially negative diagonal")
    diagonal = np.maximum(diagonal, 0.0)
    standard_errors = np.sqrt(diagonal)
    _require(bool(np.all(standard_errors > 0.0)),
             "CR1 standard error is zero for at least one coefficient")

    degrees_of_freedom = n_clusters - 1
    critical_value = student_t_critical_two_sided_95(degrees_of_freedom)
    coefficient_names = ["intercept", *predictor_names]
    coefficient_results: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(coefficient_names):
        estimate = float(estimates[index])
        standard_error = float(standard_errors[index])
        t_statistic = estimate / standard_error
        p_value = min(
            1.0,
            2.0 * student_t_survival(abs(t_statistic), degrees_of_freedom),
        )
        margin = critical_value * standard_error
        coefficient_results[name] = {
            "estimate": estimate,
            "cr1_standard_error": standard_error,
            "confidence_interval_95": {
                "lower": estimate - margin,
                "upper": estimate + margin,
                "critical_value": critical_value,
                "reference_distribution": "Student t",
                "degrees_of_freedom": degrees_of_freedom,
                "sidedness": "two-sided",
            },
            "test": {
                "null_hypothesis": f"{name} = 0",
                "alternative_hypothesis": f"{name} != 0",
                "t_statistic": t_statistic,
                "p_value": p_value,
                "p_value_sidedness": "two-sided",
                "reference_distribution": "Student t",
                "degrees_of_freedom": degrees_of_freedom,
            },
        }

    residual_sum_squares = float(residuals @ residuals)
    centred = y - float(np.mean(y))
    total_sum_squares = float(centred @ centred)
    _require(total_sum_squares > 0.0, "R-squared is undefined for a constant response")
    r_squared = 1.0 - residual_sum_squares / total_sum_squares

    return {
        "estimation": "ordinary least squares",
        "covariance_estimator": "one-way CR1 cluster-robust sandwich",
        "cr1_finite_sample_correction": correction,
        "cluster_variable": "source_sid",
        "n_observations": n_observations,
        "n_clusters": n_clusters,
        "n_parameters_including_intercept": n_parameters,
        "degrees_of_freedom": degrees_of_freedom,
        "source_cluster_averaging": False,
        "coefficient_order": coefficient_names,
        "coefficients": coefficient_results,
        "r_squared": r_squared,
        "residual_sum_squares": residual_sum_squares,
        "total_sum_squares": total_sum_squares,
        "covariance_matrix": covariance.tolist(),
    }

