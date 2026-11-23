"""Helpers to make sample weights a first-class citizen of resampling.

The resampling samplers historically only knew about ``X`` and ``y``.  This
module centralises the validation and the generation rules for
``sample_weight`` so that every sampler uses the same semantics.

The guiding principles are:

* weights are never modified in place and never silently replaced by uniform
  weights;
* validation happens *before* any sampling work;
* error messages only contain aggregate information (row counts, weight range
  and the sampler name), never individual samples;
* generated samples follow an explicit, documented rule
  (``weight_strategy``).

See :ref:`sample_weight` in the user guide for the full semantics.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import logging
from collections.abc import Mapping
from numbers import Real

import numpy as np

logger = logging.getLogger(__name__)

#: Rules accepted by over-samplers.  ``"inherit"`` copies the weight of the
#: sample the synthetic point is generated from, ``"nearest"`` linearly
#: interpolates between the base sample and its chosen nearest neighbour, and
#: ``"cluster"`` assigns the average weight of the pool used for generation.
WEIGHT_STRATEGIES = ("inherit", "nearest", "cluster")


def log_weight_summary(sampler_name, sample_weight, n_rows=None):
    """Log an aggregate summary of the weights.

    Only the weight range, the row count and the sampler name are emitted.
    Individual sample values are never logged.
    """
    sample_weight = np.asarray(sample_weight, dtype=float)
    if n_rows is None:
        n_rows = sample_weight.shape[0]
    logger.info(
        "%s: propagating sample weights over %d rows (%s)",
        sampler_name,
        n_rows,
        _weight_range(sample_weight),
    )


def _weight_range(sample_weight):
    """Return a compact description of the weight range.

    Only aggregate values are reported on purpose: logs and exceptions must
    never leak the content of individual samples.
    """
    sample_weight = np.asarray(sample_weight, dtype=float)
    if sample_weight.size == 0:
        return "empty"
    return (
        f"min={sample_weight.min():.6g}, max={sample_weight.max():.6g}, "
        f"mean={sample_weight.mean():.6g}"
    )


def _as_weight_array(sample_weight, *, n_samples, sampler_name):
    """Validate the raw ``sample_weight`` input.

    Parameters
    ----------
    sample_weight : array-like of shape (n_samples,)
        The weights provided by the user.

    n_samples : int
        Number of rows of ``X`` and ``y``.

    sampler_name : str
        Name of the sampler, used in error messages.

    Returns
    -------
    sample_weight : ndarray of shape (n_samples,), dtype=float
        A copy of the validated weights.
    """
    array = np.asarray(sample_weight)
    if array.ndim == 2 and 1 in array.shape:
        array = array.ravel()
    if array.ndim != 1:
        raise ValueError(
            f"{sampler_name}: 'sample_weight' must be one-dimensional. "
            f"Got an array with {array.ndim} dimensions."
        )

    if array.shape[0] != n_samples:
        raise ValueError(
            f"{sampler_name}: 'sample_weight' has {array.shape[0]} rows while "
            f"X and y have {n_samples} rows. Weights must be aligned row-wise "
            "with the data."
        )

    try:
        array = array.astype(float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{sampler_name}: 'sample_weight' could not be converted to "
            "floating point values."
        ) from exc

    if not np.all(np.isfinite(array)):
        n_invalid = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(
            f"{sampler_name}: 'sample_weight' contains {n_invalid} missing or "
            f"non-finite value(s) ({_weight_range(array)}). Weights must be "
            "finite."
        )

    if np.any(array < 0):
        n_negative = int(np.count_nonzero(array < 0))
        raise ValueError(
            f"{sampler_name}: 'sample_weight' contains {n_negative} negative "
            f"value(s) ({_weight_range(array)}). Weights must be non-negative."
        )

    if not np.any(array > 0):
        raise ValueError(
            f"{sampler_name}: all {array.shape[0]} sample weights are zero. "
            "Refusing to silently fall back to uniform weights."
        )

    return array


def _resolve_class_weight(class_weight, y, sampler_name):
    """Return a mapping ``class -> multiplicative factor``."""
    classes, counts = np.unique(y, return_counts=True)

    if class_weight is None:
        return None

    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError(
                f"{sampler_name}: 'class_weight' must be None, 'balanced' or a "
                f"mapping. Got {class_weight!r}."
            )
        n_samples = y.shape[0]
        n_classes = classes.shape[0]
        return {
            klass: n_samples / (n_classes * float(count))
            for klass, count in zip(classes, counts)
        }

    if isinstance(class_weight, Mapping):
        weights = {}
        for klass in classes:
            if klass not in class_weight:
                raise ValueError(
                    f"{sampler_name}: 'class_weight' is missing an entry for "
                    f"class {klass!r}."
                )
            value = class_weight[klass]
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError(
                    f"{sampler_name}: 'class_weight' values must be numbers. "
                    f"Got {type(value).__name__} for class {klass!r}."
                )
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"{sampler_name}: 'class_weight' values must be finite and "
                    f"non-negative. Got {value!r} for class {klass!r}."
                )
            weights[klass] = value
        return weights

    raise ValueError(
        f"{sampler_name}: 'class_weight' must be None, 'balanced' or a "
        f"mapping. Got {type(class_weight).__name__}."
    )


def reject_unsupported_weight_strategy(estimator):
    """Reject a ``weight_strategy`` set on a sampler that cannot honour it.

    Failing at composition time is deliberate: a sampling chain should never
    accept a rule that the sampler will only discover halfway through.
    """
    weight_strategy = getattr(estimator, "weight_strategy", None)
    if weight_strategy is None:
        return
    # ``weight_strategy`` must be a real constructor parameter.  Merely setting
    # the attribute on a sampler that cannot honour it must not be accepted
    # silently.
    params = (
        estimator.get_params(deep=False)
        if hasattr(estimator, "get_params")
        else {}
    )
    if "weight_strategy" not in params:
        raise ValueError(
            f"{estimator.__class__.__name__}: 'weight_strategy' is not supported "
            "by this sampler. Only over-samplers deriving generated rows accept "
            "a weight strategy."
        )


def check_effective_sample_weight(
    sample_weight, class_weight, y, *, sampler_name
):
    """Validate and combine row-level and class-level weights.

    The precedence is deterministic:

    ``effective_weight = sample_weight * class_weight[y]``

    When ``sample_weight`` is not provided, unit row weights are used before
    applying ``class_weight``.  The sampling strategy (the number of samples to
    draw or generate) is **not** affected by the weights; it only decides how
    many rows are produced.

    Parameters
    ----------
    sample_weight : array-like of shape (n_samples,) or None
        Row-level weights.

    class_weight : None, "balanced" or mapping
        Class-level multiplicative factors.

    y : ndarray of shape (n_samples,)
        The target array.

    sampler_name : str
        Name of the sampler, used in error messages.

    Returns
    -------
    sample_weight : ndarray of shape (n_samples,) or None
        The effective weights, or ``None`` when no weighting was requested.
    """
    if sample_weight is None and class_weight is None:
        return None

    n_samples = y.shape[0]
    if sample_weight is None:
        array = np.ones(n_samples, dtype=float)
    else:
        array = _as_weight_array(
            sample_weight, n_samples=n_samples, sampler_name=sampler_name
        )

    class_factors = _resolve_class_weight(class_weight, y, sampler_name)
    if class_factors is not None:
        factors = np.asarray([class_factors[klass] for klass in y], dtype=float)
        array = array * factors

        # ``class_weight`` could have zeroed out a whole class.  This must be
        # reported explicitly instead of silently producing uniform weights.
        if not np.any(array > 0):
            raise ValueError(
                f"{sampler_name}: combining 'sample_weight' and 'class_weight' "
                f"yields all-zero weights over {n_samples} rows. Refusing to "
                "silently fall back to uniform weights."
            )

    # A class whose samples all have a zero weight cannot contribute to the
    # resampling (zero-weight samples are excluded from neighbour searches and
    # synthetic sample generation).  Report it instead of degrading silently.
    # Only aggregate information is reported: no sample content, no class
    # labels, just row counts and the weight range.
    n_empty_classes = 0
    n_rows_empty_classes = 0
    for klass in np.unique(y):
        class_mask = y == klass
        if not np.any(array[class_mask] > 0):
            n_empty_classes += 1
            n_rows_empty_classes += int(np.count_nonzero(class_mask))
    if n_empty_classes:
        raise ValueError(
            f"{sampler_name}: {n_empty_classes} class(es) covering "
            f"{n_rows_empty_classes} of {n_samples} rows only have zero weights "
            f"({_weight_range(array)}). Zero-weight samples do not participate "
            "in resampling; refusing to silently fall back to uniform weights."
        )

    return array


def validate_weight_strategy(weight_strategy, *, supported, sampler_name):
    """Validate a ``weight_strategy`` parameter.

    Unsupported rules are rejected before any sampling work starts so that a
    sampling chain never fails half-way through.
    """
    if callable(weight_strategy):
        return weight_strategy

    if not isinstance(weight_strategy, str):
        raise ValueError(
            f"{sampler_name}: 'weight_strategy' must be one of {supported} or a "
            f"callable. Got {type(weight_strategy).__name__}."
        )

    if weight_strategy not in supported:
        raise ValueError(
            f"{sampler_name}: 'weight_strategy'={weight_strategy!r} is not "
            f"supported. Supported strategies are {tuple(supported)} or a "
            "callable."
        )
    return weight_strategy


def positive_weight_mask(sample_weight):
    """Return the mask of samples that may take part in resampling.

    Zero-weight samples are excluded: they must neither be used to synthesise
    new samples nor influence the neighbour search.
    """
    return np.asarray(sample_weight, dtype=float) > 0


def generated_weights(
    weight_strategy, base_weight, neighbor_weight, steps, pool_weight
):
    """Compute the weights of synthetic samples.

    Parameters
    ----------
    weight_strategy : {"inherit", "nearest", "cluster"} or callable
        The rule used to derive the weight of a generated sample.

    base_weight : ndarray of shape (n_generated,)
        Weight of the sample each synthetic point is generated from.

    neighbor_weight : ndarray of shape (n_generated,)
        Weight of the selected nearest neighbour.

    steps : ndarray of shape (n_generated,)
        Interpolation step in ``[0, 1]`` between the base and the neighbour.

    pool_weight : ndarray
        Weights of the whole generation pool, used by the ``"cluster"`` rule.

    Returns
    -------
    weight : ndarray of shape (n_generated,)
        The weight of each generated sample.
    """
    base_weight = np.asarray(base_weight, dtype=float)
    neighbor_weight = np.asarray(neighbor_weight, dtype=float)
    steps = np.asarray(steps, dtype=float).ravel()

    if callable(weight_strategy):
        weight = np.asarray(
            weight_strategy(base_weight, neighbor_weight, steps), dtype=float
        )
    elif weight_strategy == "inherit":
        weight = base_weight.copy()
    elif weight_strategy == "nearest":
        # ``steps`` may be outside [0, 1] for extrapolating variants (e.g.
        # SVM-SMOTE).  Clipping keeps the interpolated weight
        # non-negative and bounded by the two endpoints, so the rule stays
        # well defined for every sampler that supports it.
        clipped_steps = np.clip(steps, 0.0, 1.0)
        weight = (1.0 - clipped_steps) * base_weight + clipped_steps * neighbor_weight
    elif weight_strategy == "cluster":
        mean_weight = (
            float(np.mean(np.asarray(pool_weight, dtype=float)))
            if np.asarray(pool_weight).size
            else 0.0
        )
        weight = np.full(base_weight.shape[0], mean_weight, dtype=float)
    else:  # pragma: no cover - validated before sampling
        raise ValueError(f"Unsupported weight strategy: {weight_strategy!r}.")

    if weight.shape != base_weight.shape:
        raise ValueError(
            "The 'weight_strategy' callable returned "
            f"{weight.shape[0]} weights for {base_weight.shape[0]} generated "
            "samples."
        )
    if not np.all(np.isfinite(weight)):
        raise ValueError(
            "The 'weight_strategy' produced missing or non-finite weights for "
            "generated samples."
        )
    if np.any(weight < 0):
        raise ValueError(
            "The 'weight_strategy' produced negative weights for generated "
            "samples."
        )
    return weight


def supports_sample_weight(estimator):
    """Whether ``estimator`` declares support for ``sample_weight``."""
    return getattr(estimator, "_sample_weight_supported", False)

