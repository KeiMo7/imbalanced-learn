"""Safeguards against resampling data leakage and synthetic-sample provenance.

This module is opt-in: nothing in here changes the behaviour of the existing
samplers unless the user explicitly instantiates :class:`ResamplingGuard` /
:class:`GuardedSampler` or enables provenance tracking.

Design principles
-----------------
* Arrays are never copied or hashed: protected inputs are registered by
  Python object identity, so the checks stay cheap for million-row datasets.
* Provenance records only contain sample identifiers (integers), never
  feature values, so sensitive fields cannot leak through the audit trail.
* Samplers that cannot provide parent information are reported as
  ``"not_supported"`` instead of fabricating a lineage.
"""

# Authors: The imbalanced-learn developers
# License: MIT

import contextvars
import itertools
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import _num_samples

from imblearn.base import is_sampler

__all__ = [
    "ResamplingLeakageError",
    "InfeasibleSamplingTargetError",
    "ResamplingProvenance",
    "SamplingRatioReport",
    "ResamplingGuard",
    "GuardedSampler",
    "check_sampling_ratio",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ResamplingLeakageError(RuntimeError):
    """Raised when a resampler is applied to data registered as validation or
    test data while a :class:`ResamplingGuard` is active.

    The error message names the guarded role (``"validation"`` /
    ``"test"``), the offending sampler and the step at which the call was
    intercepted.
    """


class InfeasibleSamplingTargetError(ValueError):
    """Raised when the requested resampling target cannot be reached with the
    data at hand.

    The message distinguishes the two failure modes:

    * ``"insufficient_samples"``: not enough real samples of a class
      (e.g. under-sampling below the available count, SMOTE without a
      neighbour);
    * ``"parameter_conflict"``: contradictory parameters (e.g. an
      over-sampling target smaller than the current class count).
    """

    def __init__(self, message, reason=None):
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
# Context variable carrying the active provenance sink. It is ``None`` unless
# provenance tracking is explicitly enabled, in which case every code path is
# a cheap ``is None`` check: existing behaviour and performance are unchanged.
_PROVENANCE = contextvars.ContextVar("imblearn_provenance", default=None)

_counter = itertools.count()


def _new_record_id():
    """Return a process-unique identifier for a provenance record."""
    return f"prov-{uuid.uuid4().hex[:12]}-{next(_counter)}"


@dataclass
class ResamplingProvenance:
    """Lineage of the samples returned by one ``fit_resample`` call.

    The record deliberately stores **sample identifiers only** -- integer
    positions in the input array -- and never feature values, so it is safe
    to persist alongside data that contains sensitive fields.

    Parameters
    ----------
    sampler : str
        Class name of the sampler that produced the samples.

    n_input_samples : int
        Number of samples passed to ``fit_resample``.

    n_output_samples : int
        Number of samples returned by ``fit_resample``.

    synthetic_sample_ids : ndarray of shape (n_synthetic,), dtype=int
        Positions, in the resampled output, of the samples that did not exist
        in the input.

    parent_sample_ids : dict or None
        Mapping ``output position -> tuple of input positions`` from which
        each synthetic sample was generated. ``None`` when
        :attr:`supported` is ``False``; a missing key for a synthetic sample
        means that specific sampler could not name its parents.

    support : {"full", "partial", "not_supported"}
        - ``"full"``: every synthetic sample has parent information;
        - ``"partial"``: the sampler exposes selection indices
          (``sample_indices_``) but no generation-level lineage;
        - ``"not_supported"``: no lineage information is available.

    record_id : str
        Unique identifier of this record, usable as a cross-reference in
        audit logs.
    """

    sampler: str
    n_input_samples: int
    n_output_samples: int
    synthetic_sample_ids: np.ndarray
    parent_sample_ids: dict | None
    support: str
    record_id: str = field(default_factory=_new_record_id)

    @property
    def supported(self):
        """Return ``True`` when at least partial lineage is available."""
        return self.support != "not_supported"

    def parents_of(self, output_position):
        """Return the input positions that generated ``output_position``.

        Returns ``None`` when the sample is original or its lineage is not
        available. Feature values are never returned.
        """
        if self.parent_sample_ids is None:
            return None
        return self.parent_sample_ids.get(int(output_position))

    def to_dict(self):
        """Return a plain-dict, JSON-serialisable view of the record.

        Integer keys of :attr:`parent_sample_ids` are converted to strings.
        """
        parents = None
        if self.parent_sample_ids is not None:
            parents = {
                str(key): list(value) for key, value in self.parent_sample_ids.items()
            }
        return {
            "record_id": self.record_id,
            "sampler": self.sampler,
            "support": self.support,
            "n_input_samples": self.n_input_samples,
            "n_output_samples": self.n_output_samples,
            "synthetic_sample_ids": np.asarray(self.synthetic_sample_ids).tolist(),
            "parent_sample_ids": parents,
        }


def _current_provenance():
    """Return the active provenance sink, or ``None`` when tracking is off."""
    return _PROVENANCE.get()


@contextmanager
def provenance_context(sink):
    """Install ``sink`` as the provenance sink for the current context.

    Nested and concurrent (``contextvars``) calls get independent sinks, so
    cross-validation folds never share synthetic samples.
    """
    token = _PROVENANCE.set(sink)
    try:
        yield sink
    finally:
        _PROVENANCE.reset(token)


# ---------------------------------------------------------------------------
# Ratio feasibility and accounting
# ---------------------------------------------------------------------------
@dataclass
class SamplingRatioReport:
    """Result of :func:`check_sampling_ratio`.

    Parameters
    ----------
    feasible : bool
        Whether the target can be reached on the given data.

    reason : str or None
        When not feasible, either ``"insufficient_samples"`` or
        ``"parameter_conflict"``.

    target_counts : dict
        Desired final number of samples per class.

    current_counts : dict
        Number of samples per class in the input data.

    actual_counts : dict or None
        Observed counts after resampling. ``None`` until
        :meth:`record_actual` is called.

    deviations : dict
        Per-class signed difference between actual and target counts
        (``actual - target``). Empty until actual counts are recorded.
    """

    feasible: bool
    reason: str | None
    target_counts: dict
    current_counts: dict
    actual_counts: dict | None = None
    deviations: dict = field(default_factory=dict)

    def record_actual(self, y_resampled):
        """Fill :attr:`actual_counts` and :attr:`deviations` from ``y``."""
        classes, counts = np.unique(y_resampled, return_counts=True)
        self.actual_counts = dict(zip(classes.tolist(), counts.tolist()))
        self.deviations = {
            klass: int(self.actual_counts.get(klass, 0) - target)
            for klass, target in self.target_counts.items()
        }
        return self

    def to_dict(self):
        return {
            "feasible": self.feasible,
            "reason": self.reason,
            "target_counts": dict(self.target_counts),
            "current_counts": dict(self.current_counts),
            "actual_counts": self.actual_counts,
            "deviations": dict(self.deviations),
        }


def _counts(y):
    classes, counts = np.unique(y, return_counts=True)
    return dict(zip(classes.tolist(), counts.tolist()))


def check_sampling_ratio(
    sampling_strategy, y, sampling_type, *, raise_on_infeasible=True
):
    """Pre-check whether a resampling target is reachable.

    This is a dry run: it resolves ``sampling_strategy`` exactly as the
    samplers do and reports conflicts *before* any array is allocated, so an
    interrupted check never leaves a half-built output behind.

    Parameters
    ----------
    sampling_strategy : float, str, dict or callable
        Same meaning as the ``sampling_strategy`` parameter of the samplers.

    y : array-like of shape (n_samples,)
        Target vector of the data that would be resampled.

    sampling_type : {"over-sampling", "under-sampling", "clean-sampling"}
        Kind of sampler the strategy is intended for.

    raise_on_infeasible : bool, default=True
        Raise :class:`InfeasibleSamplingTargetError` when the target is
        unreachable; otherwise return a report with ``feasible=False``.

    Returns
    -------
    report : SamplingRatioReport
        The feasibility report, including the resolved per-class targets.
    """
    from imblearn.utils._validation import (
        check_sampling_strategy as _resolve_strategy,
    )

    y = np.asarray(y)
    current_counts = _counts(y)
    try:
        # The library resolver raises plain ValueErrors on conflicts.
        resolved = _resolve_strategy(sampling_strategy, y, sampling_type)
    except ValueError as exc:
        message = str(exc)
        # Distinguish "not enough rows exist" from contradictory targets.
        if (
            "should be less or equal" in message
            or "required to generate new" in message
            or "is/are not present" in message
        ):
            reason = "insufficient_samples"
        else:
            reason = "parameter_conflict"
        report = SamplingRatioReport(
            feasible=False,
            reason=reason,
            target_counts={},
            current_counts=current_counts,
        )
        if raise_on_infeasible:
            raise InfeasibleSamplingTargetError(message, reason=reason)
        return report

    # ``resolved`` holds the *number of samples to add/remove* per class.
    target_counts = {}
    reason = None
    for klass, delta in resolved.items():
        available = current_counts[klass]
        if sampling_type == "over-sampling":
            target_counts[klass] = available + int(delta)
        else:
            target_counts[klass] = int(delta)
            if int(delta) > available:
                reason = "insufficient_samples"
    feasible = reason is None
    report = SamplingRatioReport(
        feasible=feasible,
        reason=reason,
        target_counts=target_counts,
        current_counts=current_counts,
    )
    if not feasible and raise_on_infeasible:
        klass = next(
            klass
            for klass, target in target_counts.items()
            if target > current_counts[klass]
        )
        raise InfeasibleSamplingTargetError(
            f"Cannot reach the target of {target_counts[klass]} samples for class "
            f"{klass!r}: only {current_counts[klass]} samples are available and "
            f"{sampling_type} cannot create the missing ones.",
            reason=reason,
        )
    return report


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
class ResamplingGuard:
    """Track which arrays are training / validation / test data.

    The guard registers arrays **by object identity** -- no hashing, no
    copy -- and intercepts resampling calls made through
    :class:`GuardedSampler`. A resampling call on an array registered as
    ``"validation"`` or ``"test"`` raises :class:`ResamplingLeakageError`.

    Arrays that were never registered are treated as training data, matching
    the historical behaviour when the guard is not used.

    Examples
    --------
    >>> from imblearn.safety import ResamplingGuard
    >>> guard = ResamplingGuard()
    >>> guard.register_test(X_test)  # doctest: +SKIP
    """

    def __init__(self):
        # id() -> (role, repr-like label). Keeping the label makes the error
        # message actionable without retaining the array itself.
        self._registry = {}
        # provenance records collected during the guard lifetime
        self.provenance_records_ = []

    def register(self, X, role, *, name=None):
        """Mark ``X`` as belonging to ``role``.

        Parameters
        ----------
        X : array-like
            The array to protect. Registration uses ``id(X)``; the array is
            not copied and its contents are not read.

        role : {"train", "validation", "test"}
            Role of the data. Only ``"train"`` may be resampled.

        name : str, default=None
            Optional human-readable label included in error messages.
        """
        if role not in ("train", "validation", "test"):
            raise ValueError(
                f"role must be one of 'train', 'validation', 'test'. Got {role!r}."
            )
        label = name if name is not None else f"{role} array (id={id(X)})"
        self._registry[id(X)] = (role, label)
        return self

    def register_train(self, X, *, name=None):
        """Register ``X`` as training data."""
        return self.register(X, "train", name=name)

    def register_validation(self, X, *, name=None):
        """Register ``X`` as validation data; resampling it raises."""
        return self.register(X, "validation", name=name)

    def register_test(self, X, *, name=None):
        """Register ``X`` as test data; resampling it raises."""
        return self.register(X, "test", name=name)

    def role_of(self, X):
        """Return the registered role of ``X`` or ``"train"`` if unknown."""
        entry = self._registry.get(id(X))
        return entry[0] if entry is not None else "train"

    def check_resample_allowed(self, X, sampler, step="fit_resample"):
        """Raise :class:`ResamplingLeakageError` if ``X`` is protected."""
        entry = self._registry.get(id(X))
        if entry is not None and entry[0] in ("validation", "test"):
            role, label = entry
            raise ResamplingLeakageError(
                f"Refusing to resample {label}: it is registered as {role!r} data "
                f"with this ResamplingGuard, but resampling is only legal on "
                f"training data. The call was intercepted at step {step!r} of "
                f"sampler {type(sampler).__name__}. Split the data before "
                f"resampling and apply the sampler to the training partition only "
                f"(e.g. inside an imblearn.Pipeline so each CV fold resamples its "
                f"own training data)."
            )

    @contextmanager
    def fold(self):
        """Scope provenance collection to one cross-validation fold.

        Synthetic samples produced in one fold can never be carried into the
        next one: each fold gets a fresh provenance sink, and records
        collected inside the context are attached to the guard when it exits
        without error.
        """
        sink = _ProvenanceSink()
        with provenance_context(sink):
            yield sink
        # Publish only on clean exit: an interrupted sampling leaves no
        # partial record and no output array is returned to the caller.
        self.provenance_records_.extend(sink.records)


# ---------------------------------------------------------------------------
# Internal provenance sink and helpers used by the sampler hooks
# ---------------------------------------------------------------------------
class _ProvenanceSink:
    """Accumulates provenance produced within one guard/fold context."""

    def __init__(self):
        self.records = []
        # number of rows in the array handed to fit_resample; synthetic
        # output positions start after the original rows
        self.n_input_samples = None
        # global input position of each row of the base subset and of the
        # neighbour subset currently used for generation; set by the SMOTE
        # samplers before calling _make_samples
        self.class_index_map = None
        self.neighbor_index_map = None

    def add_record(self, record):
        self.records.append(record)


# ---------------------------------------------------------------------------
# GuardedSampler
# ---------------------------------------------------------------------------
class GuardedSampler(BaseEstimator):
    """Wrap a sampler to enforce a :class:`ResamplingGuard` and collect
    provenance.

    The wrapper delegates every parameter and every computation to the
    wrapped sampler, so numerical results are identical to using the sampler
    directly. It adds two opt-in behaviours:

    1. before ``fit_resample``, the guard verifies that the input is not
       registered validation/test data;
    2. after the call, a :class:`ResamplingProvenance` record is stored in
       :attr:`provenance_` (``support="not_supported"`` when the sampler
       cannot provide lineage -- it is never fabricated).

    Parameters
    ----------
    sampler : sampler object
        Any imbalanced-learn sampler (an object with ``fit_resample``).

    guard : ResamplingGuard
        The guard holding the array-role registry.

    Attributes
    ----------
    provenance_ : ResamplingProvenance or None
        Lineage of the last ``fit_resample`` call. ``None`` until fitted.

    Examples
    --------
    >>> from imblearn.over_sampling import SMOTE
    >>> from imblearn.safety import GuardedSampler, ResamplingGuard
    >>> guard = ResamplingGuard()
    >>> safe_smote = GuardedSampler(SMOTE(random_state=42), guard=guard)
    >>> X_res, y_res = safe_smote.fit_resample(X, y)  # doctest: +SKIP
    """

    _estimator_type = "sampler"

    def __init__(self, sampler, guard):
        self.sampler = sampler
        self.guard = guard

    # -- scikit-learn plumbing ------------------------------------------------
    def __sklearn_tags__(self):
        return self.sampler.__sklearn_tags__()

    def _more_tags(self):
        return getattr(self.sampler, "_more_tags", lambda: {})()

    def get_params(self, deep=True):
        return {"sampler": self.sampler, "guard": self.guard}

    def set_params(self, **params):
        if "sampler" in params:
            self.sampler = params.pop("sampler")
        if "guard" in params:
            self.guard = params.pop("guard")
        if params:
            self.sampler.set_params(**params)
        return self

    def __sklearn_clone__(self):
        # Pipelines and cross-validation clone every step before fitting. The
        # clone must keep sharing the same guard, otherwise fold records
        # would be collected on objects the user can never reach.
        return GuardedSampler(clone(self.sampler), self.guard)

    # -- provenance construction ---------------------------------------------
    def _build_provenance(self, X, y, X_resampled, y_resampled, inner):
        n_in = _num_samples(X)
        n_out = _num_samples(X_resampled)
        sampler_name = type(inner).__name__
        sink = _current_provenance()

        synthetic_ids = np.arange(n_in, n_out)
        parent_ids = None
        support = "not_supported"

        # 1. Generation-level lineage emitted by the SMOTE-family hook.
        hook_parents = getattr(inner, "_provenance_parents_", None)
        if hook_parents:
            parent_ids = dict(hook_parents)
            support = "full"
        elif hasattr(inner, "sample_indices_"):
            # 2. Samplers that only expose which input rows were selected.
            indices = np.asarray(inner.sample_indices_)
            support = "partial"
            parent_ids = {}
            for out_pos, src_pos in enumerate(indices):
                if out_pos >= n_in:
                    # duplicated/selected rows beyond the original block
                    parent_ids[int(out_pos)] = (int(src_pos),)
            synthetic_ids = np.setdiff1d(np.arange(n_out), np.arange(n_in))

        record = ResamplingProvenance(
            sampler=sampler_name,
            n_input_samples=n_in,
            n_output_samples=n_out,
            synthetic_sample_ids=synthetic_ids,
            parent_sample_ids=parent_ids,
            support=support,
        )
        self.provenance_ = record
        if sink is not None:
            sink.add_record(record)
        if hasattr(self.guard, "provenance_records_") and sink is None:
            # also keep an instance-level trail when used outside a fold()
            self.guard.provenance_records_.append(record)
        return record

    # -- sampler API ----------------------------------------------------------
    def fit_resample(self, X, y, **params):
        """Check the guard, delegate resampling, record provenance.

        The wrapped arrays are never modified: the inner sampler contract of
        returning new arrays is preserved.
        """
        if not (hasattr(self.sampler, "fit_resample") and is_sampler(self.sampler)):
            raise TypeError(
                f"GuardedSampler expects an imbalanced-learn sampler. Got "
                f"{type(self.sampler).__name__!r}."
            )
        self.guard.check_resample_allowed(X, self.sampler)
        inner = clone(self.sampler)

        active_sink = _current_provenance()
        if active_sink is not None:
            # Running inside a guard.fold() (or nested) context: reuse it so
            # the fold owns the collected record.
            sink = active_sink
            context = nullcontext()
        else:
            # Standalone use: provide a private sink so lineage is still
            # collected on self.provenance_ without requiring fold().
            sink = _ProvenanceSink()
            context = provenance_context(sink)

        with context:
            # Reset on every call so a previous fold can never leak into
            # this one.
            sink.class_index_map = None
            sink.neighbor_index_map = None
            sink.n_input_samples = _num_samples(X)
            # Fresh per-call lineage containers: clones inside
            # cross-validation never inherit another fold's synthetic samples.
            if hasattr(inner, "_provenance_parents_"):
                inner._provenance_parents_ = {}
            if hasattr(inner, "_provenance_offset_"):
                inner._provenance_offset_ = _num_samples(X)

            try:
                X_res, y_res = inner.fit_resample(X, y, **params)
            except BaseException:
                # An interrupted sampling must not leave a half-built record.
                self.provenance_ = None
                raise

        self.sampler_ = inner
        self._build_provenance(X, y, X_res, y_res, inner)
        return X_res, y_res

    def fit(self, X, y, **params):
        """Fit the wrapped sampler after checking the guard."""
        self.guard.check_resample_allowed(X, self.sampler, step="fit")
        self.sampler_ = clone(self.sampler).fit(X, y, **params)
        return self
