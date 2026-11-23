# Authors: imbalanced-learn developers
# License: MIT
"""Multi-label over-sampling driven by label-set combinations."""

import numbers
import warnings
from collections import OrderedDict

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.utils import _safe_indexing, check_random_state
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.validation import _num_samples
from sklearn_compat.utils.validation import validate_data

from imblearn.base import BaseSampler
from imblearn.over_sampling._multilabel import (
    EMPTY_LABEL_SET,
    check_multilabel_target,
    group_by_label_set,
    indicator_block,
)
from imblearn.utils._validation import (
    ArraysTransformer,
    _check_X,
    check_neighbors_object,
)

_SINGLETON_OPTIONS = ("keep", "merge", "skip")


class MultiLabelSMOTE(BaseSampler):
    """Over-sample a multi-label dataset by label-set combination.

    Standard single-label samplers only understand one class per sample. When
    the target is a multi-label indicator matrix -- a sample may carry several
    labels at once -- they either reject the input outright or collapse it onto
    a single class, which makes the rare label combinations impossible to
    resample.

    :class:`MultiLabelSMOTE` treats the *full set of active labels* of each
    sample as its resampling unit. Two samples belong to the same group only if
    they carry exactly the same label combination. SMOTE interpolation is then
    run independently inside every group, so every synthetic sample inherits a
    label combination that genuinely exists in the training data. Synthetic
    samples can never carry an unseen combination, an all-zero label set, or an
    out-of-range label.

    Read more in the :ref:`User Guide <multilabel_smote>`.

    Parameters
    ----------
    sampling_strategy : str, dict or callable, default="auto"
        Budget per label-set combination.

        - ``"auto"``: every non-empty combination is brought up to the size of
          the most frequent non-empty combination;
        - ``"all"``: same as ``"auto"`` but the all-zero combination is also
          eligible when ``zero_label_rows="resample"``;
        - ``"minority"``: only the smallest non-empty combination is resampled,
          up to the size of the second smallest one;
        - ``dict``: keys are tuples of label indices (a label set) and values
          are the *final* number of samples requested for that combination,
          e.g. ``{(0, 2): 25, (1,): 10}``. The empty tuple ``()`` targets the
          all-zero combination. Values cannot be smaller than the current
          counts, and every requested combination must exist in the training
          target;
        - callable: a function taking the target matrix ``y`` and returning
          such a dictionary.

    label_budget : dict or None, default=None
        Maximum number of *synthetic* samples allowed to carry each label
        after resampling, keyed by label index, e.g. ``{0: 50, 2: 10}``. Labels
        absent from the dict are uncapped. This protects a label from being
        flooded while other combinations are filled. By default a cap that
        makes a requested target unreachable raises a :class:`ValueError`
        reporting the exact per-label shortfall instead of silently
        under-sampling; see ``enforce_label_budget``.

    enforce_label_budget : {"raise", "warn"}, default="raise"
        What to do when ``label_budget`` cannot satisfy the requested counts:

        - ``"raise"``: raise a :class:`ValueError` listing each per-label
          shortfall;
        - ``"warn"``: emit a :class:`UserWarning`, record the shortfall in
          ``label_budget_shortfall_`` and resample with the reduced budget.

    total_budget : int or None, default=None
        Maximum total number of *synthetic* samples generated across all
        combinations. ``None`` means no global cap. When the cap is binding the
        requested counts are scaled down proportionally (largest remainder
        rounding), so every combination still receives an integer budget.

    k_neighbors : int or object, default=5
        Number of nearest neighbours used within each label-set group, either
        as an integer or as an object exposing ``kneighbors`` and
        ``kneighbors_graph``. Groups smaller than ``k_neighbors + 1`` fall back
        to all available in-group neighbours.

    singleton_strategy : {"keep", "merge", "skip"}, default="keep"
        Behaviour for a label-set combination appearing exactly once that the
        budget asks to grow:

        - ``"keep"``: the lone original sample is kept but no synthetic sample
          is generated for the combination;
        - ``"merge"``: the sample is interpolated with the nearest neighbours
          of the closest non-singleton, non-empty combination. Synthetic rows
          carry that neighbour combination's label set, never a new one;
        - ``"skip"``: the singleton sample is removed from the resampled
          output entirely.

    zero_label_rows : {"keep", "resample"}, default="keep"
        Behaviour for rows whose label set is empty:

        - ``"keep"``: all-zero rows are forwarded untouched and are never used
          to synthesise data;
        - ``"resample"``: the all-zero combination is resampled like any other
          group (it still needs at least two members for SMOTE).

    zero_label_rows_threshold : int, float or None, default=None
        Guard against targets dominated by unlabelled rows. ``None`` accepts
        any number of all-zero rows. An integer sets an absolute maximum count,
        a float in ``[0, 1]`` a maximum fraction; exceeding it raises a
        :class:`ValueError` reporting the count and offending rows.

    output_sparse : bool or None, default=None
        Sparsity of the returned target matrix ``y_resampled``.

        - ``None``: preserve the input sparsity -- a sparse target yields a
          CSR matrix and a dense target yields a numpy array;
        - ``True``: always return a CSR matrix, even for dense input;
        - ``False``: always return a dense numpy array. Set this when chaining
          with a classifier that requires a dense multi-label target (most
          scikit-learn classifiers do), avoiding a manual ``.toarray()``.

        The feature matrix ``X_resampled`` always preserves the input
        sparsity independently of this flag.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of neighbour selection and interpolation.
        Passing the same integer makes repeated runs bit-for-bit identical.

    Attributes
    ----------
    sampling_strategy_ : OrderedDict[tuple, int]
        Final target size (originals + synthetic) per resampled label set.

    label_set_counts_ : OrderedDict[tuple, int]
        Number of training samples per label set observed at fit time.

    label_set_indices_ : OrderedDict[tuple, ndarray]
        Training row indices belonging to each label set.

    n_synthetic_ : OrderedDict[tuple, int]
        Number of synthetic samples actually generated per label set. It may be
        smaller than the request for singleton groups under the ``"keep"``
        strategy.

    label_budget_shortfall_ : dict[int, int]
        Per-label number of synthetic samples that could not be generated
        because of ``label_budget``. Empty when no cap was binding.

    sample_indices_ : ndarray of shape (n_samples_resampled,)
        Indices of the original samples kept (negative values denote synthetic
        samples).

    nn_k_ : estimator object
        Validated nearest-neighbours estimator created from ``k_neighbors``.

    n_labels_in_ : int
        Number of label columns observed during ``fit``.

    n_features_in_ : int
        Number of features observed during ``fit``.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``. Defined only when ``X`` has
        feature names that are all strings.

    See Also
    --------
    SMOTE : Single-label synthetic over-sampling.

    Notes
    -----
    Sparse targets stay sparse: label sets are read directly from the CSR
    ``indices``/``indptr`` arrays and no ``n_labels``-squared structure is ever
    allocated, so memory grows linearly with the number of stored ones.

    Synthetic samples are generated from training rows only. The sampler never
    looks at ``X`` or ``y`` beyond the data passed to ``fit_resample``, so a
    validation set cannot leak into the generated rows.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from imblearn.over_sampling import MultiLabelSMOTE
    >>> X = np.arange(56, dtype=float).reshape(14, 4)
    >>> y = np.array(
    ...     [
    ...         [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
    ...         [1, 0, 0], [1, 0, 0],
    ...         [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
    ...         [0, 1, 0],
    ...         [1, 0, 1], [1, 0, 1], [1, 0, 1],
    ...     ]
    ... )
    >>> sampler = MultiLabelSMOTE(random_state=42)
    >>> X_res, y_res = sampler.fit_resample(X, sparse.csr_matrix(y))
    >>> sparse.issparse(y_res)
    True
    >>> sampler.n_synthetic_[(0, 2)]
    3
    """

    _sampling_type = "over-sampling"
    _required_parameters = []

    # Signals to the common estimator checks that this sampler consumes a 2D
    # multi-label indicator target instead of a 1D single-label target; the
    # single-label-specific checks are therefore not applicable.
    _multilabel_sampler = True

    _parameter_constraints: dict = {
        "sampling_strategy": [
            StrOptions({"auto", "all", "minority"}),
            dict,
            callable,
        ],
        "label_budget": [dict, None],
        "enforce_label_budget": [StrOptions({"raise", "warn"})],
        "total_budget": [Interval(numbers.Integral, 0, None, closed="left"), None],
        "k_neighbors": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
        ],
        "singleton_strategy": [StrOptions(set(_SINGLETON_OPTIONS))],
        "zero_label_rows": [StrOptions({"keep", "resample"})],
        "zero_label_rows_threshold": [
            Interval(numbers.Integral, 0, None, closed="left"),
            Interval(numbers.Real, 0.0, 1.0, closed="both"),
            None,
        ],
        "output_sparse": ["boolean", None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        label_budget=None,
        enforce_label_budget="raise",
        total_budget=None,
        k_neighbors=5,
        singleton_strategy="keep",
        zero_label_rows="keep",
        zero_label_rows_threshold=None,
        output_sparse=None,
        random_state=None,
    ):
        # ``BaseSampler`` stores ``sampling_strategy`` for API compatibility;
        # the single-label validation of the base class is bypassed below.
        super().__init__(sampling_strategy=sampling_strategy)
        self.label_budget = label_budget
        self.enforce_label_budget = enforce_label_budget
        self.total_budget = total_budget
        self.k_neighbors = k_neighbors
        self.singleton_strategy = singleton_strategy
        self.zero_label_rows = zero_label_rows
        self.zero_label_rows_threshold = zero_label_rows_threshold
        self.output_sparse = output_sparse
        self.random_state = random_state

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def _check_X_y(self, X, y):
        """Validate a multi-label indicator target and the feature matrix."""
        if y is None:
            raise ValueError(
                "MultiLabelSMOTE requires y to be passed, but the target y is "
                "None. Expected a binary indicator matrix of shape "
                "(n_samples, n_labels)."
            )
        # Validate X first: it surfaces feature-side errors (e.g. complex
        # data) with the message callers and the common checks expect.
        X_checked = _check_X(X)
        y_checked = check_multilabel_target(
            y, zero_label_rows_threshold=self.zero_label_rows_threshold
        )
        if _num_samples(X_checked) != y_checked.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples. "
                f"Got X with {_num_samples(X_checked)} rows and y with "
                f"{y_checked.shape[0]} rows."
            )
        # Record n_features_in_/feature_names_in_ without forcing sklearn's
        # 1D target validation: the multi-label target is validated by
        # ``check_multilabel_target`` and stays 2D (and possibly sparse).
        validate_data(
            self,
            X=X_checked,
            y="no_validation",
            reset=True,
            skip_check_array=True,
            accept_sparse=["csr", "csc"],
        )
        self.n_labels_in_ = y_checked.shape[1]
        return X_checked, y_checked

    def fit(self, X, y):
        """Check inputs and compute the label-set statistics.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : {array-like, sparse matrix} of shape (n_samples, n_labels)
            Binary multi-label indicator matrix.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        self._validate_params()
        X, y = self._check_X_y(X, y)
        self._build_strategy(y)
        return self

    def fit_resample(self, X, y):
        """Resample the multi-label dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data to sample.

        y : {array-like, sparse matrix} of shape (n_samples, n_labels)
            Binary multi-label indicator matrix aligned row-by-row with ``X``.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix, dataframe} of shape \
                (n_samples_new, n_features)
            The resampled feature matrix. Original rows keep their order and
            synthetic rows are appended, so ``X`` and ``y`` stay aligned.

        y_resampled : {ndarray, sparse matrix} of shape \
                (n_samples_new, n_labels)
            The indicator matrix of the resampled data. Dense input yields a
            dense matrix, sparse input yields a CSR matrix.
        """
        self._validate_params()
        arrays_transformer = ArraysTransformer(X, y)
        X, y = self._check_X_y(X, y)
        X_out, y_out = self._fit_resample(X, y)
        # ``output_sparse`` is an explicit request about the representation of
        # the target; do not let the caller's container type override it.
        if self.output_sparse is None:
            return arrays_transformer.transform(X_out, y_out)
        X_out, _ = arrays_transformer.transform(X_out, y_out)
        if self.output_sparse:
            y_out = y_out if sparse.issparse(y_out) else sparse.csr_matrix(y_out)
        elif sparse.issparse(y_out):
            y_out = y_out.toarray()
        return X_out, y_out

    # ------------------------------------------------------------------
    # strategy resolution and budgets
    # ------------------------------------------------------------------
    def _build_strategy(self, y):
        groups, counts = group_by_label_set(y)
        self.label_set_indices_ = groups
        self.label_set_counts_ = counts

        non_empty = OrderedDict(
            (label_set, count)
            for label_set, count in counts.items()
            if label_set != EMPTY_LABEL_SET
        )
        if not non_empty:
            raise ValueError(
                "Every row of the multi-label target is all-zero: at least "
                "one labelled sample is required to fit MultiLabelSMOTE."
            )

        strategy = self._resolve_request(y, counts, non_empty)
        strategy = self._apply_zero_policy(
            strategy,
            explicit_zero_request=isinstance(self.sampling_strategy, dict),
        )
        self.singleton_plan_ = self._plan_singletons(strategy)
        strategy = self._apply_singleton_policy(strategy)
        strategy = self._apply_budgets(strategy)
        self.sampling_strategy_ = strategy
        return strategy

    def _resolve_request(self, y, counts, non_empty):
        """Turn ``sampling_strategy`` into final sizes per label set."""
        request = self.sampling_strategy
        if callable(request):
            request = request(y)

        if isinstance(request, str):
            if request == "minority":
                sorted_sets = sorted(non_empty, key=non_empty.get)
                target = sorted_sets[0]
                goal = non_empty[sorted_sets[1]] if len(sorted_sets) > 1 else 1
                if goal <= non_empty[target]:
                    return OrderedDict()
                return OrderedDict([(target, int(goal))])
            # "auto" and "all" equalise to the largest non-empty combination.
            # "all" additionally considers the all-zero combination when the
            # zero-row policy allows resampling it; "auto" never touches
            # unlabelled rows.
            goal = max(non_empty.values())
            eligible = dict(counts) if request == "all" else dict(non_empty)
            return OrderedDict(
                (label_set, max(goal, count)) for label_set, count in eligible.items()
            )

        if isinstance(request, dict):
            return self._validate_dict_strategy(request, counts)

        raise ValueError(
            "`sampling_strategy` must be one of {'auto', 'all', 'minority'}, "
            f"a dict or a callable. Got {request!r}."
        )

    def _validate_dict_strategy(self, request, counts):
        normalised = OrderedDict()
        unknown = [label_set for label_set in request if tuple(label_set) not in counts]
        if unknown:
            raise ValueError(
                f"Label set(s) {unknown} requested in `sampling_strategy` are "
                "not present in the training target. Synthetic samples may "
                "only carry a combination observed in training. Observed "
                f"combinations: {list(counts)}."
            )
        for label_set, target_size in request.items():
            label_set = tuple(label_set)
            if not isinstance(target_size, numbers.Integral) or target_size < 0:
                raise ValueError(
                    "Values of `sampling_strategy` must be non-negative "
                    f"integers. Got {target_size!r} for label set {label_set}."
                )
            if target_size < counts[label_set]:
                raise ValueError(
                    "With over-sampling, the requested size must be greater "
                    "than or equal to the current count. Label set "
                    f"{label_set} currently has {counts[label_set]} samples "
                    f"and {target_size} were requested."
                )
            normalised[label_set] = int(target_size)
        return normalised

    def _apply_zero_policy(self, strategy, *, explicit_zero_request=False):
        if self.zero_label_rows == "keep":
            if explicit_zero_request and EMPTY_LABEL_SET in strategy:
                raise ValueError(
                    "`sampling_strategy` targets the all-zero label "
                    "combination but `zero_label_rows='keep'` forbids "
                    "resampling unlabelled rows. Set "
                    "`zero_label_rows='resample'` to grow the all-zero "
                    "combination, or drop it from `sampling_strategy`."
                )
            strategy.pop(EMPTY_LABEL_SET, None)
        elif EMPTY_LABEL_SET in strategy:
            zero_count = self.label_set_counts_.get(EMPTY_LABEL_SET, 0)
            if zero_count < 2:
                warnings.warn(
                    "The all-zero label combination has "
                    f"{zero_count} sample(s) and cannot be interpolated; it "
                    "is kept as-is instead of being resampled.",
                    UserWarning,
                    stacklevel=3,
                )
                strategy.pop(EMPTY_LABEL_SET, None)
        return strategy

    def _apply_singleton_policy(self, strategy):
        """Freeze combinations seen once when they cannot be interpolated.

        With ``singleton_strategy='keep'`` the lone sample is kept but the
        requested growth is cancelled, with an explicit warning. With
        ``'skip'`` the growth is cancelled too and the lone row is dropped when
        the resampled data are assembled. ``'merge'`` leaves the request in
        place because it interpolates with a donor combination instead.
        """
        plan = self.singleton_plan_
        for label_set, action in plan.items():
            if action == "merge":
                continue
            requested = strategy[label_set]
            current = self.label_set_counts_[label_set]
            if requested <= current:
                continue
            if action == "keep":
                warnings.warn(
                    f"Label set {label_set} appears only once and cannot be "
                    "interpolated with `singleton_strategy='keep'`: its "
                    "single original sample is kept and the requested "
                    f"{requested - current} synthetic sample(s) are not "
                    "generated.",
                    UserWarning,
                    stacklevel=4,
                )
            strategy[label_set] = current
        return strategy

    def _apply_budgets(self, strategy):
        """Enforce total and per-label synthetic budgets."""
        counts = self.label_set_counts_
        synthetic = OrderedDict(
            (label_set, max(0, goal - counts[label_set]))
            for label_set, goal in strategy.items()
        )

        if self.total_budget is not None:
            total_requested = sum(synthetic.values())
            if total_requested > self.total_budget:
                synthetic = self._scale_budget(synthetic, self.total_budget)

        if self.label_budget is not None:
            synthetic, shortfall = self._apply_label_caps(synthetic)
            self.label_budget_shortfall_ = shortfall
            if shortfall and self.enforce_label_budget == "raise":
                details = ", ".join(
                    f"label {label}: short by {gap} synthetic sample(s)"
                    for label, gap in sorted(shortfall.items())
                )
                raise ValueError(
                    "The per-label `label_budget` makes the requested "
                    "resampling target unreachable: " + details + ". Relax the "
                    "corresponding cap, lower the requested "
                    "`sampling_strategy`, or set "
                    "`enforce_label_budget='warn'` to accept the reduced "
                    "budget explicitly."
                )
            if shortfall:
                warnings.warn(
                    "The per-label `label_budget` reduced the plan: "
                    + ", ".join(
                        f"label {label} short by {gap}"
                        for label, gap in sorted(shortfall.items())
                    ),
                    UserWarning,
                    stacklevel=3,
                )
        else:
            self.label_budget_shortfall_ = {}

        return OrderedDict(
            (label_set, counts[label_set] + n_new)
            for label_set, n_new in synthetic.items()
        )

    @staticmethod
    def _scale_budget(synthetic, total_budget):
        """Proportionally scale positive synthetic counts (largest remainder).

        Combinations requesting zero synthetic samples keep their zero
        allocation; only positive requests share the capped budget.
        """
        labels = list(synthetic)
        requested = np.asarray(list(synthetic.values()), dtype=np.float64)
        positive = requested > 0
        if not positive.any():
            return synthetic
        positive_values = requested[positive]
        scale = total_budget / positive_values.sum()
        scaled = positive_values * scale
        floors = np.floor(scaled).astype(np.intp)
        # never allocate more than requested, nor more than the cap
        floors = np.minimum(floors, positive_values.astype(np.intp))
        remainder = min(total_budget, int(positive_values.sum())) - int(floors.sum())
        order = np.argsort(-(scaled - floors), kind="stable")
        for idx in order[: max(0, remainder)]:
            if floors[idx] < positive_values[idx]:
                floors[idx] += 1
        result_values = np.zeros_like(requested, dtype=np.intp)
        result_values[positive] = floors
        return OrderedDict(zip(labels, result_values.tolist()))

    def _apply_label_caps(self, synthetic):
        caps = dict(self.label_budget)
        for label in caps:
            if not isinstance(label, numbers.Integral) or label < 0:
                raise ValueError(
                    "Keys of `label_budget` must be non-negative label "
                    f"indices. Got {label!r}."
                )
        n_labels = self.n_labels_in_
        bad = [label for label in caps if label >= n_labels]
        if bad:
            raise ValueError(
                f"`label_budget` references label index/indices {bad} but the "
                f"target only has {n_labels} label column(s) (indices 0 to "
                f"{n_labels - 1})."
            )
        for label, cap in caps.items():
            if not isinstance(cap, numbers.Integral) or cap < 0:
                raise ValueError(
                    "Values of `label_budget` must be non-negative integers. "
                    f"Got {cap!r} for label {label}."
                )

        # Shave off the largest requested allocations first so the rarest
        # combinations keep as much budget as possible.
        allocation = OrderedDict(synthetic)
        shortfall = {}
        for label, cap in caps.items():
            current = sum(
                n_new for label_set, n_new in allocation.items() if label in label_set
            )
            if current <= cap:
                continue
            excess = current - cap
            # The shortfall is what the cap forbids, regardless of the shaving
            # below: the caller must know that the requested target cannot be
            # reached exactly.
            shortfall[label] = int(excess)
            candidates = sorted(
                (
                    label_set
                    for label_set in allocation
                    if label in label_set and allocation[label_set] > 0
                ),
                key=lambda label_set: self.label_set_counts_[label_set],
                reverse=True,
            )
            for label_set in candidates:
                if excess == 0:
                    break
                cut = min(excess, allocation[label_set])
                allocation[label_set] -= cut
                excess -= cut
        return allocation, shortfall

    # ------------------------------------------------------------------
    # resampling
    # ------------------------------------------------------------------
    def _fit_resample(self, X, y):
        strategy = self._build_strategy(y)
        self.nn_k_ = check_neighbors_object(
            "k_neighbors", self.k_neighbors, additional_neighbor=1
        )
        rng = check_random_state(self.random_state)

        target_is_sparse = sparse.issparse(y)
        output_is_sparse = (
            target_is_sparse if self.output_sparse is None else bool(self.output_sparse)
        )
        target_dtype = y.dtype
        n_labels = self.n_labels_in_

        singleton_plan = self.singleton_plan_

        # Every original row is kept once. Only rows of skipped singletons are
        # dropped, so original rows and labels stay aligned row by row.
        skipped_rows = set()
        for label_set, action in singleton_plan.items():
            if action == "skip":
                skipped_rows.update(self.label_set_indices_[label_set].tolist())

        keep_mask = np.ones(y.shape[0], dtype=bool)
        if skipped_rows:
            keep_mask[np.asarray(sorted(skipped_rows), dtype=np.intp)] = False
        original_positions = np.flatnonzero(keep_mask)

        X_parts = [_safe_indexing(X, original_positions)]
        y_parts = []
        original_y = (
            y[original_positions]
            if target_is_sparse
            else _safe_indexing(y, original_positions)
        )
        if output_is_sparse and not sparse.issparse(original_y):
            original_y = sparse.csr_matrix(original_y)
        elif not output_is_sparse and sparse.issparse(original_y):
            # Explicit user request for a dense target; densify the kept rows
            # only, never the training target as a whole.
            original_y = original_y.toarray()
        y_parts.append(original_y)

        sample_indices = original_positions.tolist()
        n_synthetic = OrderedDict()
        for label_set, target_size in strategy.items():
            current = self.label_set_counts_[label_set]
            n_new = target_size - current
            if n_new <= 0:
                n_synthetic[label_set] = 0
                continue

            action = singleton_plan.get(label_set, "smote")
            if action == "merge":
                donor_set = self._nearest_combination(label_set)
                X_new = self._synthesise(
                    X,
                    base_indices=self.label_set_indices_[label_set],
                    pool_indices=self.label_set_indices_[donor_set],
                    n_new=n_new,
                    rng=rng,
                )
                synthetic_set = donor_set
            else:
                X_new = self._synthesise(
                    X,
                    base_indices=self.label_set_indices_[label_set],
                    pool_indices=self.label_set_indices_[label_set],
                    n_new=n_new,
                    rng=rng,
                )
                synthetic_set = label_set

            X_parts.append(X_new)
            y_parts.append(
                indicator_block(
                    synthetic_set,
                    X_new.shape[0],
                    n_labels,
                    sparse_output=output_is_sparse,
                    dtype=target_dtype,
                )
            )
            sample_indices.extend([-1] * X_new.shape[0])
            n_synthetic[label_set] = int(X_new.shape[0])

        self.n_synthetic_ = n_synthetic
        self.sample_indices_ = np.asarray(sample_indices, dtype=np.intp)

        X_resampled = self._vstack_X(X, X_parts)
        if output_is_sparse:
            y_resampled = sparse.vstack(y_parts, format="csr").astype(
                target_dtype, copy=False
            )
        else:
            y_resampled = np.vstack(y_parts).astype(target_dtype, copy=False)

        self._check_alignment(X_resampled, y_resampled)
        return X_resampled, y_resampled

    def _plan_singletons(self, strategy):
        """Decide how each targeted singleton combination is handled."""
        plan = {}
        for label_set in strategy:
            if label_set == EMPTY_LABEL_SET:
                continue
            if self.label_set_counts_[label_set] == 1:
                plan[label_set] = self.singleton_strategy
        return plan

    def _nearest_combination(self, singleton_set):
        """Find the closest non-singleton, non-empty label set.

        Distance is the symmetric difference size between label sets, breaking
        ties by the larger donor pool. This only inspects label membership, so
        no feature matrix -- and no validation/future data -- is read.
        """
        donors = [
            label_set
            for label_set, count in self.label_set_counts_.items()
            if label_set != EMPTY_LABEL_SET
            and label_set != singleton_set
            and count >= 2
        ]
        if not donors:
            raise ValueError(
                f"Label set {singleton_set} appears once and no other "
                "non-singleton combination is available to merge with. Use "
                "`singleton_strategy='keep'` or 'skip', or provide more "
                "labelled training samples."
            )
        singleton = set(singleton_set)
        donors.sort(
            key=lambda label_set: (
                len(singleton.symmetric_difference(label_set)),
                -self.label_set_counts_[label_set],
            )
        )
        return donors[0]

    def _synthesise(self, X, base_indices, pool_indices, n_new, rng):
        """Generate ``n_new`` SMOTE points inside one donor pool.

        Base points are the samples whose group is being grown; the neighbour
        pool is always a non-singleton group. Both index arrays refer to
        training rows only, so no validation or future-batch data can leak in.
        """
        is_dataframe = hasattr(X, "iloc")
        X_base = _safe_indexing(X, base_indices)
        X_pool = _safe_indexing(X, pool_indices)
        # NearestNeighbors and the interpolation below speak numpy/sparse.
        # DataFrames are converted to values for the computation; the original
        # column names are restored when stacking the output in ``_vstack_X``.
        if is_dataframe:
            X_base_values = X_base.to_numpy()
            X_pool_values = X_pool.to_numpy()
        else:
            X_base_values = X_base
            X_pool_values = X_pool

        nn = clone(self.nn_k_)
        k = min(nn.n_neighbors, X_pool_values.shape[0])
        nn.set_params(n_neighbors=k)
        nn.fit(X_pool_values)
        same_pool = base_indices.shape == pool_indices.shape and np.array_equal(
            base_indices, pool_indices
        )
        neighbours = nn.kneighbors(X_base_values, return_distance=False)
        if same_pool:
            # Drop the trivial self-neighbour, mirroring SMOTE.
            neighbours = neighbours[:, 1:]
        if neighbours.shape[1] == 0:  # pragma: no cover - defensive
            raise ValueError(
                "No usable neighbour was found inside a label-set group with "
                "at least two samples. Please report this bug."
            )

        base_draws = rng.randint(0, X_base_values.shape[0], size=n_new)
        neighbour_cols = rng.randint(0, neighbours.shape[1], size=n_new)
        neighbour_pos = neighbours[base_draws, neighbour_cols]
        steps = rng.uniform(size=n_new)
        X_seed = X_base_values[base_draws]
        X_neighbour = X_pool_values[neighbour_pos]
        if sparse.issparse(X_pool_values):
            step_matrix = X_pool_values.__class__(steps[:, None])
            X_new = X_seed + step_matrix.multiply(X_neighbour - X_seed)
            X_new = X_new.astype(X_pool_values.dtype)
        else:
            X_new = X_seed + steps[:, None] * (X_neighbour - X_seed)
            X_new = np.asarray(X_new, dtype=X_base_values.dtype)
            if is_dataframe:
                import pandas as pd

                X_new = pd.DataFrame(X_new, columns=X_base.columns)
        return X_new

    # ------------------------------------------------------------------
    # output helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _vstack_X(X, parts):
        if sparse.issparse(X):
            return sparse.vstack(parts, format=X.format)
        if hasattr(X, "iloc"):  # pandas DataFrame
            import pandas as pd

            return pd.concat(parts, axis=0, ignore_index=True)
        return np.vstack(parts)

    def _check_alignment(self, X_resampled, y_resampled):
        if X_resampled.shape[0] != y_resampled.shape[0]:
            raise RuntimeError(
                "Internal error: resampled X and y are misaligned "
                f"({X_resampled.shape[0]} vs {y_resampled.shape[0]} rows). "
                "Please report this bug."
            )
        if y_resampled.shape[1] != self.n_labels_in_:
            raise RuntimeError(
                "Internal error: the resampled target lost label columns "
                f"({y_resampled.shape[1]} vs {self.n_labels_in_})."
            )
        if self.sample_indices_.shape[0] != y_resampled.shape[0]:
            raise RuntimeError(
                "Internal error: `sample_indices_` does not match the "
                "resampled number of rows. Please report this bug."
            )

    def _more_tags(self):
        return {
            "X_types": ["2darray", "sparse", "dataframe"],
            "sample_indices": True,
        }

    def __sklearn_tags__(self):
        from sklearn_compat.utils._tags import TargetTags

        from imblearn.utils._tags import InputTags, SamplerTags, Tags

        tags = Tags(
            estimator_type="sampler",
            target_tags=TargetTags(required=True, two_d_labels=True),
            transformer_tags=None,
            regressor_tags=None,
            classifier_tags=None,
            sampler_tags=SamplerTags(sample_indices=True),
        )
        tags.input_tags = InputTags()
        tags.input_tags.two_d_array = True
        tags.input_tags.sparse = True
        tags.input_tags.dataframe = True
        return tags
