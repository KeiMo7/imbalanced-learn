"""Class to perform random over-sampling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections.abc import Mapping
from numbers import Real

import numpy as np
from scipy import sparse
from sklearn.utils import _safe_indexing, check_array, check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn_compat.utils.validation import validate_data

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution, check_target_type
from imblearn.utils._docstring import _random_state_docstring
from imblearn.utils._memory import (
    MEMORY_PARAM_CONSTRAINTS,
    MemoryProfile,
    _array_nbytes,
    _dtype_nbytes,
    _supports_chunking,
    coerce_dense_chunkable,
    estimate_index_bytes,
    fill_by_indices_chunked,
    validate_chunk_size,
)
from imblearn.utils._validation import _check_X


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class RandomOverSampler(BaseOverSampler):
    """Class to perform random over-sampling.

    Object to over-sample the minority class(es) by picking samples at random
    with replacement. The bootstrap can be generated in a smoothed manner.

    Read more in the :ref:`User Guide <random_over_sampler>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    shrinkage : float or dict, default=None
        Parameter controlling the shrinkage applied to the covariance matrix.
        when a smoothed bootstrap is generated. The options are:

        - if `None`, a normal bootstrap will be generated without perturbation.
          It is equivalent to `shrinkage=0` as well;
        - if a `float` is given, the shrinkage factor will be used for all
          classes to generate the smoothed bootstrap;
        - if a `dict` is given, the shrinkage factor will specific for each
          class. The key correspond to the targeted class and the value is
          the shrinkage factor.

        The value needs of the shrinkage parameter needs to be higher or equal
        to 0.

        .. versionadded:: 0.8

    chunk_size : int or None, default=None
        Number of rows materialised per chunk. When set on a plain bootstrap
        (``shrinkage=None``), the resampled arrays are pre-allocated and
        filled chunk by chunk so the temporary memory footprint stays bounded
        by ``chunk_size * n_features`` instead of duplicating the input and
        accumulating per-class arrays. The bootstrap indices are computed
        globally with the same random draws, hence the chunked path returns
        identical results to the regular path for a given ``random_state``.
        Only dense arrays are supported; the smoothed bootstrap
        (``shrinkage`` not ``None``) keeps using the regular path.

        .. versionadded:: 0.15

    memory_guard : bool, default=False
        When ``True``, estimate the peak extra memory before resampling and
        degrade to returning the input unchanged (with a
        :class:`~python:warnings.UserWarning`) instead of raising a
        :class:`MemoryError`.

        .. versionadded:: 0.15

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    shrinkage_ : dict or None
        The per-class shrinkage factor used to generate the smoothed bootstrap
        sample. When `shrinkage=None` a normal bootstrap will be generated.

        .. versionadded:: 0.8

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    degraded_ : bool
        Defined only when ``memory_guard=True``. ``True`` means the sampler
        skipped resampling and returned the input unchanged because the
        estimated peak memory was too high or a :class:`MemoryError`
        occurred; ``False`` means the resampling completed normally.

        .. versionadded:: 0.15

    degradation_message_ : str
        Defined only when ``degraded_`` is ``True``. Human-readable
        explanation of the degradation with actionable advice.

        .. versionadded:: 0.15

    See Also
    --------
    BorderlineSMOTE : Over-sample using the borderline-SMOTE variant.

    SMOTE : Over-sample using SMOTE.

    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

    SMOTEN : Over-sample using the SMOTE variant specifically for categorical
        features only.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    When generating a smoothed bootstrap, this method is also known as Random
    Over-Sampling Examples (ROSE) [1]_.

    .. warning::
       Since smoothed bootstrap are generated by adding a small perturbation
       to the drawn samples, this method is not adequate when working with
       sparse matrices.

    References
    ----------
    .. [1] G Menardi, N. Torelli, "Training and assessing classification
       rules with imbalanced data," Data Mining and Knowledge
       Discovery, 28(1), pp.92-122, 2014.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import RandomOverSampler
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> ros = RandomOverSampler(random_state=42)
    >>> X_res, y_res = ros.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})
    """

    _parameter_constraints: dict = {
        **BaseOverSampler._parameter_constraints,
        "shrinkage": [Interval(Real, 0, None, closed="left"), dict, None],
        **MEMORY_PARAM_CONSTRAINTS,
    }

    # The regular path seeds the output with ``X.copy()`` and accumulates a
    # per-class copy of every bootstrapped row before ``vstack``: it
    # duplicates the input and can transiently hold ~2-3x its size.
    _copies_input_default = True
    _supports_chunking_default = True

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        shrinkage=None,
        chunk_size=None,
        memory_guard=False,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.shrinkage = shrinkage
        self.chunk_size = chunk_size
        self.memory_guard = memory_guard

    def _check_X_y(self, X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X = _check_X(X)
        validate_data(self, X=X, y=y, reset=True, skip_check_array=True)
        return X, y, binarize_y

    def _build_bootstrap_indices(self, X, y):
        """Resolve shrinkage settings and draw every bootstrap index.

        All random number generation happens here, in the exact same order as
        in the historical implementation, so that regular and chunked paths
        share the same draws for a given ``random_state``.
        """
        random_state = check_random_state(self.random_state)

        if isinstance(self.shrinkage, Real):
            self.shrinkage_ = dict.fromkeys(self.sampling_strategy_, self.shrinkage)
        elif self.shrinkage is None or isinstance(self.shrinkage, Mapping):
            self.shrinkage_ = self.shrinkage

        if self.shrinkage_ is not None:
            missing_shrinkage_keys = (
                self.sampling_strategy_.keys() - self.shrinkage_.keys()
            )
            if missing_shrinkage_keys:
                raise ValueError(
                    "`shrinkage` should contain a shrinkage factor for "
                    "each class that will be resampled. The missing "
                    f"classes are: {repr(missing_shrinkage_keys)}"
                )

            for klass, shrink_factor in self.shrinkage_.items():
                if shrink_factor < 0:
                    raise ValueError(
                        "The shrinkage factor needs to be >= 0. "
                        f"Got {shrink_factor} for class {klass}."
                    )

        sample_indices = range(X.shape[0])
        bootstrap_per_class = []
        for class_sample, num_samples in self.sampling_strategy_.items():
            target_class_indices = np.flatnonzero(y == class_sample)
            bootstrap_indices = random_state.choice(
                target_class_indices,
                size=num_samples,
                replace=True,
            )
            sample_indices = np.append(sample_indices, bootstrap_indices)
            bootstrap_per_class.append(
                (class_sample, num_samples, target_class_indices, bootstrap_indices)
            )

        self.sample_indices_ = np.array(sample_indices)
        return random_state, bootstrap_per_class

    def _fit_resample(self, X, y):
        chunk_size = validate_chunk_size(self.chunk_size)
        random_state, bootstrap_per_class = self._build_bootstrap_indices(X, y)

        if chunk_size is not None:
            if self.shrinkage_ is not None:
                raise ValueError(
                    "The chunked path of RandomOverSampler does not support "
                    "the smoothed bootstrap (`shrinkage` is not None). Use "
                    "`chunk_size=None` with shrinkage, or remove the "
                    "shrinkage to enable chunking."
                )
            # One explicit, documented conversion for pandas inputs; object
            # and sparse inputs raise with an actionable message.
            X_dense = coerce_dense_chunkable(X)
            y_dense = np.asarray(y)
            n_output = self.sample_indices_.shape[0]
            X_resampled = np.empty_like(X_dense, shape=(n_output, X_dense.shape[1]))
            y_resampled = np.empty_like(y_dense, shape=(n_output,))
            fill_by_indices_chunked(
                X_dense, self.sample_indices_, X_resampled, chunk_size
            )
            fill_by_indices_chunked(
                y_dense.reshape(-1, 1),
                self.sample_indices_,
                y_resampled.reshape(-1, 1),
                chunk_size,
            )
            return X_resampled, y_resampled

        if self.shrinkage_ is not None:
            # smoothed bootstrap imposes to make numerical operation; we need
            # to be sure to have only numerical data in X
            try:
                X = check_array(X, accept_sparse=["csr", "csc"], dtype="numeric")
            except ValueError as exc:
                raise ValueError(
                    "When shrinkage is not None, X needs to contain only "
                    "numerical data to later generate a smoothed bootstrap "
                    "sample."
                ) from exc

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for (
            class_sample,
            num_samples,
            target_class_indices,
            bootstrap_indices,
        ) in bootstrap_per_class:
            if self.shrinkage_ is not None:
                # generate a smoothed bootstrap with a perturbation
                n_samples, n_features = X.shape
                smoothing_constant = (4 / ((n_features + 2) * n_samples)) ** (
                    1 / (n_features + 4)
                )
                if sparse.issparse(X):
                    _, X_class_variance = mean_variance_axis(
                        X[target_class_indices, :],
                        axis=0,
                    )
                    X_class_scale = np.sqrt(X_class_variance, out=X_class_variance)
                else:
                    X_class_scale = np.std(X[target_class_indices, :], axis=0)
                smoothing_matrix = np.diagflat(
                    self.shrinkage_[class_sample] * smoothing_constant * X_class_scale
                )
                X_new = random_state.randn(num_samples, n_features)
                X_new = X_new.dot(smoothing_matrix) + X[bootstrap_indices, :]
                if sparse.issparse(X):
                    X_new = sparse.csr_matrix(X_new, dtype=X.dtype)
                X_resampled.append(X_new)
            else:
                # generate a bootstrap
                X_resampled.append(_safe_indexing(X, bootstrap_indices))

            y_resampled.append(_safe_indexing(y, bootstrap_indices))

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled

    def _make_memory_profile(self, X, y, n_output, chunk_size):
        row_bytes = _array_nbytes(X) // max(1, X.shape[0])
        output_bytes = n_output * row_bytes + n_output * _dtype_nbytes(y)
        # Regular path keeps X.copy(), every per-class bootstrap copy and the
        # final vstacked output alive at the same time.
        peak_full = output_bytes + 2 * _array_nbytes(X)
        notes = [
            "Regular path copies the whole input (X.copy()) and keeps every "
            "per-class bootstrap array plus the vstacked output: peak ~2x "
            "input plus the output.",
        ]
        chunked_supported = _supports_chunking(X)
        if chunked_supported:
            notes.append(
                "Chunked path pre-allocates the output once and fills it in "
                "place, removing the X.copy() and per-class accumulators."
            )
        else:
            notes.append("Sparse or heterogeneous input: no chunked path is offered.")
        peak_chunked = None
        temp_chunked = None
        if chunked_supported and chunk_size is not None:
            temp_chunked = (
                estimate_index_bytes(X.shape[0], n_output)
                + chunk_size * row_bytes
            )
            peak_chunked = output_bytes + temp_chunked
        if getattr(self, "shrinkage_", None) is not None:
            peak_full += _array_nbytes(X)
            notes.append(
                "Smoothed bootstrap additionally allocates per-class "
                "smoothing matrices and randn draws; chunking is not "
                "supported with shrinkage."
            )
            chunked_supported = False
            peak_chunked = None
            temp_chunked = None
        return MemoryProfile(
            copies_input=True,
            chunked_supported=chunked_supported,
            peak_extra_bytes_full=peak_full,
            peak_extra_bytes_chunked=peak_chunked,
            output_bytes=output_bytes,
            scaling="over-sampling: O((1 + alpha) * n * p) where alpha is the "
            "fraction of added minority rows",
            temp_extra_bytes_chunked=temp_chunked,
            notes=tuple(notes),
        )

    def _more_tags(self):
        return {
            "X_types": ["2darray", "string", "sparse", "dataframe"],
            "sample_indices": True,
            "allow_nan": True,
            "_xfail_checks": {
                "check_complex_data": "Robust to this type of data.",
            },
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.sampler_tags.sample_indices = True
        return tags
