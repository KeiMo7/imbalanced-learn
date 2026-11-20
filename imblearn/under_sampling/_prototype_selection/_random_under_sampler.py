"""Class to perform random under-sampling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
from sklearn.utils import _safe_indexing, check_random_state
from sklearn_compat.utils.validation import validate_data

from imblearn.under_sampling.base import BaseUnderSampler
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
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class RandomUnderSampler(BaseUnderSampler):
    """Class to perform random under-sampling.

    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.

    Read more in the :ref:`User Guide <controlled_under_sampling>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    replacement : bool, default=False
        Whether the sample is with or without replacement.

    chunk_size : int or None, default=None
        Number of rows materialised per chunk. When set, the resampled
        arrays are pre-allocated and filled chunk by chunk so the temporary
        memory footprint stays bounded by ``chunk_size * n_features``. The
        selection indices are computed globally with the same random draws,
        hence the chunked path returns identical results to the regular path
        for a given ``random_state``. Only dense arrays are supported.

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
    NearMiss : Undersample using near-miss samples.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import RandomUnderSampler
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> rus = RandomUnderSampler(random_state=42)
    >>> X_res, y_res = rus.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 100, 1: 100}})
    """

    _parameter_constraints: dict = {
        **BaseUnderSampler._parameter_constraints,
        "replacement": ["boolean"],
        "random_state": ["random_state"],
        **MEMORY_PARAM_CONSTRAINTS,
    }

    # The regular path only builds index arrays and then a single output
    # array via safe indexing: it never copies the full input up-front.
    _copies_input_default = False
    _supports_chunking_default = True

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        replacement=False,
        chunk_size=None,
        memory_guard=False,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.replacement = replacement
        self.chunk_size = chunk_size
        self.memory_guard = memory_guard

    def _check_X_y(self, X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X = _check_X(X)
        X, y = validate_data(self, X=X, y=y, reset=True, skip_check_array=True)
        return X, y, binarize_y

    def _select_indices(self, X, y):
        """Compute the selection indices using the historical RNG sequence.

        This is shared by the regular and chunked paths so that both produce
        the exact same draws for a given ``random_state``.
        """
        random_state = check_random_state(self.random_state)
        idx_under = np.empty((0,), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                index_target_class = random_state.choice(
                    range(np.count_nonzero(y == target_class)),
                    size=n_samples,
                    replace=self.replacement,
                )
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (
                    idx_under,
                    np.flatnonzero(y == target_class)[index_target_class],
                ),
                axis=0,
            )
        return idx_under

    def _fit_resample(self, X, y):
        chunk_size = validate_chunk_size(self.chunk_size)
        idx_under = self._select_indices(X, y)
        self.sample_indices_ = idx_under

        if chunk_size is not None:
            # One explicit, documented conversion for pandas inputs; object
            # and sparse inputs raise with an actionable message.
            X_dense = coerce_dense_chunkable(X)
            y_dense = np.asarray(y)
            X_resampled = np.empty_like(
                X_dense, shape=(idx_under.shape[0], X_dense.shape[1])
            )
            y_resampled = np.empty_like(y_dense, shape=(idx_under.shape[0],))
            fill_by_indices_chunked(X_dense, idx_under, X_resampled, chunk_size)
            fill_by_indices_chunked(
                y_dense.reshape(-1, 1),
                idx_under,
                y_resampled.reshape(-1, 1),
                chunk_size,
            )
            return X_resampled, y_resampled

        return _safe_indexing(X, idx_under), _safe_indexing(y, idx_under)

    def _make_memory_profile(self, X, y, n_output, chunk_size):
        row_bytes = _array_nbytes(X) // max(1, X.shape[0])
        output_bytes = n_output * row_bytes + n_output * _dtype_nbytes(y)
        index_bytes = estimate_index_bytes(X.shape[0], n_output)
        # Regular path: one output array + O(n) index bookkeeping. The input
        # is not duplicated, so chunking cannot shrink the peak below it; it
        # only bounds the size of each temporary fancy-indexed materialisation.
        peak_full = output_bytes + index_bytes
        chunked_supported = _supports_chunking(X)
        peak_chunked = None
        temp_chunked = None
        if chunked_supported and chunk_size is not None:
            temp_chunked = chunk_size * row_bytes
            # The output itself dominates; the per-chunk scratch space adds at
            # most ``chunk_size`` rows, so the peak stays comparable to the
            # regular path rather than exploding with the input.
            peak_chunked = max(peak_full, output_bytes + temp_chunked)
        notes = (
            "Input is never copied: only selection indices and the output "
            "arrays are allocated.",
            "Under-sampling already shrinks the data, so the chunked path "
            "does not lower the peak: it bounds the size of each temporary "
            "fancy-indexed copy to chunk_size rows.",
        )
        return MemoryProfile(
            copies_input=False,
            chunked_supported=chunked_supported,
            peak_extra_bytes_full=peak_full,
            peak_extra_bytes_chunked=peak_chunked,
            output_bytes=output_bytes,
            scaling="under-sampling: O(n_output * p); extra memory decreases "
            "linearly with the number of removed rows",
            temp_extra_bytes_chunked=temp_chunked,
            notes=notes,
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
