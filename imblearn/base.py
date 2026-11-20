"""Base class for sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin
from sklearn.preprocessing import label_binarize
from sklearn.utils._metadata_requests import METHODS
from sklearn.utils.multiclass import check_classification_targets
from sklearn_compat.base import _fit_context
from sklearn_compat.utils.validation import validate_data

from imblearn.utils import check_sampling_strategy, check_target_type
from imblearn.utils._memory import MemoryGuard, validate_chunk_size
from imblearn.utils._tags import get_tags
from imblearn.utils._validation import ArraysTransformer

if "fit_predict" not in METHODS:
    METHODS.append("fit_predict")
if "fit_transform" not in METHODS:
    METHODS.append("fit_transform")
METHODS.append("fit_resample")

try:
    from sklearn.utils._metadata_requests import SIMPLE_METHODS

    SIMPLE_METHODS.append("fit_resample")
except ImportError:
    # in older versions of scikit-learn, only METHODS is used
    pass


class SamplerMixin(metaclass=ABCMeta):
    """Mixin class for samplers with abstract method.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _estimator_type = "sampler"

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, **params):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        X, y, _ = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_resample(self, X, y, **params):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        check_classification_targets(y)
        original_X = X
        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        memory_guard = getattr(self, "memory_guard", False)
        chunk_size = validate_chunk_size(getattr(self, "chunk_size", None))
        guard = None
        if memory_guard:
            # Fitted attributes specific to the memory guard are only created
            # when the feature is opted in, keeping the historical attribute
            # set unchanged otherwise.
            self.degraded_ = False
            guard = MemoryGuard(
                self,
                X,
                y,
                self._expected_n_output(X, y),
                chunk_size,
                self._make_memory_profile,
            )
            if not guard.preflight():
                # Degraded path: no resampling, original arrays are returned.
                self.degraded_ = True
                self.degradation_message_ = guard.message
                return X, y

        try:
            output = self._fit_resample(X, y, **params)
        except MemoryError as error:
            if not memory_guard:
                raise
            # Release every partially allocated output before degrading.
            self._release_partial_outputs()
            guard.handle_memory_error(error)
            self.degraded_ = True
            self.degradation_message_ = guard.message
            return X, y

        y_ = (
            label_binarize(output[1], classes=np.unique(y)) if binarize_y else output[1]
        )

        X_, y_ = arrays_transformer.transform(output[0], y_)
        # The chunked path builds ndarrays and therefore loses the original
        # row labels. Re-align them with the selected indices so that chunked
        # and regular paths produce identical pandas objects.
        sample_indices = getattr(self, "sample_indices_", None)
        if (
            sample_indices is not None
            and hasattr(original_X, "columns")
            and hasattr(X_, "columns")
        ):
            try:
                new_index = original_X.index.to_numpy()[sample_indices]
                X_.index = new_index
                if hasattr(y_, "index"):
                    y_.index = new_index
            except (IndexError, TypeError):
                pass
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    @abstractmethod
    def _fit_resample(self, X, y, **params):
        """Base method defined in each sampler to defined the sampling
        strategy.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray of shape (n_samples_new,)
            The corresponding label of `X_resampled`.

        """
        pass


class BaseSampler(SamplerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, sampling_strategy="auto"):
        self.sampling_strategy = sampling_strategy

    # ------------------------------------------------------------------
    # Memory contract. The defaults describe a sampler whose memory
    # characteristics are unknown to the framework: it does not claim a
    # chunked path. Samplers with a bounded-memory implementation override
    # ``_make_memory_profile`` (and expose ``chunk_size`` / ``memory_guard``
    # constructor parameters).
    # ------------------------------------------------------------------

    #: Whether the regular implementation copies the full input array.
    _copies_input_default = True
    #: Whether a bounded-memory chunked path exists for this sampler.
    _supports_chunking_default = False

    def memory_profile(self, X, y):
        """Report the memory characteristics of this sampler for ``X`` and ``y``.

        The method never copies the data: it only inspects array metadata and
        the resolved :attr:`sampling_strategy_` (it calls :meth:`fit` first
        when the strategy has not been resolved yet).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        profile : :class:`~imblearn.utils._memory.MemoryProfile`
            Dataclass exposing ``copies_input``, ``chunked_supported``,
            estimated peak extra bytes for the full and chunked paths, the
            output size, the scaling law and implementation notes.
        """
        if not hasattr(self, "sampling_strategy_"):
            self.fit(X, y)
        X_validated, y_validated, _ = self._check_X_y(X, y)
        n_output = self._expected_n_output(X_validated, y_validated)
        return self._make_memory_profile(
            X_validated,
            y_validated,
            n_output,
            validate_chunk_size(getattr(self, "chunk_size", None)),
        )

    def _expected_n_output(self, X, y):
        """Best-effort number of output rows, used for memory estimates."""
        strategy = getattr(self, "sampling_strategy_", None)
        if strategy is None:
            return X.shape[0]
        n_output = X.shape[0]
        if self._sampling_type == "over-sampling":
            # Only the targeted classes grow; strategy values are the final
            # number of samples requested for each of them.
            for klass, target in strategy.items():
                current = int(np.count_nonzero(y == klass))
                n_output += max(0, int(target) - current)
        elif self._sampling_type in ("under-sampling", "clean-sampling"):
            for klass, target in strategy.items():
                current = int(np.count_nonzero(y == klass))
                n_output -= max(0, current - int(target))
        return n_output

    def _make_memory_profile(self, X, y, n_output, chunk_size):
        from imblearn.utils._memory import (
            MemoryProfile,
            _array_nbytes,
            _dtype_nbytes,
            _supports_chunking,
            estimate_index_bytes,
        )

        output_bytes = n_output * (
            _array_nbytes(X) // max(1, X.shape[0])
        ) + n_output * _dtype_nbytes(y)
        chunked_supported = self._supports_chunking_default and _supports_chunking(X)
        peak_chunked = None
        temp_chunked = None
        notes = (
            "Generic profile: output allocation plus sampler-specific "
            "temporary arrays; consult the sampler documentation for its "
            "copy points.",
        )
        if chunked_supported and chunk_size is not None:
            row_bytes = _array_nbytes(X) // max(1, X.shape[0])
            temp_chunked = (
                estimate_index_bytes(X.shape[0], n_output)
                + chunk_size * row_bytes
            )
            peak_chunked = output_bytes + temp_chunked
        return MemoryProfile(
            copies_input=self._copies_input_default,
            chunked_supported=chunked_supported,
            peak_extra_bytes_full=output_bytes + _array_nbytes(X),
            peak_extra_bytes_chunked=peak_chunked,
            output_bytes=output_bytes,
            scaling="O(n * p) plus output O(n_output * p)",
            temp_extra_bytes_chunked=temp_chunked,
            notes=notes,
        )

    def _release_partial_outputs(self):
        """Drop sampler-specific partial allocations after a MemoryError."""
        for attr in ("sample_indices_", "shrinkage_"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _check_X_y(self, X, y, accept_sparse=None):
        if accept_sparse is None:
            accept_sparse = ["csr", "csc"]
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = validate_data(self, X=X, y=y, reset=True, accept_sparse=accept_sparse)
        return X, y, binarize_y

    def fit(self, X, y, **params):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        return super().fit(X, y, **params)

    def fit_resample(self, X, y, **params):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        return super().fit_resample(X, y, **params)

    def _more_tags(self):
        return {"X_types": ["2darray", "sparse", "dataframe"]}

    def __sklearn_tags__(self):
        from sklearn_compat.utils._tags import TargetTags

        from imblearn.utils._tags import InputTags, SamplerTags, Tags

        tags = Tags(
            estimator_type="sampler",
            target_tags=TargetTags(required=True),
            transformer_tags=None,
            regressor_tags=None,
            classifier_tags=None,
            sampler_tags=SamplerTags(),
        )
        tags.input_tags = InputTags()
        tags.input_tags.two_d_array = True
        tags.input_tags.sparse = True
        tags.input_tags.dataframe = True
        return tags


def _identity(X, y):
    return X, y


def is_sampler(estimator):
    """Return True if the given estimator is a sampler, False otherwise.

    Parameters
    ----------
    estimator : object
        Estimator to test.

    Returns
    -------
    is_sampler : bool
        True if estimator is a sampler, otherwise False.
    """

    if hasattr(estimator, "_estimator_type") and estimator._estimator_type == "sampler":
        return True
    tags = get_tags(estimator)
    if hasattr(tags, "sampler_tags") and tags.sampler_tags is not None:
        return True
    return False


class FunctionSampler(BaseSampler):
    """Construct a sampler from calling an arbitrary callable.

    Read more in the :ref:`User Guide <function_sampler>`.

    Parameters
    ----------
    func : callable, default=None
        The callable to use for the transformation. This will be passed the
        same arguments as transform, with args and kwargs forwarded. If func is
        None, then func will be the identity function.

    accept_sparse : bool, default=True
        Whether sparse input are supported. By default, sparse inputs are
        supported.

    kw_args : dict, default=None
        The keyword argument expected by ``func``.

    validate : bool, default=True
        Whether or not to bypass the validation of ``X`` and ``y``. Turning-off
        validation allows to use the ``FunctionSampler`` with any type of
        data.

        .. versionadded:: 0.6

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    sklearn.preprocessing.FunctionTransfomer : Stateless transformer.

    Notes
    -----
    See
    :ref:`sphx_glr_auto_examples_applications_plot_outlier_rejections.py`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from imblearn import FunctionSampler
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

    We can create to select only the first ten samples for instance.

    >>> def func(X, y):
    ...   return X[:10], y[:10]
    >>> sampler = FunctionSampler(func=func)
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> np.all(X_res == X[:10])
    True
    >>> np.all(y_res == y[:10])
    True

    We can also create a specific function which take some arguments.

    >>> from collections import Counter
    >>> from imblearn.under_sampling import RandomUnderSampler
    >>> def func(X, y, sampling_strategy, random_state):
    ...   return RandomUnderSampler(
    ...       sampling_strategy=sampling_strategy,
    ...       random_state=random_state).fit_resample(X, y)
    >>> sampler = FunctionSampler(func=func,
    ...                           kw_args={'sampling_strategy': 'auto',
    ...                                    'random_state': 0})
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> print(f'Resampled dataset shape {sorted(Counter(y_res).items())}')
    Resampled dataset shape [(0, 100), (1, 100)]
    """

    _sampling_type = "bypass"

    _parameter_constraints: dict = {
        "func": [callable, None],
        "accept_sparse": ["boolean"],
        "kw_args": [dict, None],
        "validate": ["boolean"],
    }

    def __init__(self, *, func=None, accept_sparse=True, kw_args=None, validate=True):
        super().__init__()
        self.func = func
        self.accept_sparse = accept_sparse
        self.kw_args = kw_args
        self.validate = validate

    def fit(self, X, y):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        self._validate_params()
        # we need to overwrite SamplerMixin.fit to bypass the validation
        if self.validate:
            check_classification_targets(y)
            X, y, _ = self._check_X_y(X, y, accept_sparse=self.accept_sparse)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        return self

    def fit_resample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        self._validate_params()
        arrays_transformer = ArraysTransformer(X, y)

        if self.validate:
            check_classification_targets(y)
            X, y, binarize_y = self._check_X_y(X, y, accept_sparse=self.accept_sparse)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        output = self._fit_resample(X, y)

        if self.validate:
            y_ = (
                label_binarize(output[1], classes=np.unique(y))
                if binarize_y
                else output[1]
            )
            X_, y_ = arrays_transformer.transform(output[0], y_)
            return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

        return output

    def _fit_resample(self, X, y):
        func = _identity if self.func is None else self.func
        output = func(X, y, **(self.kw_args if self.kw_args else {}))
        return output
