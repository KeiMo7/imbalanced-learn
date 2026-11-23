"""Base class for sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin
from sklearn.preprocessing import label_binarize
from sklearn.utils import _safe_indexing
from sklearn.utils._metadata_requests import METHODS
from sklearn.utils.multiclass import check_classification_targets
from sklearn_compat.base import _fit_context
from sklearn_compat.utils.validation import validate_data

from imblearn.utils import check_sampling_strategy, check_target_type
from imblearn.utils._sample_weight import (
    WEIGHT_STRATEGIES,
    _as_weight_array,
    check_effective_sample_weight,
    log_weight_summary,
    reject_unsupported_weight_strategy,
    validate_weight_strategy,
)
from imblearn.utils._tags import get_tags
from imblearn.utils._validation import ArraysTransformer, WeightArraysTransformer

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

    #: Whether the sampler implements the ``sample_weight`` contract.  Derived
    #: classes opt in; the default keeps third-party samplers on the
    #: historical, weight-agnostic path.
    _sample_weight_supported = False
    #: How the resampled weights are obtained.  ``"indices"`` means the
    #: sampler exposes ``sample_indices_`` aligned with its output and the
    #: weights are a row subset; ``"generated"`` means the sampler creates new
    #: rows and computes their weights itself; ``"none"`` means the sampler
    #: cannot propagate weights.
    _weight_propagation = "none"
    _weight_strategies = WEIGHT_STRATEGIES

    def _check_weight_params(self):
        """Validate the weight-related parameters before any sampling work.

        This is deliberately separated from the actual resampling so that a
        sampling chain fails *before* any data is touched when a rule is not
        supported, instead of failing half-way through.
        """
        weight_strategy = getattr(self, "weight_strategy", None)
        if weight_strategy is not None:
            reject_unsupported_weight_strategy(self)
            self._weight_strategy_ = validate_weight_strategy(
                weight_strategy,
                supported=self._weight_strategies,
                sampler_name=self.__class__.__name__,
            )
        else:
            self._weight_strategy_ = None

    def _accepts_sample_weight(self):
        """Whether this sampler implements the sample-weight contract."""
        return getattr(self, "_sample_weight_supported", False)

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
        self._check_weight_params()
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_resample(self, X, y, sample_weight=None, class_weight=None, **params):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights associated with each row of ``X`` and ``y``.  When
            provided, the sampler returns the resampled weights as a third
            output.  Original weights are kept for the retained samples; the
            weights of synthesised samples follow ``weight_strategy``.

            See :ref:`sample_weight` for the exact semantics.

            .. versionadded:: 0.15

        class_weight : None, "balanced" or dict, default=None
            Class-level multiplicative factors applied on top of
            ``sample_weight`` (``effective = sample_weight * class_weight[y]``).
            When a sampler exposes a ``class_weight`` constructor parameter,
            that value is used unless this argument is provided.  It never
            changes the number of samples requested by ``sampling_strategy``.

            .. versionadded:: 0.15

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.

        sample_weight_resampled : array-like of shape (n_samples_new,)
            The weight of each resampled row, aligned row-wise with
            ``X_resampled`` and ``y_resampled``.  Only returned when
            ``sample_weight`` is not ``None``.
        """
        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)

        if class_weight is not None:
            params.pop("class_weight", None)
        else:
            class_weight = params.pop("class_weight", None)
        if class_weight is None:
            class_weight = getattr(self, "class_weight", None)

        weight_transformer = WeightArraysTransformer(sample_weight)

        if sample_weight is not None or class_weight is not None:
            n_samples = getattr(y, "shape", None)
            n_samples = n_samples[0] if n_samples is not None else len(y)
            if sample_weight is not None:
                sample_weight = _as_weight_array(
                    sample_weight,
                    n_samples=n_samples,
                    sampler_name=self.__class__.__name__,
                )

        X, y, binarize_y = self._check_X_y(X, y)

        self._check_weight_params()

        effective_weight = check_effective_sample_weight(
            sample_weight,
            class_weight,
            y,
            sampler_name=self.__class__.__name__,
        )

        if effective_weight is None:
            self.sampling_strategy_ = check_sampling_strategy(
                self.sampling_strategy, y, self._sampling_type
            )
            # Historical path: no weight argument is forwarded at all, so the
            # random sequence and the outputs are strictly unchanged.
            output = self._fit_resample(X, y, **params)
            y_ = (
                label_binarize(output[1], classes=np.unique(y))
                if binarize_y
                else output[1]
            )
            X_, y_ = arrays_transformer.transform(output[0], y_)
            return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

        if not self._accepts_sample_weight():
            raise ValueError(
                f"{self.__class__.__name__} does not support 'sample_weight'. "
                "Remove the weights or use a sampler that implements the "
                "sample-weight contract."
            )

        log_weight_summary(self.__class__.__name__, effective_weight)

        return self._fit_resample_weighted(
            X,
            y,
            effective_weight,
            binarize_y=binarize_y,
            arrays_transformer=arrays_transformer,
            weight_transformer=weight_transformer,
            params=params,
        )

    def _fit_resample_weighted(
        self,
        X,
        y,
        sample_weight,
        *,
        binarize_y,
        arrays_transformer,
        weight_transformer,
        params,
    ):
        """Run a sampler that received sample weights.

        Zero-weight samples are removed before any sampling decision: they
        must not appear in neighbour searches nor be used to synthesise new
        samples.  The ``sample_indices_`` attribute, when present, is mapped
        back to the original (unfiltered) rows so that downstream consumers
        such as the balanced ensembles keep indexing the original data.
        """
        positive = sample_weight > 0
        if positive.all():
            X_in, y_in, w_in = X, y, sample_weight
            mapping = None
        else:
            X_in = _safe_indexing(X, positive)
            y_in = _safe_indexing(y, positive)
            w_in = sample_weight[positive]
            mapping = np.flatnonzero(positive)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y_in, self._sampling_type
        )

        propagation = getattr(self, "_weight_propagation", "generated")
        if propagation == "indices":
            output = self._fit_resample(X_in, y_in, **params)
            weights = self._weights_from_indices(output, w_in)
        elif propagation == "chain":
            output = self._fit_resample(X_in, y_in, sample_weight=w_in, **params)
            if len(output) != 3:
                raise ValueError(
                    f"{self.__class__.__name__} did not propagate 'sample_weight' "
                    "through its sampling chain."
                )
            weights = np.asarray(output[2], dtype=float)
        elif propagation == "generated":
            output = self._fit_resample(X_in, y_in, sample_weight=w_in, **params)
            if len(output) != 3:
                raise ValueError(
                    f"{self.__class__.__name__} did not return resampled "
                    "'sample_weight'."
                )
            weights = np.asarray(output[2], dtype=float)
            output = output[:2]
        else:  # pragma: no cover - defensive
            raise ValueError(
                f"{self.__class__.__name__} declares an unknown weight "
                f"propagation mode {propagation!r}."
            )

        if mapping is not None:
            sample_indices = getattr(self, "sample_indices_", None)
            if sample_indices is not None:
                self.sample_indices_ = mapping[np.asarray(sample_indices, dtype=int)]

        y_ = (
            label_binarize(output[1], classes=np.unique(y_in))
            if binarize_y
            else output[1]
        )
        X_, y_ = arrays_transformer.transform(output[0], y_)

        weights = weight_transformer.transform(weights, X_, y_, output[1])
        return X_, y_, weights

    def _weights_from_indices(self, output, sample_weight):
        """Derive resampled weights from ``sample_indices_``."""
        if len(output) != 2:
            raise ValueError(
                f"{self.__class__.__name__} returned an unexpected number of "
                "outputs while propagating sample weights."
            )
        sample_indices = getattr(self, "sample_indices_", None)
        if sample_indices is None:
            raise ValueError(
                f"{self.__class__.__name__} does not expose 'sample_indices_' "
                "and cannot align resampled sample weights."
            )
        sample_indices = np.asarray(sample_indices, dtype=int)
        if sample_indices.shape[0] != output[1].shape[0]:
            raise ValueError(
                f"{self.__class__.__name__}: 'sample_indices_' has "
                f"{sample_indices.shape[0]} entries while the resampled target "
                f"has {output[1].shape[0]} rows."
            )
        return np.asarray(sample_weight, dtype=float)[sample_indices]


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

    def fit_resample(self, X, y, sample_weight=None, class_weight=None, **params):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights associated with each row of ``X`` and ``y``.
            See :meth:`SamplerMixin.fit_resample`.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.

        sample_weight_resampled : array-like of shape (n_samples_new,)
            Only returned when ``sample_weight`` is not ``None``.
        """
        return super().fit_resample(
            X, y, sample_weight=sample_weight, class_weight=class_weight, **params
        )

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

    def fit_resample(self, X, y, sample_weight=None, class_weight=None):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights associated with each row of ``X`` and ``y``.  When
            provided, the callable is expected to return the resampled weights
            as a third output; the weights are otherwise passed through.

            .. versionadded:: 0.15

        class_weight : None, "balanced" or dict, default=None
            Class-level multiplicative factors applied on top of
            ``sample_weight``.

            .. versionadded:: 0.15

        Returns
        -------
        X_resampled : {array-like, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.

        sample_weight_resampled : array-like of shape (n_samples_new,)
            Only returned when ``sample_weight`` is not ``None``.
        """
        self._validate_params()
        arrays_transformer = ArraysTransformer(X, y)

        effective_weight = None
        if sample_weight is not None or class_weight is not None:
            n_samples = getattr(y, "shape", None)
            n_samples = n_samples[0] if n_samples is not None else len(y)
            if sample_weight is not None:
                sample_weight = _as_weight_array(
                    sample_weight,
                    n_samples=n_samples,
                    sampler_name=self.__class__.__name__,
                )
            effective_weight = check_effective_sample_weight(
                sample_weight,
                class_weight,
                np.asarray(y),
                sampler_name=self.__class__.__name__,
            )

        if self.validate:
            check_classification_targets(y)
            X, y, binarize_y = self._check_X_y(X, y, accept_sparse=self.accept_sparse)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        if effective_weight is None:
            output = self._fit_resample(X, y)
        else:
            if self.func is None:
                # The identity function does not resample: keep the weights as
                # they are, row-wise aligned with ``X`` and ``y``.
                output = (X, y, effective_weight)
            else:
                output = self._fit_resample_weighted(X, y, effective_weight)

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

    def _fit_resample_weighted(self, X, y, sample_weight):
        from inspect import Parameter, signature

        func = self.func
        if func is None:
            return X, y, sample_weight
        try:
            accepts_weight = (
                "sample_weight" in signature(func).parameters
                or any(
                    param.kind is Parameter.VAR_KEYWORD
                    for param in signature(func).parameters.values()
                )
            )
        except (TypeError, ValueError):
            accepts_weight = False
        if not accepts_weight:
            raise ValueError(
                "FunctionSampler: 'func' does not accept a 'sample_weight' "
                "argument and cannot propagate weights."
            )
        output = func(
            X, y, sample_weight=sample_weight, **(self.kw_args if self.kw_args else {})
        )
        if len(output) != 3:
            raise ValueError(
                "FunctionSampler: 'sample_weight' was provided but 'func' did not "
                "return resampled weights. Return a 3-tuple "
                "(X_resampled, y_resampled, sample_weight_resampled) or call the "
                "sampler without weights."
            )
        return output
