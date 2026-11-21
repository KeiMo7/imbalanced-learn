.. _common_pitfalls:

=========================================
Common pitfalls and recommended practices
=========================================

This section is a complement to the documentation given
`[here] <https://scikit-learn.org/dev/common_pitfalls.html>`_ in scikit-learn.
Indeed, we will highlight the issue of misusing resampling, leading to a
**data leakage**. Due to this leakage, the performance of a model reported
will be over-optimistic.

Data leakage
============

As mentioned in the scikit-learn documentation, data leakage occurs when
information that would not be available at prediction time is used when
building the model.

In the resampling setting, there is a common pitfall that corresponds to
resample the **entire** dataset before splitting it into a train and a test
partitions. Note that it would be equivalent to resample the train and test
partitions as well.

Such of a processing leads to two issues:

* the model will not be tested on a dataset with class distribution similar
  to the real use-case. Indeed, by resampling the entire dataset, both the
  training and testing set will be potentially balanced while the model should
  be tested on the natural imbalanced dataset to evaluate the potential bias
  of the model;
* the resampling procedure might use information about samples in the dataset
  to either generate or select some of the samples. Therefore, we might use
  information of samples which will be later used as testing samples which
  is the typical data leakage issue.

We will demonstrate the wrong and right ways to do some sampling and emphasize
the tools that one should use, avoiding to fall in the trap.

We will use the adult census dataset. For the sake of simplicity, we will only
use the numerical features. Also, we will make the dataset more imbalanced to
increase the effect of the wrongdoings::

  >>> from sklearn.datasets import fetch_openml
  >>> from imblearn.datasets import make_imbalance
  >>> X, y = fetch_openml(
  ...     data_id=1119, as_frame=True, return_X_y=True
  ... )
  >>> X = X.select_dtypes(include="number")
  >>> X, y = make_imbalance(
  ...     X, y, sampling_strategy={">50K": 300}, random_state=1
  ... )

Let's first check the balancing ratio on this dataset::

  >>> from collections import Counter
  >>> {key: value / len(y) for key, value in Counter(y).items()}
  {'<=50K': 0.988..., '>50K': 0.011...}

To later highlight some of the issue, we will keep aside a left-out set that we
will not use for the evaluation of the model::

  >>> from sklearn.model_selection import train_test_split
  >>> X, X_left_out, y, y_left_out = train_test_split(
  ...     X, y, stratify=y, random_state=0
  ... )

We will use a :class:`sklearn.ensemble.HistGradientBoostingClassifier` as a
baseline classifier. First, we will train and check the performance of this
classifier, without any preprocessing to alleviate the bias toward the majority
class. We evaluate the generalization performance of the classifier via
cross-validation::

  >>> from sklearn.ensemble import HistGradientBoostingClassifier
  >>> from sklearn.model_selection import cross_validate
  >>> model = HistGradientBoostingClassifier(random_state=0)
  >>> cv_results = cross_validate(
  ...     model, X, y, scoring="balanced_accuracy",
  ...     return_train_score=True, return_estimator=True,
  ...     n_jobs=-1
  ... )
  >>> print(
  ...     f"Balanced accuracy mean +/- std. dev.: "
  ...     f"{cv_results['test_score'].mean():.3f} +/- "
  ...     f"{cv_results['test_score'].std():.3f}"
  ... )
  Balanced accuracy mean +/- std. dev.: 0.609 +/- 0.024

We see that the classifier does not give good performance in terms of balanced
accuracy mainly due to the class imbalance issue.

In the cross-validation, we stored the different classifiers of all folds. We
will show that evaluating these classifiers on the left-out data will give
close statistical performance::

  >>> import numpy as np
  >>> from sklearn.metrics import balanced_accuracy_score
  >>> scores = []
  >>> for fold_id, cv_model in enumerate(cv_results["estimator"]):
  ...     scores.append(
  ...         balanced_accuracy_score(
  ...             y_left_out, cv_model.predict(X_left_out)
  ...         )
  ...     )
  >>> print(
  ...     f"Balanced accuracy mean +/- std. dev.: "
  ...     f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
  ... )
  Balanced accuracy mean +/- std. dev.: 0.628 +/- 0.009

Let's now show the **wrong** pattern to apply when it comes to resampling to
alleviate the class imbalance issue. We will use a sampler to balance the
**entire** dataset and check the statistical performance of our classifier via
cross-validation::

  >>> from imblearn.under_sampling import RandomUnderSampler
  >>> sampler = RandomUnderSampler(random_state=0)
  >>> X_resampled, y_resampled = sampler.fit_resample(X, y)
  >>> model = HistGradientBoostingClassifier(random_state=0)
  >>> cv_results = cross_validate(
  ...     model, X_resampled, y_resampled, scoring="balanced_accuracy",
  ...     return_train_score=True, return_estimator=True,
  ...     n_jobs=-1
  ... )
  >>> print(
  ...     f"Balanced accuracy mean +/- std. dev.: "
  ...     f"{cv_results['test_score'].mean():.3f} +/- "
  ...     f"{cv_results['test_score'].std():.3f}"
  ... )
  Balanced accuracy mean +/- std. dev.: 0.724 +/- 0.042

The cross-validation performance looks good, but evaluating the classifiers
on the left-out data shows a different picture::

  >>> scores = []
  >>> for fold_id, cv_model in enumerate(cv_results["estimator"]):
  ...     scores.append(
  ...         balanced_accuracy_score(
  ...             y_left_out, cv_model.predict(X_left_out)
  ...        )
  ...     )
  >>> print(
  ...     f"Balanced accuracy mean +/- std. dev.: "
  ...     f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
  ... )
  Balanced accuracy mean +/- std. dev.: 0.698 +/- 0.014

We see that the performance is now worse than the cross-validated performance.
Indeed, the data leakage gave us too optimistic results due to the reason
stated earlier in this section.

We will now illustrate the correct pattern to use. Indeed, as in scikit-learn,
using a :class:`~imblearn.pipeline.Pipeline` avoids to make any data leakage
because the resampling will be delegated to imbalanced-learn and does not
require any manual steps::

  >>> from imblearn.pipeline import make_pipeline
  >>> model = make_pipeline(
  ...     RandomUnderSampler(random_state=0),
  ...     HistGradientBoostingClassifier(random_state=0)
  ... )
  >>> cv_results = cross_validate(
  ...     model, X, y, scoring="balanced_accuracy",
  ...     return_train_score=True, return_estimator=True,
  ...     n_jobs=-1
  ... )
  >>> print(
  ...     f"Balanced accuracy mean +/- std. dev.: "
  ...     f"{cv_results['test_score'].mean():.3f} +/- "
  ...     f"{cv_results['test_score'].std():.3f}"
  ... )
  Balanced accuracy mean +/- std. dev.: 0.732 +/- 0.019

We observe that we get good statistical performance as well. However, now we
can check the performance of the model from each cross-validation fold to
ensure that we have similar performance::

  >>> scores = []
  >>> for fold_id, cv_model in enumerate(cv_results["estimator"]):
  ...     scores.append(
  ...         balanced_accuracy_score(
  ...             y_left_out, cv_model.predict(X_left_out)
  ...        )
  ...     )
  >>> print(
  ...     f"Balanced accuracy mean +/- std. dev.: "
  ...     f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
  ... )
  Balanced accuracy mean +/- std. dev.: 0.727 +/- 0.008

We see that the statistical performance are very close to the cross-validation
study that we perform, without any sign of over-optimistic results.

Opt-in resampling guards and synthetic-sample provenance
========================================================

.. versionadded:: 0.15

The leakage pattern shown above is only caught if the caller remembers the
correct ordering of *split first, resample second*. The
:mod:`imblearn.safety` module makes the boundary between training data and
held-out data enforceable instead of conventional. Everything in this module
is **opt-in**: samplers used directly behave exactly as before, with no extra
computation.

Guarding against resampling a validation or test set
-----------------------------------------------------

A :class:`~imblearn.safety.ResamplingGuard` records the role of each array.
Arrays are registered **by Python object identity** -- their contents are
neither copied nor hashed -- so the guard stays cheap on million-row datasets.
A :class:`~imblearn.safety.GuardedSampler` wraps any existing sampler and
refuses to run when the input is registered as validation or test data::

  >>> from imblearn.over_sampling import SMOTE
  >>> from imblearn.safety import GuardedSampler, ResamplingGuard
  >>> from imblearn.safety import ResamplingLeakageError
  >>> X_train, X_test, y_train, y_test = train_test_split(
  ...     X, y, stratify=y, random_state=0
  ... )
  >>> guard = ResamplingGuard()
  >>> guard.register_test(X_test, name="X_test")
  >>> safe_smote = GuardedSampler(SMOTE(random_state=42), guard=guard)
  >>> try:
  ...     safe_smote.fit_resample(X_test, y_test)
  ... except ResamplingLeakageError as err:
  ...     print(type(err).__name__)
  ResamplingLeakageError

The error message names the registered array, its role (``"validation"`` or
``"test"``), the sampler and the intercepted step (``fit`` or
``fit_resample``). Arrays that were never registered default to the
``"train"`` role, matching the historical behaviour.

:class:`~imblearn.safety.GuardedSampler` is itself a sampler and can be used
as a step of an :class:`~imblearn.pipeline.Pipeline`, including inside
:func:`~sklearn.model_selection.cross_validate`. The wrapper clones the
wrapped sampler for each fit while sharing the same guard, so every
cross-validation fold is checked independently.

Tracing synthetic samples back to their parents
------------------------------------------------

When resampling runs inside a ``ResamplingGuard.fold()`` context (or through a
:class:`~imblearn.safety.GuardedSampler`), each call stores a
:class:`~imblearn.safety.ResamplingProvenance` record in
``sampler.provenance_`` and accumulates it on ``guard.provenance_records_``::

  >>> guard = ResamplingGuard()
  >>> safe_smote = GuardedSampler(SMOTE(random_state=42), guard=guard)
  >>> with guard.fold():
  ...     X_resampled, y_resampled = safe_smote.fit_resample(X_train, y_train)
  >>> provenance = safe_smote.provenance_
  >>> provenance.support
  'full'
  >>> # output position 0 of the first synthetic block -> two input positions
  >>> provenance.parents_of(X_train.shape[0])
  (..., ...)

The record only contains **sample identifiers** -- integer positions in the
input array -- and never feature values, so sensitive fields cannot leak
through the audit trail. ``ResamplingProvenance.to_dict()`` returns a
JSON-serialisable view for audit logs.

Support depends on what each sampler can report honestly:

* ``"full"``: every synthetic sample names its parent input positions. This
  covers the SMOTE family (``SMOTE``, ``BorderlineSMOTE``, ``SVMSMOTE``,
  ``KMeansSMOTE``, ``SMOTENC``, ``SMOTEN``), where each synthetic point is
  generated between two identified neighbours.
* ``"partial"``: the sampler exposes which input rows were selected through
  ``sample_indices_`` (e.g. ``RandomOverSampler``, ``RandomUnderSampler``)
  but cannot explain a generation rule.
* ``"not_supported"``: no lineage is available (e.g. ``ADASYN`` and custom
  :class:`~imblearn.FunctionSampler` callables). The record says so
  explicitly instead of inventing parents; ``parent_sample_ids`` is then
  ``None``.

Cross-validation fold isolation
-------------------------------

Each ``guard.fold()`` context owns a private provenance sink, backed by
:class:`contextvars.ContextVar`, so synthetic samples and lineage produced in
one fold can never be carried into the next one. Records are published on the
guard only when the fold exits without an error; if sampling is interrupted,
no half-built record and no partial output array is kept.

Checking target ratios before and after resampling
--------------------------------------------------

:func:`~imblearn.safety.check_sampling_ratio` resolves a
``sampling_strategy`` against the current class counts without touching any
array, and returns a :class:`~imblearn.safety.SamplingRatioReport`. Two
unreachable cases are distinguished:

* ``reason="insufficient_samples"``: the target asks for more rows than
  exist (e.g. under-sampling a class of 180 rows down to 5000);
* ``reason="parameter_conflict"``: contradictory parameters (e.g. an
  over-sampling target smaller than the current class size).

Calling ``record_actual(y_resampled)`` on the report fills the observed
counts and the per-class deviation (``actual - target``), so the achieved
ratio can be checked against the requested one.

Known limitations
-----------------

* Role registration relies on object identity. Passing a *different* array
  object that happens to hold the same rows (a copy, a view materialised by a
  transformer) is treated as a new, unregistered array. Register the exact
  object you hand to the sampler, or place the sampler inside a pipeline so
  the guarded step always receives the training partition.
* With joblib-backed parallelism (``n_jobs > 1``), worker processes do not
  share the parent's guard registry; run the guarded pipeline with
  ``n_jobs=1`` or register arrays inside each worker. Sequential
  cross-validation and single-process pipelines are fully supported.
* Provenance positions refer to the array passed to ``fit_resample``. When a
  transformer precedes the sampler in a pipeline, the identifiers index the
  transformed training data of that fold, not the raw source rows.
* The guard does not replace the :class:`~imblearn.pipeline.Pipeline`
  pattern; it enforces it. Samplers still never modify their input arrays and
  still return new arrays, whether guarded or not.

Serialization and compatibility boundary
----------------------------------------

The safety layer is additive and stores no state on the existing samplers:

* Pickles of samplers, pipelines and models written by previous
  imbalanced-learn versions load unchanged, and their ``fit_resample``
  outputs are bit-for-bit identical. Old artefacts simply have no
  ``provenance_`` attribute; guard them with a
  :class:`~imblearn.safety.GuardedSampler` at call time to obtain lineage for
  new resampling runs.
* :class:`~imblearn.safety.ResamplingProvenance` and
  :class:`~imblearn.safety.SamplingRatioReport` are plain dataclasses that can
  be serialised through their ``to_dict()`` method. They are not required to
  unpickle an old model and should be stored alongside (not inside) legacy
  artefacts.
