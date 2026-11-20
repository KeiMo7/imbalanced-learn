.. _memory_management:

=============================
Memory management in samplers
=============================

.. currentmodule:: imblearn

Resampling a large dataset can silently multiply its memory footprint: some
samplers copy the whole input before building the resampled output, while
others only allocate the final arrays. Starting from version 0.15 every
sampler exposes a *declarative memory contract* and the two index-based
samplers (:class:`~imblearn.over_sampling.RandomOverSampler` and
:class:`~imblearn.under_sampling.RandomUnderSampler`) offer a bounded-memory
*chunked* path. All new behaviour is opt-in: with the switches left at their
default values, results and memory behaviour are identical to previous
versions.

Reporting the memory characteristics
====================================

Every sampler implements ``memory_profile(X, y)``, which returns a
:class:`~imblearn.utils._memory.MemoryProfile` describing how the sampler
treats the given input. The method only inspects array metadata; it never
copies the data.

    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import RandomOverSampler
    >>> X, y = make_classification(
    ...     n_classes=2, weights=[0.1, 0.9], n_informative=3,
    ...     n_redundant=1, flip_y=0, n_features=10,
    ...     n_clusters_per_class=1, n_samples=1000, random_state=0,
    ... )
    >>> profile = RandomOverSampler(random_state=0).memory_profile(X, y)
    >>> profile.copies_input
    True
    >>> profile.chunked_supported
    True
    >>> profile.scaling
    'over-sampling: O((1 + alpha) * n * p) where alpha is the fraction of added minority rows'

The profile reports:

* ``copies_input``: whether the regular implementation duplicates the full
  input array;
* ``chunked_supported``: whether the bounded-memory path accepts this input
  type (dense fixed-dtype arrays);
* ``peak_extra_bytes_full`` and ``peak_extra_bytes_chunked``: estimated
  *extra* bytes on top of the input, for each path;
* ``output_bytes``: the unavoidable size of the resampled arrays;
* ``temp_extra_bytes_chunked``: the temporary, non-output scratch space of
  the chunked path. It depends on ``chunk_size`` and the row width, not on
  the total number of rows, so it stays constant as the dataset grows;
* ``scaling``: the relation between the sampling ratio and the extra memory;
* ``notes``: the concrete copy points of the implementation.

Memory characteristics of the samplers
======================================

The table below summarises the historical behaviour. "Copies input" refers to
the regular path (``chunk_size=None``).

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Sampler
     - copies input
     - chunked path
     - Main extra allocations
   * - :class:`~over_sampling.RandomOverSampler`
     - yes (bootstrap)
     - yes (dense only)
     - ``X.copy()`` + per-class copies + ``vstack``
   * - :class:`~under_sampling.RandomUnderSampler`
     - no
     - yes (dense only)
     - index arrays + one output array
   * - :class:`~over_sampling.SMOTE` and variants
     - yes
     - no
     - ``X.copy()`` + neighbour graph + synthetic rows
   * - :class:`~over_sampling.ADASYN`
     - yes
     - no
     - ``X.copy()`` + neighbour graph + synthetic rows
   * - :class:`~under_sampling.ClusterCentroids`
     - yes
     - no
     - clustering state + centroids
   * - Cleaning samplers (ENN, Tomek, ...)
     - no
     - no
     - neighbour graph + index mask
   * - :class:`~combine.SMOTEENN` / :class:`~combine.SMOTETomek`
     - yes
     - no
     - sum of the two embedded stages
   * - :class:`~FunctionSampler`
     - depends on func
     - no
     - whatever the wrapped function allocates

For samplers without a chunked path, ``memory_profile`` still returns a
conservative generic profile so that memory estimates remain available
pipeline-wide.

Bounded-memory chunked resampling
=================================

Passing an integer ``chunk_size`` pre-allocates the output arrays at their
final size and fills them chunk by chunk:

    >>> from imblearn.under_sampling import RandomUnderSampler
    >>> regular = RandomUnderSampler(random_state=0)
    >>> chunked = RandomUnderSampler(random_state=0, chunk_size=10_000)
    >>> X_reg, y_reg = regular.fit_resample(X, y)
    >>> X_chk, y_chk = chunked.fit_resample(X, y)
    >>> np.array_equal(X_reg, X_chk) and np.array_equal(y_reg, y_chk)
    True

The selection indices are always computed globally, using the exact same
random number generator calls as the regular path. The chunk size therefore
only changes *where* the output is materialised, never the draws:

* results are bit-identical to the regular path for a given
  ``random_state``, for every positive ``chunk_size``;
* floating-point precision is unchanged — chunking performs plain row copies,
  no arithmetic is reordered;
* the temporary footprint is bounded by ``chunk_size * n_features`` instead
  of growing with the input size; the output allocation itself and the
  ``O(n)`` integer index arrays remain.

Where the win is expected to come from differs between the two samplers:

* :class:`~over_sampling.RandomOverSampler` duplicates the whole input in its
  regular path, so chunking removes that copy and the per-class accumulators:
  the peak drops sharply (see the benchmark below).
* :class:`~under_sampling.RandomUnderSampler` never copies the input and
  already shrinks the data, so chunking cannot reduce its peak below the
  output size. Its benefit is bounding each temporary fancy-indexed copy to
  ``chunk_size`` rows, giving a predictable, flat temporary footprint
  (``temp_extra_bytes_chunked``) instead of one large intermediate.

Restrictions: the chunked path accepts dense in-memory arrays with a fixed
dtype only. Sparse matrices, heterogeneous object arrays and the smoothed
bootstrap of ``RandomOverSampler`` (``shrinkage`` not ``None``) raise an
explicit ``ValueError`` and keep using the regular path.

Measured time and memory
========================

The benchmark below uses a dense ``float64`` array of 2,000,000 rows and 20
features (305 MiB), a binary 90/10 target and a fixed ``random_state``. Each
configuration runs in an isolated subprocess and the table reports the median
of three runs; "sampler extra" is the process peak RSS minus the data-only
baseline. The script and machine-readable results live in
``maint_tools/memory_benchmark/``.

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 10 14 16 16

   * - sampler
     - mode
     - chunk_size
     - time (s)
     - peak RSS (MiB)
     - sampler extra (MiB)
     - output shape
   * - RandomOverSampler
     - full
     - None
     - 0.887
     - 1685.2
     - 586.6
     - (3600000, 20)
   * - RandomOverSampler
     - chunked
     - 100000
     - 0.498
     - 1148.1
     - 49.5
     - (3600000, 20)
   * - RandomOverSampler
     - chunked
     - 10000
     - 0.475
     - 1135.8
     - 37.2
     - (3600000, 20)
   * - RandomUnderSampler
     - full
     - None
     - 0.194
     - 1125.2
     - 26.5
     - (400000, 20)
   * - RandomUnderSampler
     - chunked
     - 100000
     - 0.191
     - 1124.9
     - 26.3
     - (400000, 20)
   * - RandomUnderSampler
     - chunked
     - 10000
     - 0.185
     - 1125.2
     - 26.5
     - (400000, 20)

Readings:

* over-sampling is where the feature pays off: the chunked path removes the
  duplicated input and the per-class copies, cutting the sampler's extra
  footprint from 586.6 MiB to 37.2 MiB (about 16x) while also running faster
  because the large ``vstack`` and repeated array growth disappear;
* under-sampling confirms the profile contract above: its regular path does
  not copy the input, so the peak is essentially unchanged and the time is
  statistically identical;
* no extra process, thread or synchronisation point is introduced — the
  chunked path is a plain in-process loop over pre-allocated arrays.

Memory guard and graceful degradation
=====================================

Setting ``memory_guard=True`` makes the sampler estimate its peak extra
memory before resampling:

* when the input has more than
  :data:`~imblearn.utils._memory.LARGE_DATA_THRESHOLD` rows and no
  ``chunk_size`` is set, a :class:`UserWarning` is emitted with the estimate;
* when the estimate exceeds 90% of the currently available memory, the
  sampler *degrades*: it returns the original ``X`` and ``y`` unchanged,
  emits a :class:`UserWarning` with actionable advice (set a chunk size,
  reduce the ratio, free memory) and sets ``degraded_`` to ``True``;
* if a :class:`MemoryError` is nevertheless raised mid-run, the partially
  allocated arrays are released and the same degradation applies.

    >>> import warnings
    >>> sampler = RandomUnderSampler(random_state=0, memory_guard=True)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     X_out, y_out = sampler.fit_resample(X, y)
    >>> getattr(sampler, "degraded_", False)
    False

Without the guard (the default), memory errors propagate exactly as they did
before.

Combining samplers
==================

When samplers are chained (for instance inside a
:class:`~imblearn.pipeline.Pipeline`), each stage only holds its own input,
its indices and its output: the combined estimate is the *sum* of the
per-stage profiles, obtainable by calling ``memory_profile`` on each stage
with the output of the previous one. No stage introduces a hidden
multiplicative copy, so composing chunked samplers cannot turn a bounded
footprint into an order-of-magnitude blow-up.

Parallelism, processes and determinism
======================================

The chunked path is single-process and single-threaded: it introduces no
worker pool, no inter-process communication and no synchronisation point.
Consequently the only source of randomness is the sampler's
``random_state``, and results are deterministic for every chunk size.
Samplers that already support ``n_jobs`` (for instance the ensemble methods)
are untouched by this feature; their documented process/seed semantics keep
applying, and determinism still requires an explicit ``random_state``.

Serialization and compatibility boundaries
==========================================

* Pickles of samplers created with previous imbalanced-learn versions load
  unchanged: the new parameters are keyword-only with historical defaults
  (``chunk_size=None``, ``memory_guard=False``), so unpickled estimators run
  the regular path and reproduce their old results. Old artifacts need no
  migration.
* Estimators pickled with the new parameters can only be loaded by an
  imbalanced-learn version providing them (0.15 or later).
* ``sample_indices_`` keeps its historical shape, dtype and ordering; the
  chunked path does not introduce new fitted attributes besides
  ``degraded_`` / ``degradation_message_``, which only exist when the guard
  is enabled.

Known limitations
=================

* Only :class:`~over_sampling.RandomOverSampler` (plain bootstrap) and
  :class:`~under_sampling.RandomUnderSampler` implement the chunked path.
  Neighbour-based samplers (SMOTE family, ENN, Tomek, ...) need global
  neighbour structures and cannot be row-chunked without changing their
  numerical results; they still report their profile via
  ``memory_profile``.
* Sparse and object-dtype inputs are rejected by the chunked path.
* The pre-flight estimate relies on :mod:`psutil` when installed and falls
  back to ``/proc/meminfo`` on Linux; when neither is available the
  pre-flight check is skipped (mid-run ``MemoryError`` handling still works).
  No new mandatory dependency is introduced.
