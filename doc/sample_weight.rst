.. _sample_weight:

==========================================
Sample weights in resampling
==========================================

Resampling changes the number of rows of a dataset.  Historically,
:meth:`~imblearn.base.BaseSampler.fit_resample` only returned ``X`` and ``y``:
the sample weights passed to a downstream trainer were kept as they were and
therefore no longer aligned row-wise with the resampled data.  In practice the
weights were dropped at the sampling entry point.

This section documents how weights are handled when they are passed to
``fit_resample``.

Passing weights
---------------

All samplers accept a ``sample_weight`` keyword argument::

    X_res, y_res, w_res = RandomUnderSampler(random_state=0).fit_resample(
        X, y, sample_weight=sample_weight
    )

When ``sample_weight`` is ``None`` - the historical call - the sampler returns
two outputs and behaves exactly as before, including the sequence of random
numbers it draws.  When weights are provided, a third output is returned with
the resampled weights, aligned row-wise with ``X_res`` and ``y_res``.

``class_weight`` can be passed as well.  The effective per-row weight is::

    effective_weight = sample_weight * class_weight[y]

with ``class_weight`` accepting ``None``, ``"balanced"`` or a mapping.  The
sampling strategy still decides *how many* rows are produced; the weights only
carry the relative importance of the rows.

Weight semantics per sampling family
------------------------------------

Under-sampling and cleaning
    Retained rows keep the weight they had before resampling.  The weights of
    discarded rows do not participate in any later statistic.

Over-sampling
    The original rows keep their weight.  The weight of a generated row is
    defined by the ``weight_strategy`` parameter:

    ``"inherit"`` (default)
        The generated row copies the weight of the base sample it is generated
        from.  If the same sample is drawn several times, each copy keeps the
        same weight.

    ``"nearest"``
        The weight is linearly interpolated between the base sample and the
        selected nearest neighbour, using the same step as the features.  This
        mimics weighting each endpoint by the same amount as the generated
        point.

    ``"cluster"``
        Every generated row receives the average weight of the pool it is
        generated from (the over-sampled class for SMOTE).

    callable
        ``callable(base_weight, neighbor_weight, steps)`` returning an array of
        weights for the generated rows.

Combining over- and under-sampling
    The stages form an explicit, ordered chain: over-sampling generates the
    weights of the new rows, then the cleaning or under-sampling stage keeps a
    subset of them.  This is what a :class:`~imblearn.pipeline.Pipeline` does
    when sampling steps are chained.

Zero-weight rows
----------------

A row with a zero weight cannot contribute to the resampling: it is removed
before any neighbour search, before any synthetic sample generation and before
any under-sampling decision.  This makes ``sample_weight = 0`` a reliable way
to exclude a row from resampling while keeping it available for other
pipelines.

Invalid weights
---------------

The following inputs are rejected *before* any sampling work starts, so that a
sampling chain never fails half-way through:

- negative weights;
- missing (``NaN``) or infinite weights;
- a length that does not match the number of rows of ``X`` and ``y``;
- weights that are all zero, or a class whose rows all have a zero weight;
- an unsupported ``weight_strategy``.

Error messages only report aggregate information - row counts, weight range and
the name of the sampler - and never the content of an individual sample.

Pipeline usage
--------------

A :class:`~imblearn.pipeline.Pipeline` threads the weights through its sampling
steps.  With the historical, non-routed API the weights are given to the
sampler step::

    pipe = make_pipeline(
        SMOTE(random_state=0),
        RandomUnderSampler(random_state=0),
        LogisticRegression(),
    )
    pipe.fit(X, y, smote__sample_weight=sample_weight)

Each sampler resamples the weights and the final estimator receives the
resampled weights, so ``X``, ``y`` and ``sample_weight`` stay aligned.  With
metadata routing enabled, request the parameter on the steps instead::

    sampler.set_fit_resample_request(sample_weight=True)
    clf.set_fit_request(sample_weight=True)
    pipeline.fit(X, y, sample_weight=sample_weight)
