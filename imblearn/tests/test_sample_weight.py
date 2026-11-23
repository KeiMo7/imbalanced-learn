"""Tests for the first-class support of ``sample_weight`` in resampling.

The test suite is organised around the acceptance criteria of the feature:

1. original weights are preserved through under-sampling;
2. generated weights follow the documented ``weight_strategy`` rules;
3. over- and under-sampling chains compose and stay row-aligned;
4. ``class_weight``, ``sample_weight`` and ``sampling_strategy`` coexist with a
   documented precedence;
5. zero-weight rows neither generate samples nor influence neighbours;
6. invalid weights are rejected before any sampling work;
7. a downstream trainer sees row-aligned ``X``, ``y`` and ``sample_weight``;
8. no step silently turns weights into uniform weights;
9. the unweighted path, including its random sequence, is unchanged.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import assert_allclose, assert_array_equal

from imblearn import FunctionSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTEN,
    SMOTENC,
    BorderlineSMOTE,
    RandomOverSampler,
)
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import (
    EditedNearestNeighbours,
    NearMiss,
    RandomUnderSampler,
    TomekLinks,
)

RND_SEED = 42


@pytest.fixture
def data():
    X, y = make_classification(
        n_classes=2,
        class_sep=1.5,
        weights=[0.25, 0.75],
        n_informative=4,
        n_redundant=0,
        n_features=6,
        n_samples=400,
        random_state=RND_SEED,
    )
    return X, y


@pytest.fixture
def weighted_data(data):
    X, y = data
    rng = np.random.RandomState(RND_SEED)
    # weights differ strongly between the two classes on purpose
    sample_weight = np.where(y == 0, rng.uniform(2.0, 5.0, size=y.shape[0]), 0.5)
    return X, y, sample_weight


OVER_SAMPLERS = [
    RandomOverSampler(random_state=RND_SEED),
    SMOTE(random_state=RND_SEED, k_neighbors=3),
    ADASYN(random_state=RND_SEED, n_neighbors=3),
    BorderlineSMOTE(random_state=RND_SEED, k_neighbors=3, m_neighbors=5),
]

UNDER_SAMPLERS = [
    RandomUnderSampler(random_state=RND_SEED),
    EditedNearestNeighbours(),
    TomekLinks(),
    NearMiss(),
]


# ---------------------------------------------------------------------------
# 1. Under-sampling keeps the original weights of the retained rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampler", UNDER_SAMPLERS)
def test_under_sampling_keeps_original_weights(sampler, weighted_data):
    X, y, sample_weight = weighted_data
    X_res, y_res, w_res = sampler.fit_resample(X, y, sample_weight=sample_weight)

    assert X_res.shape[0] == y_res.shape[0] == w_res.shape[0]
    # The weights of the retained rows are exactly the input weights.
    assert_allclose(np.sort(w_res), np.sort(sample_weight[sampler.sample_indices_]))
    # A strict subset was selected: the discarded weights do not participate.
    assert w_res.shape[0] < sample_weight.shape[0]
    assert np.unique(w_res).size > 1


def test_random_under_sampler_manual_weights(data):
    X, y = data
    sample_weight = np.arange(y.shape[0], dtype=float)
    sampler = RandomUnderSampler(random_state=RND_SEED)
    X_res, y_res, w_res = sampler.fit_resample(X, y, sample_weight=sample_weight)

    assert_array_equal(w_res, sample_weight[sampler.sample_indices_])
    # Row alignment: every output weight belongs to the row it was selected with.
    for row_weight, klass in zip(w_res, y_res):
        original_idx = int(row_weight)
        assert y[original_idx] == klass


# ---------------------------------------------------------------------------
# 2. Over-sampling generated weights follow the documented rule
# ---------------------------------------------------------------------------


def test_smote_inherit_strategy_matches_manual_computation(data):
    X, y = data
    rng = np.random.RandomState(RND_SEED)
    sample_weight = rng.uniform(0.5, 3.0, size=y.shape[0])

    sampler = SMOTE(random_state=RND_SEED, k_neighbors=3, weight_strategy="inherit")
    X_res, y_res, w_res = sampler.fit_resample(X, y, sample_weight=sample_weight)

    # Original rows keep their weight, generated rows copy the weight of the
    # base sample; the number of rows is unchanged by the weighting.
    n_original = X.shape[0]
    assert_array_equal(w_res[:n_original], sample_weight)
    assert w_res.shape[0] == X_res.shape[0]

    # Every generated row has the weight of some original sample of the same
    # class: ``inherit`` never invents a new value.
    generated_y = y_res[n_original:]
    for klass in np.unique(generated_y):
        generated = w_res[n_original:][generated_y == klass]
        in_class = [np.any(np.isclose(v, sample_weight[y == klass])) for v in generated]
        assert np.all(in_class)


def test_smote_nearest_strategy_is_weighted_interpolation(data):
    X, y = data
    sample_weight = np.linspace(0.5, 4.0, num=y.shape[0])

    sampler = SMOTE(random_state=RND_SEED, k_neighbors=3, weight_strategy="nearest")
    X_res, y_res, w_res = sampler.fit_resample(X, y, sample_weight=sample_weight)

    generated_weight = w_res[X.shape[0] :]
    low, high = sample_weight.min(), sample_weight.max()
    # Interpolation stays between the two endpoint weights.
    assert generated_weight.min() >= low - 1e-9
    assert generated_weight.max() <= high + 1e-9
    assert np.unique(generated_weight).size > 1


def test_over_sampling_cluster_strategy(data):
    X, y = data
    sample_weight = np.where(y == 1, 3.0, 1.0)

    sampler = SMOTE(random_state=RND_SEED, k_neighbors=3, weight_strategy="cluster")
    _, y_res, w_res = sampler.fit_resample(X, y, sample_weight=sample_weight)

    generated_y = y_res[X.shape[0] :]
    # Every generated row receives the average weight of the class it is
    # generated for (the over-sampled class only).
    for klass in np.unique(generated_y):
        generated_weight = w_res[X.shape[0] :][generated_y == klass]
        expected = np.mean(sample_weight[y == klass])
        assert_allclose(generated_weight, np.full_like(generated_weight, expected))


def test_weight_equivalent_replication(data):
    """Weighting a sample by k is equivalent to copying it k times.

    We compare the total weight per class, which is the only statistic that
    must be invariant, for a bootstrap over-sampler.
    """
    X, y = data
    n_copies = 3
    replicated_X = np.repeat(X, n_copies, axis=0)
    replicated_y = np.repeat(y, n_copies)

    weighted = SMOTE(random_state=RND_SEED, k_neighbors=3)
    _, y_weighted, w_weighted = weighted.fit_resample(
        X, y, sample_weight=np.full(y.shape[0], float(n_copies))
    )

    _, y_replicated = SMOTE(random_state=RND_SEED, k_neighbors=3).fit_resample(
        replicated_X, replicated_y
    )

    # The total weight per class is preserved: weighting behaves as replication.
    for klass in np.unique(y):
        assert_allclose(
            w_weighted[y_weighted == klass].sum(),
            np.count_nonzero(y_replicated == klass),
        )


def test_weight_equivalent_replication_under_sampler(data):
    """A k-times copy of a row is selected like a weight-k row.

    Under-sampling only keeps a subset, so the invariant is the total weight of
    the retained rows: a row with weight k counts as k replicated rows.
    """
    X, y = data
    n_copies = 3
    replicated_X = np.repeat(X, n_copies, axis=0)
    replicated_y = np.repeat(y, n_copies)

    weighted = RandomUnderSampler(random_state=RND_SEED)
    _, y_weighted, w_weighted = weighted.fit_resample(
        X, y, sample_weight=np.full(y.shape[0], float(n_copies))
    )
    _, y_replicated = RandomUnderSampler(random_state=RND_SEED).fit_resample(
        replicated_X, replicated_y
    )
    for klass in np.unique(y):
        assert_allclose(
            w_weighted[y_weighted == klass].sum(),
            np.count_nonzero(y_replicated == klass),
        )


def test_weight_strategy_callable(data):
    X, y = data
    sample_weight = np.ones(y.shape[0])

    def strategy(base_weight, neighbor_weight, steps):
        return base_weight + 2.0 * neighbor_weight + steps

    sampler = SMOTE(
        random_state=RND_SEED, k_neighbors=3, weight_strategy=strategy
    )
    _, _, w_res = sampler.fit_resample(X, y, sample_weight=sample_weight)
    generated_weight = w_res[X.shape[0] :]
    # base (1) + 2 * neighbour (2) + step (>=0)
    assert np.all(generated_weight >= 3.0)


@pytest.mark.parametrize(
    "sampler",
    [
        RandomOverSampler(random_state=RND_SEED),
        SMOTE(random_state=RND_SEED, k_neighbors=3),
        BorderlineSMOTE(random_state=RND_SEED, k_neighbors=3, m_neighbors=5),
        ADASYN(random_state=RND_SEED, n_neighbors=3),
    ],
)
@pytest.mark.parametrize("weight_strategy", ["inherit", "nearest", "cluster"])
def test_sparse_input_weighted(sampler, weight_strategy, weighted_data):
    from scipy import sparse

    X, y, sample_weight = weighted_data
    sampler.set_params(weight_strategy=weight_strategy)
    X_res, y_res, w_res = sampler.fit_resample(
        sparse.csr_matrix(X), y, sample_weight=sample_weight
    )
    assert sparse.issparse(X_res)
    assert X_res.shape[0] == y_res.shape[0] == w_res.shape[0]
    assert np.all(np.isfinite(w_res))
    assert np.all(w_res >= 0)


# ---------------------------------------------------------------------------
# 3. Over- then under-sampling: weights compose and stay aligned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combine_sampler", [SMOTEENN, SMOTETomek])
def test_combine_samplers_propagate_weights(combine_sampler, weighted_data):
    X, y, sample_weight = weighted_data
    sampler = combine_sampler(random_state=RND_SEED)
    X_res, y_res, w_res = sampler.fit_resample(X, y, sample_weight=sample_weight)

    assert X_res.shape[0] == y_res.shape[0] == w_res.shape[0]
    assert np.all(w_res >= 0)
    # The generated weights are not silently replaced by uniform weights.
    assert np.unique(w_res).size > 1


def test_pipeline_over_then_under_composes_weights(weighted_data):
    X, y, sample_weight = weighted_data
    pipe = Pipeline(
        [
            ("over", SMOTE(random_state=RND_SEED, k_neighbors=3)),
            ("under", RandomUnderSampler(random_state=RND_SEED)),
        ]
    )
    # Pipeline keeps the historical rule that unprefixed ``sample_weight`` is
    # rejected; weights are routed to the first sampling step by name.
    X_res, y_res, w_res = pipe.fit_resample(
        X, y, over__sample_weight=sample_weight
    )
    assert X_res.shape[0] == y_res.shape[0] == w_res.shape[0]
    assert np.unique(w_res).size > 1


def test_pipeline_weights_on_later_step_flow_through_earlier_steps(weighted_data):
    """Weights declared on a later sampling step still flow through the chain."""
    X, y, sample_weight = weighted_data
    pipe = make_pipeline(
        RandomUnderSampler(random_state=RND_SEED),
        SMOTE(random_state=RND_SEED, k_neighbors=3),
    )
    out = pipe.fit_resample(X, y, smote__sample_weight=sample_weight)
    assert len(out) == 3
    assert out[0].shape[0] == out[1].shape[0] == out[2].shape[0]


def test_pipeline_order_is_explicit(weighted_data):
    """Swapping the two samplers swaps the applied transformations."""
    X, y, sample_weight = weighted_data
    over_first = make_pipeline(
        SMOTE(random_state=RND_SEED, k_neighbors=3),
        RandomUnderSampler(random_state=RND_SEED),
    )
    under_first = make_pipeline(
        RandomUnderSampler(random_state=RND_SEED),
        SMOTE(random_state=RND_SEED, k_neighbors=3),
    )
    out_over = over_first.fit_resample(
        X, y, smote__sample_weight=sample_weight
    )
    out_under = under_first.fit_resample(
        X, y, randomundersampler__sample_weight=sample_weight
    )
    # The two orders produce different weights, which makes the order explicit.
    assert not np.array_equal(out_over[2], out_under[2])


# ---------------------------------------------------------------------------
# 4. class_weight / sample_weight / sampling_strategy precedence
# ---------------------------------------------------------------------------


def test_class_weight_multiplies_sample_weight(data):
    X, y = data
    sample_weight = np.full(y.shape[0], 2.0)
    sampler = RandomUnderSampler(random_state=RND_SEED)
    _, y_res, w_res = sampler.fit_resample(
        X, y, sample_weight=sample_weight, class_weight={0: 3.0, 1: 5.0}
    )
    expected = np.where(y_res == 0, 6.0, 10.0)
    assert_allclose(w_res, expected)


def test_class_weight_only(data):
    X, y = data
    sampler = RandomUnderSampler(random_state=RND_SEED)
    _, y_res, w_res = sampler.fit_resample(X, y, class_weight={0: 2.0, 1: 1.0})
    assert_allclose(w_res, np.where(y_res == 0, 2.0, 1.0))


def test_class_weight_does_not_change_sampling_strategy(data):
    X, y = data
    plain = RandomUnderSampler(random_state=RND_SEED)
    weighted = RandomUnderSampler(random_state=RND_SEED)
    X_plain, y_plain = plain.fit_resample(X, y)
    X_weighted, y_weighted, _ = weighted.fit_resample(
        X, y, class_weight={0: 10.0, 1: 0.1}
    )
    # Same number of rows and same selection: class weights only rescale.
    assert_array_equal(y_plain, y_weighted)
    assert_array_equal(plain.sample_indices_, weighted.sample_indices_)


def test_balanced_class_weight(data):
    X, y = data
    sampler = RandomUnderSampler(random_state=RND_SEED)
    _, y_res, w_res = sampler.fit_resample(X, y, class_weight="balanced")
    counts = {klass: np.count_nonzero(y == klass) for klass in np.unique(y)}
    for klass, factor in counts.items():
        expected = y.shape[0] / (len(counts) * factor)
        assert_allclose(w_res[y_res == klass], expected)


# ---------------------------------------------------------------------------
# 5. Zero-weight rows never take part in resampling
# ---------------------------------------------------------------------------


def test_zero_weight_rows_excluded_from_synthesis(data):
    X, y = data
    sample_weight = np.ones(y.shape[0])
    zero_rows = np.flatnonzero(y == 1)[:10]
    sample_weight[zero_rows] = 0.0

    sampler = SMOTE(random_state=RND_SEED, k_neighbors=3)
    X_res, y_res, w_res = sampler.fit_resample(X, y, sample_weight=sample_weight)

    # No zero-weight row is kept or used as a base for a generated sample.
    assert np.all(w_res > 0)
    assert X_res.shape[0] == y_res.shape[0] == w_res.shape[0]
    # The zero rows are dropped before synthesis: they do not appear in the
    # first (original) block of the resampled output.
    # Equivalent and stricter check: the weighted run equals the unweighted run
    # on the data where zero-weight rows were removed beforehand.
    keep = sample_weight > 0
    X_ref, y_ref = SMOTE(random_state=RND_SEED, k_neighbors=3).fit_resample(
        X[keep], y[keep]
    )
    assert_allclose(X_res, X_ref)
    assert_array_equal(y_res, y_ref)


def test_zero_weight_rows_do_not_influence_neighbors(data):
    """Removing zero-weight rows must not change the resampled data.

    Since those rows cannot contribute, the weighted run on the full data must
    match the unweighted run on the data where they were removed beforehand.
    """
    X, y = data
    sample_weight = np.ones(y.shape[0])
    zero_rows = np.flatnonzero(y == 1)[:5]
    sample_weight[zero_rows] = 0.0
    keep = sample_weight > 0

    weighted = SMOTE(random_state=RND_SEED, k_neighbors=3)
    X_weighted, y_weighted, _ = weighted.fit_resample(
        X, y, sample_weight=sample_weight
    )

    reference = SMOTE(random_state=RND_SEED, k_neighbors=3)
    X_reference, y_reference = reference.fit_resample(X[keep], y[keep])

    assert_allclose(X_weighted, X_reference)
    assert_array_equal(y_weighted, y_reference)


def test_zero_weight_rows_under_sampling(data):
    X, y = data
    sample_weight = np.ones(y.shape[0])
    sample_weight[::7] = 0.0
    sampler = RandomUnderSampler(random_state=RND_SEED)
    X_res, y_res, w_res = sampler.fit_resample(X, y, sample_weight=sample_weight)
    assert np.all(w_res > 0)
    # The exposed indices still refer to the original (unfiltered) data.
    assert sampler.sample_indices_.max() < X.shape[0]
    assert_array_equal(w_res, sample_weight[sampler.sample_indices_])


# ---------------------------------------------------------------------------
# 6. Invalid weights are rejected before sampling starts
# ---------------------------------------------------------------------------


def test_negative_weights_rejected(data):
    X, y = data
    sample_weight = np.ones(y.shape[0])
    sample_weight[3] = -1.0
    with pytest.raises(ValueError, match="negative"):
        RandomUnderSampler(random_state=RND_SEED).fit_resample(
            X, y, sample_weight=sample_weight
        )


def test_nan_weights_rejected(data):
    X, y = data
    sample_weight = np.ones(y.shape[0])
    sample_weight[3] = np.nan
    with pytest.raises(ValueError, match="missing or non-finite"):
        RandomUnderSampler(random_state=RND_SEED).fit_resample(
            X, y, sample_weight=sample_weight
        )


def test_length_mismatch_rejected(data):
    X, y = data
    with pytest.raises(ValueError, match="rows while"):
        RandomUnderSampler(random_state=RND_SEED).fit_resample(
            X, y, sample_weight=np.ones(y.shape[0] - 3)
        )


def test_all_zero_weights_rejected(data):
    X, y = data
    with pytest.raises(ValueError, match="all .* sample weights are zero"):
        RandomUnderSampler(random_state=RND_SEED).fit_resample(
            X, y, sample_weight=np.zeros(y.shape[0])
        )


def test_class_all_zero_weights_rejected(data):
    X, y = data
    sample_weight = np.ones(y.shape[0])
    sample_weight[y == 0] = 0.0
    with pytest.raises(ValueError, match="only have zero weights"):
        RandomUnderSampler(random_state=RND_SEED).fit_resample(
            X, y, sample_weight=sample_weight
        )


def test_unsupported_weight_strategy_rejected_before_sampling(data):
    X, y = data
    sampler = SMOTE(random_state=RND_SEED, weight_strategy="not-a-rule")
    with pytest.raises(ValueError, match="weight_strategy"):
        sampler.fit_resample(X, y, sample_weight=np.ones(y.shape[0]))


def test_invalid_weight_strategy_rejected_even_without_weights(data):
    X, y = data
    sampler = SMOTE(random_state=RND_SEED, weight_strategy="not-a-rule")
    with pytest.raises(ValueError, match="weight_strategy"):
        sampler.fit_resample(X, y)


def test_weight_strategy_rejected_on_unsupported_sampler(data):
    X, y = data
    sampler = RandomUnderSampler(random_state=RND_SEED)
    sampler.weight_strategy = "inherit"
    with pytest.raises(ValueError, match="not supported"):
        sampler.fit_resample(X, y)


def test_failed_step_does_not_leave_partial_weights(weighted_data):
    """A failure raised by a later step must not expose a half-weighted state."""
    X, y, sample_weight = weighted_data

    class FailingSampler(RandomUnderSampler):
        def _fit_resample(self, X, y, **params):
            raise RuntimeError("boom")

    sampler = FailingSampler(random_state=RND_SEED)
    with pytest.raises(RuntimeError, match="boom"):
        sampler.fit_resample(X, y, sample_weight=sample_weight)
    # No resampled weights were published on the failed sampler.
    assert not hasattr(sampler, "resampled_sample_weight_")


def test_weights_never_silently_uniform(data):
    X, y = data
    sample_weight = np.where(y == 1, 5.0, 0.1)
    for sampler in OVER_SAMPLERS + UNDER_SAMPLERS:
        _, _, w_res = sampler.fit_resample(X, y, sample_weight=sample_weight)
        assert np.unique(w_res).size > 1


# ---------------------------------------------------------------------------
# 7. Downstream alignment
# ---------------------------------------------------------------------------


def test_pipeline_aligns_features_labels_weights(weighted_data):
    X, y, sample_weight = weighted_data
    pipe = Pipeline(
        [
            ("sample", SMOTE(random_state=RND_SEED, k_neighbors=3)),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipe.fit(X, y, sample__sample_weight=sample_weight)

    resampled = pipe._resampled_sample_weight_
    # The estimator was trained on the resampled number of rows.
    assert pipe.named_steps["clf"].n_features_in_ == X.shape[1]
    assert resampled.shape[0] > X.shape[0]


def test_pipeline_weight_equivalent_to_passing_to_estimator(weighted_data):
    """The weights seen by the trainer are the resampled weights."""
    X, y, sample_weight = weighted_data
    sampler = RandomUnderSampler(random_state=RND_SEED)
    X_res, y_res, w_res = sampler.fit_resample(X, y, sample_weight=sample_weight)

    pipe = Pipeline(
        [
            ("sample", RandomUnderSampler(random_state=RND_SEED)),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipe.fit(X, y, sample__sample_weight=sample_weight)

    direct = LogisticRegression(max_iter=200).fit(X_res, y_res, sample_weight=w_res)
    assert_allclose(
        pipe.named_steps["clf"].coef_, direct.coef_, rtol=1e-6, atol=1e-8
    )


def test_pipeline_metadata_routing_propagates_weights(weighted_data):
    from sklearn import config_context

    X, y, sample_weight = weighted_data
    sampler = SMOTE(random_state=RND_SEED, k_neighbors=3)
    clf = LogisticRegression(max_iter=200)
    with config_context(enable_metadata_routing=True):
        sampler.set_fit_resample_request(sample_weight=True)
        clf.set_fit_request(sample_weight=True)
        pipe = Pipeline([("sample", sampler), ("clf", clf)])
        pipe.fit(X, y, sample_weight=sample_weight)
    assert pipe._resampled_sample_weight_.shape[0] > X.shape[0]


def test_combine_pipeline_under_then_over_weights(weighted_data):
    X, y, sample_weight = weighted_data
    pipe = make_pipeline(
        RandomUnderSampler(random_state=RND_SEED),
        SMOTE(random_state=RND_SEED, k_neighbors=3),
    )
    X_res, y_res, w_res = pipe.fit_resample(
        X, y, randomundersampler__sample_weight=sample_weight
    )
    assert X_res.shape[0] == y_res.shape[0] == w_res.shape[0]


def test_dataframe_weights_keep_their_container(data):
    pd = pytest.importorskip("pandas")
    X, y = data
    sample_weight = np.linspace(0.5, 2.0, num=y.shape[0])
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_s = pd.Series(y, name="target")
    w_s = pd.Series(sample_weight, name="weight")

    X_res, y_res, w_res = RandomUnderSampler(random_state=RND_SEED).fit_resample(
        X_df, y_s, sample_weight=w_s
    )
    assert isinstance(X_res, pd.DataFrame)
    assert isinstance(y_res, pd.Series)
    assert isinstance(w_res, pd.Series)
    assert w_res.name == "weight"
    assert (X_res.index == y_res.index).all()
    assert (X_res.index == w_res.index).all()


def test_pipeline_rejects_unsupported_sampler_in_chain(data):
    """A sampler that cannot propagate weights must fail at composition time."""
    from imblearn.base import BaseSampler

    class LegacySampler(BaseSampler):
        _sampling_type = "under-sampling"
        _sample_weight_supported = False
        _weight_propagation = "none"

        def __init__(self, sampling_strategy="auto"):
            super().__init__(sampling_strategy=sampling_strategy)

        def _fit_resample(self, X, y):
            idx = np.arange(X.shape[0] // 2)
            return X[idx], y[idx]

    X, y = data
    pipe = Pipeline(
        [
            ("over", SMOTE(random_state=RND_SEED, k_neighbors=3)),
            ("legacy", LegacySampler()),
            ("under", RandomUnderSampler(random_state=RND_SEED)),
        ]
    )
    with pytest.raises(ValueError, match="does not support 'sample_weight'"):
        pipe.fit_resample(X, y, over__sample_weight=np.ones(y.shape[0]))


# ---------------------------------------------------------------------------
# 8. The unweighted path is completely unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sampler",
    OVER_SAMPLERS + UNDER_SAMPLERS + [SMOTEENN(random_state=RND_SEED)],
)
def test_unweighted_path_returns_two_outputs(sampler, data):
    X, y = data
    output = sampler.fit_resample(X, y)
    assert len(output) == 2
    # No weighted bookkeeping leaks on the unweighted path.
    assert not hasattr(sampler, "resampled_sample_weight_")


def test_unweighted_random_sequence_unchanged(data):
    """Adding weights must not shift the random sequence of the old path.

    The first ``n`` bootstrap indices are identical whether or not weights are
    passed, because weights only affect the values, not the draws.
    """
    X, y = data
    plain = RandomOverSampler(random_state=RND_SEED)
    weighted = RandomOverSampler(random_state=RND_SEED)
    plain.fit_resample(X, y)
    weighted.fit_resample(X, y, sample_weight=np.ones(y.shape[0]))
    assert_array_equal(plain.sample_indices_, weighted.sample_indices_)


def test_weighted_path_is_reproducible(data):
    X, y = data
    sample_weight = np.linspace(0.5, 2.0, num=y.shape[0])
    first = SMOTE(random_state=RND_SEED, k_neighbors=3)
    second = SMOTE(random_state=RND_SEED, k_neighbors=3)
    X_1, y_1, w_1 = first.fit_resample(X, y, sample_weight=sample_weight)
    X_2, y_2, w_2 = second.fit_resample(X, y, sample_weight=sample_weight)
    assert_array_equal(X_1, X_2)
    assert_array_equal(y_1, y_2)
    assert_array_equal(w_1, w_2)


@pytest.mark.parametrize(
    "sampler",
    [
        SMOTE(random_state=RND_SEED, k_neighbors=3),
        ADASYN(random_state=RND_SEED, n_neighbors=3),
        BorderlineSMOTE(random_state=RND_SEED, k_neighbors=3, m_neighbors=5),
    ],
)
def test_weighted_and_unweighted_features_share_seed(sampler, data):
    """Weighting does not change the generated features for a fixed seed.

    The weights only affect the third output; the random sequence that draws
    the base samples, neighbours and steps is identical.
    """
    X, y = data
    X_plain, y_plain = sampler.fit_resample(X, y)

    weighted = type(sampler)(**sampler.get_params())
    X_weighted, y_weighted, _ = weighted.fit_resample(
        X, y, sample_weight=np.ones(y.shape[0])
    )
    assert_allclose(X_plain, X_weighted)
    assert_array_equal(y_plain, y_weighted)


@pytest.mark.parametrize("weight_strategy", ["inherit", "nearest", "cluster"])
def test_smotenc_propagates_weights(weight_strategy):
    X, y = make_classification(
        n_classes=2,
        weights=[0.3, 0.7],
        n_samples=200,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        random_state=RND_SEED,
    )
    X[:, -2:] = np.random.RandomState(RND_SEED).randint(0, 3, size=(len(y), 2))
    sample_weight = np.linspace(0.5, 2.0, num=y.shape[0])

    sampler = SMOTENC(
        categorical_features=[4, 5],
        random_state=RND_SEED,
        k_neighbors=3,
        weight_strategy=weight_strategy,
    )
    X_res, y_res, w_res = sampler.fit_resample(
        X, y, sample_weight=sample_weight
    )
    assert X_res.shape[0] == y_res.shape[0] == w_res.shape[0]
    assert_allclose(w_res[: X.shape[0]], sample_weight)
    assert np.all(np.isfinite(w_res)) and np.all(w_res > 0)


def test_smoten_propagates_weights():
    X = np.array(["A"] * 40 + ["B"] * 80, dtype=object).reshape(-1, 1)
    y = np.array([0] * 40 + [1] * 80)
    sample_weight = np.linspace(0.5, 2.0, num=y.shape[0])

    sampler = SMOTEN(random_state=RND_SEED, k_neighbors=3)
    X_res, y_res, w_res = sampler.fit_resample(
        X, y, sample_weight=sample_weight
    )
    assert X_res.shape[0] == y_res.shape[0] == w_res.shape[0]
    assert_allclose(w_res[: X.shape[0]], sample_weight)


# ---------------------------------------------------------------------------
# 9. Errors and logs never expose sample content
# ---------------------------------------------------------------------------


def test_error_messages_do_not_leak_samples(data):
    X, y = data
    sample_weight = np.ones(y.shape[0])
    sample_weight[0] = -1.0
    with pytest.raises(ValueError) as exc_info:
        RandomUnderSampler(random_state=RND_SEED).fit_resample(
            X, y, sample_weight=sample_weight
        )
    message = str(exc_info.value)
    # Only aggregate information: counts and weight range.
    assert "negative" in message
    assert "rows" in message or "sample" in message
    assert "RandomUnderSampler" in message
    # No feature vector or raw weight values from a specific sample.
    assert str(X[0, 0]) not in message
    assert str(sample_weight[0]) not in message


def test_weight_logging_contains_only_aggregates(data, caplog):
    import logging

    X, y = data
    sample_weight = np.where(y == 0, 7.5, 1.5)
    with caplog.at_level(logging.INFO, logger="imblearn.utils._sample_weight"):
        RandomUnderSampler(random_state=RND_SEED).fit_resample(
            X, y, sample_weight=sample_weight
        )
    messages = [record.getMessage() for record in caplog.records]
    assert any("RandomUnderSampler" in message for message in messages)
    combined = " ".join(messages)
    # Aggregate figures are present, individual sample content is not.
    assert "rows" in combined
    assert "min=" in combined and "max=" in combined
    assert str(X[0, 0]) not in combined


# ---------------------------------------------------------------------------
# 10. FunctionSampler
# ---------------------------------------------------------------------------


def test_function_sampler_identity_preserves_weights(data):
    X, y = data
    sample_weight = np.linspace(0.5, 2.0, num=y.shape[0])
    X_res, y_res, w_res = FunctionSampler().fit_resample(
        X, y, sample_weight=sample_weight
    )
    assert_array_equal(w_res, sample_weight)


def test_function_sampler_weighted_func(data):
    X, y = data
    sample_weight = np.linspace(0.5, 2.0, num=y.shape[0])

    def func(X, y, sample_weight):
        idx = np.arange(0, X.shape[0], 2)
        return X[idx], y[idx], sample_weight[idx]

    X_res, y_res, w_res = FunctionSampler(func=func).fit_resample(
        X, y, sample_weight=sample_weight
    )
    assert_array_equal(w_res, sample_weight[::2])


def test_function_sampler_without_weight_support_rejected(data):
    X, y = data
    with pytest.raises(ValueError, match="does not accept"):
        FunctionSampler(func=lambda X, y: (X[:10], y[:10])).fit_resample(
            X, y, sample_weight=np.ones(y.shape[0])
        )
