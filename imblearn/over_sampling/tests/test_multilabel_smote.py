"""Tests for the multi-label label-set combination-aware over-sampler."""

# Authors: imbalanced-learn developers
# License: MIT

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from imblearn.over_sampling import MultiLabelSMOTE
from imblearn.over_sampling._multilabel import (
    EMPTY_LABEL_SET,
    check_multilabel_target,
    group_by_label_set,
)
from imblearn.pipeline import Pipeline

RANDOM_STATE = 42


def _long_tail_data():
    """Dense long-tail multi-label dataset.

    Label-set distribution:
        (0,)      x30  majority
        (1,)      x20
        (0, 2)    x8   rare multi-label combination
        (3,)      x2
        ()        x2   all-zero rows
    """
    rng = np.random.RandomState(0)
    n, n_labels, n_features = 62, 4, 5
    X = rng.randn(n, n_features)
    y = np.zeros((n, n_labels), dtype=np.int64)
    y[0:30, 0] = 1
    y[30:50, 1] = 1
    y[50:58, 0] = 1
    y[50:58, 2] = 1
    y[58:60, 3] = 1
    # rows 60, 61 stay all-zero
    return X, y


def _label_sets_of(y):
    """Return the label set of every row (list, one entry per row)."""
    if sparse.issparse(y):
        y = y.toarray()
    return [tuple(int(label) for label in np.flatnonzero(row)) for row in y]


def _unique_label_sets(y):
    """Return the set of distinct label sets present in ``y``."""
    return set(_label_sets_of(y))


def _count_label_sets(y):
    counts = {}
    for label_set in _label_sets_of(y):
        counts[label_set] = counts.get(label_set, 0) + 1
    return counts


def _as_inputs(X, y, sparse_input):
    if sparse_input:
        return sparse.csr_matrix(X), sparse.csr_matrix(y)
    return X, y


# ---------------------------------------------------------------------------
# 1. dense / sparse indicator matrices
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_dense_and_sparse_indicator_supported(sparse_input):
    X, y = _long_tail_data()
    X_in, y_in = _as_inputs(X, y, sparse_input)

    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(X_in, y_in)

    assert sparse.issparse(y_res) is sparse_input
    assert sparse.issparse(X_res) is sparse_input
    assert y_res.shape[0] == X_res.shape[0]
    assert y_res.shape[1] == y.shape[1]
    if sparse_input:
        assert y_res.format == "csr"


@pytest.mark.parametrize(
    "constructor",
    [
        sparse.csr_matrix,
        sparse.csc_matrix,
        sparse.coo_matrix,
        sparse.csr_array,
        sparse.csc_array,
    ],
)
def test_all_sparse_formats_accepted(constructor):
    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(constructor(X), constructor(y))
    assert sparse.issparse(y_res)
    assert y_res.shape[1] == y.shape[1]
    assert y_res.dtype == y.dtype
    # Synthetic label sets are only real training combinations.
    _, y_dense = MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
    assert _count_label_sets(y_res) == _count_label_sets(y_dense)


def test_sparse_target_is_never_densified(monkeypatch):
    """The sparse path must not call ``toarray`` on the training target."""
    X, y = _long_tail_data()
    ys = sparse.csr_matrix(y)

    def fail_toarray(self=None):
        raise AssertionError("sparse target was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail_toarray)
    monkeypatch.setattr(sparse.csr_matrix, "todense", fail_toarray)

    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE)
    X_checked, y_checked = sampler._check_X_y(sparse.csr_matrix(X), ys)
    assert sparse.issparse(y_checked)
    assert y_checked.format == "csr"
    # Grouping reads label sets from the CSR arrays without densifying.
    groups, counts = group_by_label_set(y_checked)
    assert sum(counts.values()) == y.shape[0]


def test_explicit_zeros_and_duplicates_are_normalised():
    # Explicit zeros and duplicated entries are legal in COO format but must
    # not change the label sets that are extracted.
    # Row 0: (0, 2) with a duplicated entry at column 2 and an explicit zero
    # at column 1. Row 1: (1,). Row 2: (0, 2).
    coo = sparse.coo_matrix(
        (
            np.array([1, 0, 1, 1, 1, 1]),
            (
                np.array([0, 0, 0, 1, 2, 2]),
                np.array([0, 1, 2, 1, 0, 2]),
            ),
        ),
        shape=(3, 3),
    )
    assert coo.nnz == 6
    checked = check_multilabel_target(coo)
    # The explicit zero is dropped and the duplicated one is collapsed.
    assert checked.nnz == 5
    groups, counts = group_by_label_set(checked)
    assert counts[(0, 2)] == 2
    assert counts[(1,)] == 1
    assert list(groups[(0, 2)]) == [0, 2]


# ---------------------------------------------------------------------------
# 2. frequency by label-set combination; rare combinations are grown
# ---------------------------------------------------------------------------
def test_rare_combination_grown_to_budget():
    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE)
    _, y_res = sampler.fit_resample(X, y)

    train_counts = {
        label_set: count
        for label_set, count in sampler.label_set_counts_.items()
        if label_set != EMPTY_LABEL_SET
    }
    majority = max(train_counts.values())
    res_counts = _count_label_sets(y_res)
    for label_set in train_counts:
        assert res_counts[label_set] == majority, (label_set, res_counts[label_set])
    # the rare (0, 2) combination grew from 8 to 30
    assert res_counts[(0, 2)] == 30
    assert sampler.n_synthetic_[(0, 2)] == 22
    # the all-zero combination is forwarded untouched by default
    assert res_counts[EMPTY_LABEL_SET] == 2


def test_dict_strategy_sets_exact_combination_budget():
    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(
        random_state=RANDOM_STATE, sampling_strategy={(0, 2): 25, (3,): 10}
    )
    _, y_res = sampler.fit_resample(X, y)
    res_counts = _count_label_sets(y_res)
    assert res_counts[(0, 2)] == 25
    assert res_counts[(3,)] == 10
    assert sampler.n_synthetic_[(0, 2)] == 17
    assert sampler.n_synthetic_[(3,)] == 8
    # combinations absent from the dict are untouched
    assert res_counts[(0,)] == 30
    assert res_counts[(1,)] == 20


def test_callable_strategy_receives_target_matrix():
    X, y = _long_tail_data()
    seen = {}

    def strategy(y_target):
        seen["shape"] = y_target.shape
        return {(1,): 30}

    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE, sampling_strategy=strategy)
    _, y_res = sampler.fit_resample(X, y)
    assert seen["shape"] == y.shape
    assert _count_label_sets(y_res)[(1,)] == 30


def test_minority_strategy_only_targets_smallest_combination():
    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE, sampling_strategy="minority")
    _, y_res = sampler.fit_resample(X, y)
    res_counts = _count_label_sets(y_res)
    # (3,) has 2 members and (0, 2) has 8: the smallest grows to the second
    # smallest size.
    assert res_counts[(3,)] == 8
    assert res_counts[(0, 2)] == 8
    assert res_counts[(0,)] == 30


# ---------------------------------------------------------------------------
# 3. synthetic label sets always exist in training
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_synthetic_label_set_seen_in_training(sparse_input):
    X, y = _long_tail_data()
    train_sets = _unique_label_sets(y)
    X_in, y_in = _as_inputs(X, y, sparse_input)

    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE)
    _, y_res = sampler.fit_resample(X_in, y_in)

    synthetic_sets = _unique_label_sets(y_res[X.shape[0] :])
    assert synthetic_sets.issubset(train_sets)
    assert all(len(label_set) > 0 for label_set in synthetic_sets)
    if sparse.issparse(y_res):
        assert y_res.indices.max() < y.shape[1]
        assert y_res.data.min() == 1 and y_res.data.max() == 1
    else:
        assert np.unique(y_res).tolist() == [0, 1]


def test_no_unseen_label_combination_can_be_built():
    X, y = _long_tail_data()
    # (1, 3) never occurs in training and must be rejected as a target.
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE, sampling_strategy={(1, 3): 10})
    with pytest.raises(ValueError, match="not present in the training target"):
        sampler.fit_resample(X, y)


@pytest.mark.parametrize("singleton_strategy", ["keep", "merge", "skip"])
def test_synthetic_never_invents_labels(singleton_strategy):
    X, y = _long_tail_data()
    train_sets = _unique_label_sets(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sampler = MultiLabelSMOTE(
            random_state=RANDOM_STATE, singleton_strategy=singleton_strategy
        )
        _, y_res = sampler.fit_resample(X, y)
    assert _unique_label_sets(y_res[X.shape[0] :]).issubset(train_sets)


# ---------------------------------------------------------------------------
# 4. per-label and global budgets
# ---------------------------------------------------------------------------
def test_total_budget_caps_total_synthetic():
    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE, total_budget=10)
    _, y_res = sampler.fit_resample(X, y)
    assert sum(sampler.n_synthetic_.values()) == 10
    assert y_res.shape[0] == X.shape[0] + 10


def test_total_budget_larger_than_request_is_not_binding():
    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE, total_budget=10_000)
    _, y_res = sampler.fit_resample(X, y)
    assert sum(sampler.n_synthetic_.values()) < 10_000
    assert sampler.sampling_strategy_ == {
        (0,): 30,
        (1,): 30,
        (0, 2): 30,
        (3,): 30,
        EMPTY_LABEL_SET: 2,
    } or all(v == 30 for k, v in sampler.sampling_strategy_.items() if k != ())


def test_label_budget_hard_raises_with_shortfall():
    X, y = _long_tail_data()
    # label 2 only appears in the (0, 2) group, which wants 22 synthetic rows
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE, label_budget={2: 5})
    with pytest.raises(ValueError, match="short by 17"):
        sampler.fit_resample(X, y)


def test_label_budget_warn_mode_reports_shortfall():
    X, y = _long_tail_data()
    with pytest.warns(UserWarning, match="short by"):
        sampler = MultiLabelSMOTE(
            random_state=RANDOM_STATE,
            label_budget={2: 5},
            enforce_label_budget="warn",
        )
        _, y_res = sampler.fit_resample(X, y)
    assert sampler.label_budget_shortfall_ == {2: 17}
    # only 5 synthetic rows carry label 2
    assert int(y_res[X.shape[0] :, 2].sum()) == 5


def test_label_budget_does_not_flood_other_label():
    X, y = _long_tail_data()
    # label 0 appears in (0,) and (0, 2); cap it while label 1 is still filled.
    with pytest.warns(UserWarning, match="short by"):
        sampler = MultiLabelSMOTE(
            random_state=RANDOM_STATE,
            label_budget={0: 3},
            enforce_label_budget="warn",
        )
        _, y_res = sampler.fit_resample(X, y)
    synthetic = y_res[X.shape[0] :]
    assert int(synthetic[:, 0].sum()) <= 3
    # the label only present in (1,) is unaffected by the cap on label 0
    assert int(synthetic[:, 1].sum()) == 10


def test_label_budget_absent_label_is_uncapped():
    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE, label_budget={1: 10})
    _, y_res = sampler.fit_resample(X, y)
    assert int(y_res[X.shape[0] :, 1].sum()) == 10
    assert sampler.label_budget_shortfall_ == {}


# ---------------------------------------------------------------------------
# 5. single-positive, all-zero and very wide label rows
# ---------------------------------------------------------------------------
def test_single_positive_and_all_zero_rows_have_defined_behaviour():
    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(X, y)
    output_counts = _count_label_sets(y_res)
    assert output_counts[EMPTY_LABEL_SET] == 2
    assert EMPTY_LABEL_SET not in sampler.n_synthetic_
    # every original row is retained (default singleton "keep" + zero "keep")
    assert X_res.shape[0] == X.shape[0] + sum(sampler.n_synthetic_.values())


def test_all_zero_rows_can_be_requested_explicitly():
    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(
        random_state=RANDOM_STATE,
        sampling_strategy={EMPTY_LABEL_SET: 10},
        zero_label_rows="resample",
    )
    _, y_res = sampler.fit_resample(X, y)
    assert _count_label_sets(y_res)[EMPTY_LABEL_SET] == 10
    assert sampler.n_synthetic_[EMPTY_LABEL_SET] == 8


def test_explicit_zero_request_rejected_under_keep_policy():
    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(
        random_state=RANDOM_STATE, sampling_strategy={EMPTY_LABEL_SET: 10}
    )
    with pytest.raises(ValueError, match="zero_label_rows='keep'"):
        sampler.fit_resample(X, y)


def test_zero_label_rows_resample_policy():
    rng = np.random.RandomState(2)
    X = rng.randn(20, 4)
    y = np.zeros((20, 3), dtype=np.int64)
    y[0:10, 0] = 1
    y[10:15, 1] = 1
    y[15, 2] = 1
    # rows 16-19 form an all-zero group of 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sampler = MultiLabelSMOTE(
            random_state=RANDOM_STATE,
            zero_label_rows="resample",
            sampling_strategy="all",
        )
        _, y_res = sampler.fit_resample(X, y)
    output_counts = _count_label_sets(y_res)
    n_zero_synthetic = sampler.n_synthetic_[EMPTY_LABEL_SET]
    assert n_zero_synthetic > 0
    assert output_counts[EMPTY_LABEL_SET] == 4 + n_zero_synthetic
    # Only real training combinations are used and the all-zero group was
    # actually grown with synthetic all-zero rows.
    synthetic_sets = _unique_label_sets(y_res[X.shape[0] :])
    assert synthetic_sets.issubset(_unique_label_sets(y))
    assert synthetic_sets == {EMPTY_LABEL_SET, (1,)}


def test_wide_label_matrix_many_labels_per_row():
    rng = np.random.RandomState(1)
    n, n_labels = 40, 12
    X = rng.randn(n, 6)
    y = np.zeros((n, n_labels), dtype=np.int64)
    big_set = tuple(range(10))
    y[:20, :10] = 1
    y[20:30, 10] = 1
    y[30:40, 11] = 1
    sampler = MultiLabelSMOTE(
        random_state=RANDOM_STATE,
        sampling_strategy={big_set: 30, (10,): 10, (11,): 10},
    )
    X_res, y_res = sampler.fit_resample(X, y)
    assert _unique_label_sets(y_res[X.shape[0] :]) == {big_set}
    assert X_res.shape[1] == X.shape[1]
    assert y_res.shape[1] == n_labels


# ---------------------------------------------------------------------------
# 6. singleton combination policy
# ---------------------------------------------------------------------------
def _singleton_data():
    """Dataset where label set (3,) occurs exactly once."""
    rng = np.random.RandomState(3)
    # (0,) x8, (1,) x6, (0, 2) x4, (3,) x1 (true singleton)
    X = rng.randn(19, 4)
    y = np.zeros((19, 4), dtype=np.int64)
    y[0:8, 0] = 1
    y[8:14, 1] = 1
    y[14:18, 0] = 1
    y[14:18, 2] = 1
    y[18, 3] = 1
    return X, y


def test_singleton_strategy_keep_merge_skip():
    X, y = _singleton_data()
    train_sets = _unique_label_sets(y)
    assert (3,) in train_sets

    # keep: original retained, no synthetic for (3,)
    keep = MultiLabelSMOTE(random_state=RANDOM_STATE, singleton_strategy="keep")
    with pytest.warns(UserWarning, match="appears only once"):
        Xk, yk = keep.fit_resample(X, y)
    assert _label_sets_of(yk).count((3,)) == 1
    assert keep.n_synthetic_[(3,)] == 0

    # skip: the lone (3,) row is removed; other groups still grow
    skip = MultiLabelSMOTE(random_state=RANDOM_STATE, singleton_strategy="skip")
    Xs, ys = skip.fit_resample(X, y)
    assert (3,) not in _unique_label_sets(ys)
    n_kept_originals = X.shape[0] - 1
    n_synthetic = sum(skip.n_synthetic_.values())
    assert Xs.shape[0] == n_kept_originals + n_synthetic

    # merge: synthetic rows carry a donor combination that exists in training
    merge = MultiLabelSMOTE(random_state=RANDOM_STATE, singleton_strategy="merge")
    Xm, ym = merge.fit_resample(X, y)
    assert _label_sets_of(ym).count((3,)) == 1  # original kept
    synthetic_sets = _unique_label_sets(ym[X.shape[0] :])
    assert synthetic_sets.issubset(train_sets)
    assert (3,) not in synthetic_sets
    assert merge.n_synthetic_[(3,)] == 7


@pytest.mark.parametrize("strategy", ["keep", "merge", "skip"])
def test_singleton_policy_does_not_apply_to_non_singleton(strategy):
    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE, singleton_strategy=strategy)
    _, y_res = sampler.fit_resample(X, y)
    assert _count_label_sets(y_res)[(0, 2)] == 30
    assert sampler.n_synthetic_[(0, 2)] == 22


def test_singleton_strategy_parameter_validation():
    with pytest.raises(ValueError, match="singleton_strategy"):
        MultiLabelSMOTE(singleton_strategy="bogus").fit_resample(*_long_tail_data())


# ---------------------------------------------------------------------------
# 7. row alignment, dtypes and indices
# ---------------------------------------------------------------------------
def test_output_alignment_dtype_and_shape():
    X, y = _long_tail_data()
    y32 = y.astype(np.int32)
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(X, y32)
    assert X_res.shape[0] == y_res.shape[0]
    assert y_res.shape[1] == y32.shape[1]
    assert y_res.dtype == np.int32
    # originals come first, then synthetic rows in generation order
    assert sampler.sample_indices_.shape[0] == y_res.shape[0]
    np.testing.assert_array_equal(
        sampler.sample_indices_[: X.shape[0]], np.arange(X.shape[0])
    )
    assert (sampler.sample_indices_[X.shape[0] :] == -1).all()
    # X and y stay aligned: every original row is unchanged
    np.testing.assert_array_equal(X_res[: X.shape[0]], X)
    np.testing.assert_array_equal(y_res[: X.shape[0]], y32)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.float32, np.bool_])
def test_dtype_preserved(dtype):
    X, y = _long_tail_data()
    y_cast = y.astype(dtype)
    _, y_res = MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(X, y_cast)
    assert y_res.dtype == y_cast.dtype


def test_sparse_output_flag_overrides_input_sparsity():
    X, y = _long_tail_data()
    dense_out = MultiLabelSMOTE(random_state=RANDOM_STATE, output_sparse=False)
    _, y_res = dense_out.fit_resample(sparse.csr_matrix(X), sparse.csr_matrix(y))
    assert isinstance(y_res, np.ndarray)

    sparse_out = MultiLabelSMOTE(random_state=RANDOM_STATE, output_sparse=True)
    _, y_res2 = sparse_out.fit_resample(X, y)
    assert sparse.issparse(y_res2)
    assert y_res2.format == "csr"


def test_pandas_input_returns_pandas_output():
    X, y = _long_tail_data()
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    ydf = pd.DataFrame(y, columns=[f"l{i}" for i in range(y.shape[1])])
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(Xdf, ydf)
    assert isinstance(X_res, pd.DataFrame)
    assert isinstance(y_res, pd.DataFrame)
    assert list(X_res.columns) == list(Xdf.columns)
    assert list(y_res.columns) == list(ydf.columns)
    assert len(X_res) == len(y_res)
    assert (y_res.dtypes == ydf.dtypes).all()


def test_pandas_feature_names_recorded():
    X, y = _long_tail_data()
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE)
    sampler.fit(Xdf, y)
    np.testing.assert_array_equal(sampler.feature_names_in_, Xdf.columns.to_numpy())
    assert sampler.n_features_in_ == X.shape[1]


# ---------------------------------------------------------------------------
# 8. direct consumption by a downstream classifier
# ---------------------------------------------------------------------------
def test_pipeline_with_multilabel_classifier():
    X, y = _long_tail_data()
    pipe = Pipeline(
        [
            ("sampler", MultiLabelSMOTE(random_state=RANDOM_STATE)),
            ("clf", MultiOutputClassifier(LogisticRegression(max_iter=200))),
        ]
    )
    pipe.fit(X, y)
    pred = pipe.predict(X[:5])
    assert pred.shape == (5, y.shape[1])


def test_pipeline_sparse_input_dense_target_output():
    X, y = _long_tail_data()
    pipe = Pipeline(
        [
            (
                "sampler",
                MultiLabelSMOTE(random_state=RANDOM_STATE, output_sparse=False),
            ),
            ("clf", MultiOutputClassifier(LogisticRegression(max_iter=200))),
        ]
    )
    pipe.fit(sparse.csr_matrix(X), sparse.csr_matrix(y))
    assert pipe.predict(X[:5]).shape == (5, y.shape[1])


def test_downstream_classifier_consumes_sparse_target_directly():
    from sklearn.utils.multiclass import type_of_target

    X, y = _long_tail_data()
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(sparse.csr_matrix(X), sparse.csr_matrix(y))
    assert type_of_target(y_res) == "multilabel-indicator"
    # The resampled target can be fed to a multi-output classifier as-is.
    from sklearn.multiclass import OneVsRestClassifier

    clf = OneVsRestClassifier(LogisticRegression(max_iter=200)).fit(X_res, y_res)
    assert clf.predict(X[:3]).shape == (3, y.shape[1])


# ---------------------------------------------------------------------------
# 9. determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_fixed_seed_bit_for_bit_deterministic(sparse_input):
    X, y = _long_tail_data()
    X_in, y_in = _as_inputs(X, y, sparse_input)

    def run():
        sampler = MultiLabelSMOTE(random_state=RANDOM_STATE)
        return sampler.fit_resample(X_in, y_in)

    X1, y1 = run()
    X2, y2 = run()
    if sparse_input:
        assert (X1 != X2).nnz == 0
        assert (y1 != y2).nnz == 0
        np.testing.assert_array_equal(X1.toarray(), X2.toarray())
        np.testing.assert_array_equal(y1.toarray(), y2.toarray())
    else:
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


def test_fixed_seed_dense_and_sparse_are_equivalent():
    X, y = _long_tail_data()
    X_dense, y_dense = MultiLabelSMOTE(random_state=7).fit_resample(X, y)
    X_sparse, y_sparse = MultiLabelSMOTE(random_state=7).fit_resample(
        sparse.csr_matrix(X), sparse.csr_matrix(y)
    )
    np.testing.assert_allclose(X_dense, X_sparse.toarray())
    np.testing.assert_array_equal(y_dense, y_sparse.toarray())


def test_different_seeds_produce_different_samples():
    X, y = _long_tail_data()
    X1, _ = MultiLabelSMOTE(random_state=1).fit_resample(X, y)
    X2, _ = MultiLabelSMOTE(random_state=2).fit_resample(X, y)
    assert not np.array_equal(X1[X.shape[0] :], X2[X.shape[0] :])


# ---------------------------------------------------------------------------
# 10. error handling
# ---------------------------------------------------------------------------
def test_illegal_target_values_report_positions():
    X, y = _long_tail_data()
    y_bad = y.copy()
    y_bad[0, 0] = 2
    y_bad[3, 1] = -1
    with pytest.raises(ValueError, match="illegal entries") as exc:
        MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(X, y_bad)
    message = str(exc.value)
    assert "(0, 0)" in message
    assert "2" in message


def test_illegal_sparse_target_reports_count():
    X, y = _long_tail_data()
    y_bad = sparse.csr_matrix(y)
    y_bad[0, 0] = 3
    with pytest.raises(ValueError, match="illegal entries") as exc:
        MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(
            sparse.csr_matrix(X), y_bad
        )
    assert "1 illegal entries" in str(exc.value)


def test_row_count_mismatch_message():
    X, y = _long_tail_data()
    with pytest.raises(ValueError, match="same number of samples"):
        MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(X[:-2], y)


def test_too_many_all_zero_rows_reports_count():
    X, y = _long_tail_data()
    with pytest.raises(ValueError, match="all-zero label row") as exc:
        MultiLabelSMOTE(
            random_state=RANDOM_STATE, zero_label_rows_threshold=0.01
        ).fit_resample(X, y)
    assert "2" in str(exc.value)


def test_zero_label_rows_threshold_accepts_count():
    X, y = _long_tail_data()
    # exactly two all-zero rows are tolerated, one is not
    MultiLabelSMOTE(
        random_state=RANDOM_STATE, zero_label_rows_threshold=2
    ).fit_resample(X, y)
    with pytest.raises(ValueError, match="all-zero label row"):
        MultiLabelSMOTE(
            random_state=RANDOM_STATE, zero_label_rows_threshold=1
        ).fit_resample(X, y)


def test_one_dimensional_target_points_to_single_label_sampler():
    X, y = _long_tail_data()
    with pytest.raises(ValueError, match="1D targets"):
        MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(X, y[:, 0])


def test_dead_label_column_message():
    X, y = _long_tail_data()
    y = y.copy()
    y[:, 2] = 0  # column 2 becomes inactive everywhere
    with pytest.raises(ValueError, match="never take a positive value"):
        MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(X, y)


def test_single_label_column_message():
    X, y = _long_tail_data()
    with pytest.raises(ValueError, match="at least 2 label columns"):
        MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(X, y[:, :1])


def test_label_budget_out_of_range_index():
    X, y = _long_tail_data()
    with pytest.raises(ValueError, match="references label index"):
        MultiLabelSMOTE(
            random_state=RANDOM_STATE, label_budget={y.shape[1] + 1: 5}
        ).fit_resample(X, y)


def test_undersized_dict_strategy_rejected():
    X, y = _long_tail_data()
    with pytest.raises(ValueError, match="greater than or equal"):
        MultiLabelSMOTE(
            random_state=RANDOM_STATE, sampling_strategy={(0,): 5}
        ).fit_resample(X, y)


def test_negative_dict_strategy_rejected():
    X, y = _long_tail_data()
    with pytest.raises(ValueError, match="non-negative integers"):
        MultiLabelSMOTE(
            random_state=RANDOM_STATE, sampling_strategy={(0,): -1}
        ).fit_resample(X, y)


def test_all_zero_target_rejected():
    X, _ = _long_tail_data()
    y = np.zeros((X.shape[0], 3), dtype=np.int64)
    with pytest.raises(ValueError, match="never take a positive value"):
        MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(X, y)


def test_negative_zero_row_threshold_rejected():
    X, y = _long_tail_data()
    with pytest.raises(ValueError, match="must be an int in the range"):
        MultiLabelSMOTE(
            random_state=RANDOM_STATE, zero_label_rows_threshold=-1
        ).fit_resample(X, y)


# ---------------------------------------------------------------------------
# 11. no leakage from held-out data
# ---------------------------------------------------------------------------
def test_generated_rows_ignore_held_out_data():
    """Resampling must not read any rows outside the training batch."""
    X, y = _long_tail_data()
    # Keep every label active in the training batch.
    X_train, y_train, X_holdout, y_holdout = X[:60], y[:60], X[60:], y[60:]

    # Only the training batch is passed to the sampler.
    X_res, y_res = MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(
        X_train, y_train
    )
    # Every synthetic row must be a convex combination of two training rows.
    synthetic = X_res[X_train.shape[0] :]
    assert synthetic.shape[0] > 0
    for row in synthetic:
        distances = np.linalg.norm(X_train - row, axis=1)
        assert (
            distances.min()
            <= np.linalg.norm(X_train[:, None, :] - X_train[None, :, :], axis=-1).max()
        )

    # Changing the held-out data cannot change the resampled output.
    X_holdout_2 = X_holdout + 100.0
    X_res_2, y_res_2 = MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(
        X_train, y_train
    )
    np.testing.assert_array_equal(X_res, X_res_2)
    np.testing.assert_array_equal(y_res, y_res_2)
    # And the held-out arrays themselves were not touched.
    np.testing.assert_allclose(X_holdout_2 - 100.0, X_holdout)
    np.testing.assert_array_equal(y_holdout, y_holdout)


# ---------------------------------------------------------------------------
# 12. memory: sparse grouping is linear in nnz, not n_labels ** 2
# ---------------------------------------------------------------------------
def test_sparse_grouping_memory_linear_in_nnz():
    rng = np.random.RandomState(7)
    n = 5_000
    peaks = {}
    for n_labels in (20, 200, 2000):
        rows = np.repeat(np.arange(n), 2)
        cols = rng.randint(0, n_labels, size=n * 2)
        yy = sparse.csr_matrix(
            (np.ones(n * 2, dtype=np.int8), (rows, cols)), shape=(n, n_labels)
        )
        yy.sum_duplicates()
        yy.data[:] = 1
        yy.sort_indices()
        _, counts = group_by_label_set(yy)
        max_tuple_len = max(len(label_set) for label_set in counts)
        peaks[n_labels] = (len(counts), max_tuple_len)
        # Each stored tuple is bounded by the row nnz (2), never by n_labels.
        assert max_tuple_len <= 2
        # The number of stored ones does not depend on the width.
        assert yy.nnz <= n * 2
    # Wider matrices only create more distinct combinations, not wider tuples.
    assert peaks[2000][0] >= peaks[20][0]


def test_sparse_output_block_memory_is_proportional_to_labels():
    from imblearn.over_sampling._multilabel import indicator_block

    block = indicator_block((0, 2), 100_000, 50, sparse_output=True)
    assert block.shape == (100_000, 50)
    assert block.nnz == 200_000


# ---------------------------------------------------------------------------
# 13. single-label samplers are untouched
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sampler_name",
    ["RandomOverSampler", "SMOTE", "ADASYN", "SMOTEN", "SMOTENC", "SVMSMOTE"],
)
def test_single_label_samplers_still_public(sampler_name):
    import imblearn.over_sampling as over_sampling

    assert hasattr(over_sampling, sampler_name)
    assert sampler_name in over_sampling.__all__


def test_single_label_sampler_rejects_multilabel_and_keeps_results():
    from sklearn.datasets import make_classification

    from imblearn.over_sampling import SMOTE, RandomOverSampler

    X, y = make_classification(
        n_samples=60, n_classes=2, weights=[0.2, 0.8], random_state=0
    )
    y_res = RandomOverSampler(random_state=0).fit_resample(X, y)[1]
    assert np.unique(y_res, return_counts=True)[1].tolist() == [48, 48]

    sampler = SMOTE(random_state=0)
    X_res, y_res = sampler.fit_resample(X, y)
    assert X_res.shape[0] == y_res.shape[0]
    assert np.unique(y_res, return_counts=True)[1].tolist() == [48, 48]


# ---------------------------------------------------------------------------
# 14. additional regressions
# ---------------------------------------------------------------------------
def test_ragged_and_scalar_targets_raise_informative_error():
    X = np.random.RandomState(0).randn(2, 3)
    with pytest.raises(ValueError, match="binary indicator matrix"):
        MultiLabelSMOTE().fit_resample(X, [[1, 0, 0], [0, 1]])
    with pytest.raises(ValueError, match="binary indicator matrix"):
        MultiLabelSMOTE().fit_resample(X, 3)


def test_pandas_sparse_dataframe_target():
    X, y = _long_tail_data()
    y_sparse = pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(y))
    X_res, y_res = MultiLabelSMOTE(random_state=RANDOM_STATE).fit_resample(X, y_sparse)
    assert isinstance(y_res, pd.DataFrame)
    assert len(X_res) == len(y_res) == 122
    assert _count_label_sets(y_res.to_numpy())[(0, 2)] == 30


def test_singleton_keep_is_applied_before_budget_caps():
    X, y = _singleton_data()
    # Label 3 is only carried by the singleton: keeping it cannot trigger a
    # budget shortfall, because no synthetic row will carry label 3 anyway.
    with pytest.warns(UserWarning, match="appears only once"):
        sampler = MultiLabelSMOTE(
            random_state=RANDOM_STATE,
            singleton_strategy="keep",
            label_budget={3: 0},
        )
        _, y_res = sampler.fit_resample(X, y)
    assert sampler.label_budget_shortfall_ == {}
    assert sampler.n_synthetic_[(3,)] == 0
    assert _count_label_sets(y_res)[(3,)] == 1


def test_zero_label_rows_threshold_type_is_validated():
    X, y = _long_tail_data()
    with pytest.raises(ValueError, match="zero_label_rows_threshold"):
        MultiLabelSMOTE(
            random_state=RANDOM_STATE, zero_label_rows_threshold=2.5
        ).fit_resample(X, y)


def test_singleton_skip_drops_only_the_singleton_row():
    X, y = _singleton_data()
    sampler = MultiLabelSMOTE(random_state=RANDOM_STATE, singleton_strategy="skip")
    X_res, y_res = sampler.fit_resample(X, y)
    assert (3,) not in _unique_label_sets(y_res)
    # The rows kept are originals followed by synthetic ones, and the
    # feature rows are aligned with the label rows.
    assert X_res.shape[0] == y_res.shape[0]
    assert sampler.sample_indices_.shape[0] == y_res.shape[0]
    n_kept = y_res.shape[0] - sum(sampler.n_synthetic_.values())
    assert n_kept == X.shape[0] - 1


def test_grouping_handles_unsorted_sparse_input():
    y = sparse.csr_matrix(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    # Reverse the column indices *within each row* without touching indptr,
    # so the matrix stays valid but its indices become unsorted.
    unsorted = y.copy()
    for row in range(unsorted.shape[0]):
        start, stop = unsorted.indptr[row], unsorted.indptr[row + 1]
        unsorted.indices[start:stop] = unsorted.indices[start:stop][::-1]
    unsorted.has_sorted_indices = False
    checked = check_multilabel_target(unsorted)
    assert checked.has_sorted_indices
    _, counts = group_by_label_set(checked)
    assert counts[(0, 2)] == 2
    assert counts[(1,)] == 2
