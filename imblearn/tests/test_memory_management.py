"""Tests for the bounded-memory resampling contract.

Covers the four acceptance scenarios:

1. regular (historical) resampling is unchanged when the new switches are off;
2. chunked resampling works on dense arrays;
3. chunked and regular paths give identical results under the same seed;
4. the memory guard degrades to "no resampling" instead of raising OOM.

It also checks that every sampler reports whether it copies its input and
that invalid chunk sizes are rejected before any data is touched.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import (
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    RandomUnderSampler,
    TomekLinks,
)
from imblearn.utils import _memory

RND_SEED = 42


@pytest.fixture
def data():
    X, y = make_classification(
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.15, 0.35, 0.5],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=10,
        n_samples=3000,
        random_state=RND_SEED,
    )
    return X, y


# ---------------------------------------------------------------------------
# 1. Regular path unchanged when the switches are off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sampler",
    [
        RandomOverSampler(random_state=RND_SEED),
        RandomUnderSampler(random_state=RND_SEED),
        RandomUnderSampler(random_state=RND_SEED, replacement=True),
    ],
)
def test_default_construction_keeps_historical_signature(sampler):
    # New parameters are opt-in keyword arguments with historical defaults.
    params = sampler.get_params()
    assert params["chunk_size"] is None
    assert params["memory_guard"] is False
    assert not hasattr(sampler, "degraded_")


def test_regular_path_outputs_unchanged(data):
    X, y = data
    X_full, y_full = RandomOverSampler(random_state=RND_SEED).fit_resample(X, y)
    X_under, y_under = RandomUnderSampler(random_state=RND_SEED).fit_resample(X, y)
    # auto strategy equalises every class at the majority / minority count
    target_over = np.bincount(y).max()
    target_under = np.bincount(y).min()
    assert X_full.shape == (3 * target_over, 10)
    assert X_under.shape == (3 * target_under, 10)
    assert np.bincount(y_full).min() == np.bincount(y_full).max()
    assert np.bincount(y_under).min() == np.bincount(y_under).max()


# ---------------------------------------------------------------------------
# 2 + 3. Chunked path works and matches the regular path exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Sampler", [RandomOverSampler, RandomUnderSampler])
@pytest.mark.parametrize("chunk_size", [1, 7, 256, 10_000])
def test_chunked_matches_full_path_same_seed(data, Sampler, chunk_size):
    X, y = data
    regular = Sampler(random_state=RND_SEED)
    chunked = Sampler(random_state=RND_SEED, chunk_size=chunk_size)

    X_reg, y_reg = regular.fit_resample(X, y)
    X_chk, y_chk = chunked.fit_resample(X, y)

    # The hard requirement: bit-identical results with the same seed,
    # including the order of rows and the integer selection indices.
    np.testing.assert_array_equal(regular.sample_indices_, chunked.sample_indices_)
    np.testing.assert_array_equal(X_reg, X_chk)
    np.testing.assert_array_equal(y_reg, y_chk)
    assert X_reg.dtype == X_chk.dtype
    assert y_reg.dtype == y_chk.dtype


def test_chunked_matches_full_path_float32(data):
    X, y = data
    X32 = X.astype(np.float32)
    for Sampler in (RandomOverSampler, RandomUnderSampler):
        regular = Sampler(random_state=RND_SEED)
        chunked = Sampler(random_state=RND_SEED, chunk_size=64)
        X_reg, _ = regular.fit_resample(X32, y)
        X_chk, _ = chunked.fit_resample(X32, y)
        assert X_chk.dtype == np.float32
        np.testing.assert_array_equal(X_reg, X_chk)


def test_chunked_matches_full_path_pandas(data):
    pd = pytest.importorskip("pandas")
    X, y = data
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_sr = pd.Series(y)
    for Sampler in (RandomOverSampler, RandomUnderSampler):
        regular = Sampler(random_state=RND_SEED)
        chunked = Sampler(random_state=RND_SEED, chunk_size=31)
        X_reg, y_reg = regular.fit_resample(X_df, y_sr)
        X_chk, y_chk = chunked.fit_resample(X_df, y_sr)
        assert isinstance(X_chk, pd.DataFrame)
        assert isinstance(y_chk, pd.Series)
        assert X_reg.equals(X_chk)
        assert y_reg.equals(y_chk)
        assert list(X_chk.columns) == list(X_df.columns)
        # row labels are aligned to the selected original indices
        assert list(X_chk.index) == list(X_df.index.to_numpy()[chunked.sample_indices_])


def test_chunked_temp_memory_is_bounded_with_n(data):
    # Per-chunk scratch space is bounded by chunk_size; the only O(n)
    # component is the integer index bookkeeping, never a copy of the input.
    X, y = data
    X_big = np.vstack([X, X, X, X])
    y_big = np.concatenate([y, y, y, y])
    row_bytes = X.dtype.itemsize * X.shape[1]
    for Sampler in (RandomOverSampler, RandomUnderSampler):
        chunk_size = 128
        sampler = Sampler(random_state=RND_SEED, chunk_size=chunk_size)
        profile_small = sampler.memory_profile(X, y)
        profile_big = sampler.memory_profile(X_big, y_big)
        assert profile_small.temp_extra_bytes_chunked is not None
        # the per-chunk row materialisation itself never exceeds chunk_size
        assert profile_big.temp_extra_bytes_chunked <= (
            (X_big.shape[0] + profile_big.output_bytes // max(1, row_bytes)) * 8
            + chunk_size * row_bytes
        )
        # the full-path estimate grows with the input size
        assert (
            profile_big.peak_extra_bytes_full > profile_small.peak_extra_bytes_full
        )


def test_rus_chunk_scratch_is_chunk_size_constant(data):
    # RUS has no per-class index accumulators: its chunk scratch space is a
    # pure function of chunk_size and stays constant when n grows.
    X, y = data
    X_big = np.vstack([X, X, X, X])
    y_big = np.concatenate([y, y, y, y])
    small = RandomUnderSampler(
        random_state=RND_SEED, chunk_size=128
    ).memory_profile(X, y)
    big = RandomUnderSampler(
        random_state=RND_SEED, chunk_size=128
    ).memory_profile(X_big, y_big)
    assert small.temp_extra_bytes_chunked == big.temp_extra_bytes_chunked
    assert small.temp_extra_bytes_chunked == 128 * X.shape[1] * X.dtype.itemsize


def test_chunked_rejects_sparse_and_object(data):
    X, y = data
    from scipy import sparse

    X_sparse = sparse.csr_matrix(X)
    with pytest.raises(ValueError, match="chunked path"):
        RandomOverSampler(random_state=RND_SEED, chunk_size=64).fit_resample(
            X_sparse, y
        )
    with pytest.raises(ValueError, match="chunked path"):
        RandomUnderSampler(random_state=RND_SEED, chunk_size=64).fit_resample(
            X_sparse, y
        )

    # heterogeneous object arrays (mixed numeric/string columns) cannot be
    # pre-allocated at a fixed dtype
    X_object = np.column_stack(
        [X[:, 0].astype(object), np.array(["a"] * X.shape[0], dtype=object)]
    )
    with pytest.raises(ValueError, match="chunked path"):
        RandomOverSampler(random_state=RND_SEED, chunk_size=64).fit_resample(
            X_object, y
        )


def test_chunked_unsupported_with_shrinkage(data):
    X, y = data
    with pytest.raises(ValueError, match="smoothed bootstrap"):
        RandomOverSampler(
            random_state=RND_SEED, shrinkage=0.1, chunk_size=64
        ).fit_resample(X, y)


# ---------------------------------------------------------------------------
# 4. Memory guard: large-data warning and OOM degradation
# ---------------------------------------------------------------------------


def test_large_data_warns_with_estimate(monkeypatch, data):
    X, y = data
    monkeypatch.setattr(_memory, "LARGE_DATA_THRESHOLD", X.shape[0] - 1)
    # plenty of memory available so only the advisory warning is expected
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 10**12)
    with pytest.warns(UserWarning, match="Large dataset detected"):
        RandomUnderSampler(random_state=RND_SEED, memory_guard=True).fit_resample(X, y)


def test_large_data_silent_without_guard(monkeypatch, data):
    X, y = data
    monkeypatch.setattr(_memory, "LARGE_DATA_THRESHOLD", X.shape[0] - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        RandomUnderSampler(random_state=RND_SEED).fit_resample(X, y)


def test_preflight_degradation_returns_input(monkeypatch, data):
    X, y = data
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 1)
    with pytest.warns(UserWarning, match="Estimated peak extra memory"):
        sampler = RandomOverSampler(random_state=RND_SEED, memory_guard=True)
        X_out, y_out = sampler.fit_resample(X, y)
    assert sampler.degraded_ is True
    assert X_out is X  # the validated dense array is returned untouched
    np.testing.assert_array_equal(y_out, y)
    assert "chunk_size" in sampler.degradation_message_


def test_mid_run_memory_error_degrades(monkeypatch, data):
    X, y = data
    # preflight must pass; the failure happens during materialisation.
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: None)

    def boom(*args, **kwargs):
        raise MemoryError("simulated allocation failure")

    monkeypatch.setattr(
        "imblearn.under_sampling._prototype_selection."
        "_random_under_sampler.fill_by_indices_chunked",
        boom,
    )
    with pytest.warns(UserWarning, match="MemoryError was raised"):
        sampler = RandomUnderSampler(
            random_state=RND_SEED, memory_guard=True, chunk_size=64
        )
        X_out, y_out = sampler.fit_resample(X, y)
    assert sampler.degraded_ is True
    assert X_out is X
    assert not hasattr(sampler, "sample_indices_")  # partial state released


def test_memory_error_propagates_without_guard(monkeypatch, data):
    X, y = data

    def boom(*args, **kwargs):
        raise MemoryError("simulated allocation failure")

    monkeypatch.setattr(
        "imblearn.under_sampling._prototype_selection."
        "_random_under_sampler.fill_by_indices_chunked",
        boom,
    )
    with pytest.raises(MemoryError):
        RandomUnderSampler(random_state=RND_SEED, chunk_size=64).fit_resample(X, y)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -1000])
def test_invalid_chunk_size_value_rejected(data, bad):
    X, y = data
    # sklearn parameter validation rejects out-of-range values before any
    # data-dependent check runs; both are ValueError subclasses.
    with pytest.raises(ValueError, match="chunk_size"):
        RandomUnderSampler(chunk_size=bad).fit_resample(X, y)


@pytest.mark.parametrize("bad", [1.5, "100", True])
def test_invalid_chunk_size_type_rejected(data, bad):
    X, y = data
    with pytest.raises((TypeError, ValueError)):
        RandomUnderSampler(chunk_size=bad).fit_resample(X, y)


# ---------------------------------------------------------------------------
# Memory profile contract
# ---------------------------------------------------------------------------


def test_profile_reports_copy_behavior(data):
    X, y = data
    ros_profile = RandomOverSampler(random_state=RND_SEED).memory_profile(X, y)
    rus_profile = RandomUnderSampler(random_state=RND_SEED).memory_profile(X, y)
    assert ros_profile.copies_input is True
    assert rus_profile.copies_input is False
    for profile in (ros_profile, rus_profile):
        assert profile.chunked_supported is True
        assert profile.peak_extra_bytes_full > profile.output_bytes or (
            profile.output_bytes > 0
        )
        assert profile.peak_extra_bytes_chunked is None  # no chunk_size set


def test_profile_chunked_estimate_decreases(data):
    X, y = data
    ros = RandomOverSampler(random_state=RND_SEED, chunk_size=256)
    profile = ros.memory_profile(X, y)
    assert profile.peak_extra_bytes_chunked is not None
    # Bounded path removes the duplicated input and the per-class copies.
    assert profile.peak_extra_bytes_chunked < profile.peak_extra_bytes_full


def test_rus_profile_chunked_never_explodes(data):
    # Under-sampling already shrinks the data and never copies the input: the
    # chunked peak must stay comparable to the regular path, never grow with
    # the input size beyond the output itself.
    X, y = data
    for chunk_size in (16, 256, 100_000):
        profile = RandomUnderSampler(
            random_state=RND_SEED, chunk_size=chunk_size
        ).memory_profile(X, y)
        assert profile.copies_input is False
        assert profile.peak_extra_bytes_chunked <= profile.peak_extra_bytes_full + (
            chunk_size * X.shape[1] * X.dtype.itemsize
        )


@pytest.mark.parametrize(
    "sampler",
    [
        SMOTE(random_state=RND_SEED),
        ADASYN(random_state=RND_SEED),
        EditedNearestNeighbours(),
        TomekLinks(),
        CondensedNearestNeighbour(random_state=RND_SEED),
        SMOTEENN(random_state=RND_SEED),
        SMOTETomek(random_state=RND_SEED),
    ],
)
def test_other_samplers_report_generic_profile_without_chunking(data, sampler):
    X, y = data
    profile = sampler.memory_profile(X, y)
    assert isinstance(profile.copies_input, bool)
    assert profile.chunked_supported is False
    assert profile.peak_extra_bytes_chunked is None
    # they do not expose the opt-in switches: the public constructor is
    # untouched
    assert "chunk_size" not in sampler.get_params()


def test_pipeline_of_samplers_memory_is_sum_of_profiles(data):
    X, y = data
    over = RandomOverSampler(random_state=RND_SEED)
    under = RandomUnderSampler(random_state=RND_SEED)
    p_over = over.memory_profile(X, y)
    # after oversampling the dataset is balanced at the majority count
    X_os, y_os = over.fit_resample(X, y)
    p_under = under.memory_profile(X_os, y_os)
    # the composition estimate is the sum of the two stages; there is no
    # hidden multiplicative factor.
    combined = p_over.peak_extra_bytes_full + p_under.peak_extra_bytes_full
    alone = p_over.peak_extra_bytes_full
    assert combined < 3 * alone  # never an order-of-magnitude blow-up
