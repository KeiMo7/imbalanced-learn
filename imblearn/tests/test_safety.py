"""Tests for the opt-in resampling safety guards and provenance tracking."""

# Authors: The imbalanced-learn developers
# License: MIT

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTEN,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.pipeline import Pipeline
from imblearn.safety import (
    GuardedSampler,
    InfeasibleSamplingTargetError,
    ResamplingGuard,
    ResamplingLeakageError,
    ResamplingProvenance,
    check_sampling_ratio,
)
from imblearn.under_sampling import RandomUnderSampler

# SciPy 1.18 deprecates distance helpers used by the existing samplers; these
# pre-existing warnings are unrelated to the safety layer and are turned into
# errors by the repository-wide pytest configuration.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def data():
    X, y = make_classification(
        n_classes=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=10,
        n_clusters_per_class=1,
        n_samples=300,
        random_state=10,
    )
    return X, y


# ---------------------------------------------------------------------------
# 1. Normal resampling: guarded output is numerically identical to the raw
#    sampler, and lineage only ever contains integer identifiers.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sampler",
    [
        SMOTE(random_state=42),
        BorderlineSMOTE(random_state=42),
        SVMSMOTE(random_state=42),
        KMeansSMOTE(
            random_state=42, kmeans_estimator=10, cluster_balance_threshold=0.0
        ),
        RandomOverSampler(random_state=42),
        ADASYN(random_state=42),
        RandomUnderSampler(random_state=42),
    ],
)
def test_guarded_sampler_matches_raw_sampler(data, sampler):
    X, y = data
    raw_X, raw_y = type(sampler)(**sampler.get_params()).fit_resample(X, y)

    guard = ResamplingGuard()
    guarded = GuardedSampler(type(sampler)(**sampler.get_params()), guard=guard)
    with guard.fold():
        new_X, new_y = guarded.fit_resample(X, y)

    np.testing.assert_array_equal(raw_X, new_X)
    np.testing.assert_array_equal(raw_y, new_y)
    assert isinstance(guarded.provenance_, ResamplingProvenance)
    # provenance never leaks feature values: parents are integer tuples only
    parents = guarded.provenance_.parent_sample_ids
    if parents is not None:
        output_positions = sorted(parents)
        # lineage keys are output positions of *synthetic* rows: they start
        # right after the original rows and never collide with input rows
        if guarded.provenance_.support == "full":
            assert output_positions[0] == X.shape[0]
            assert output_positions[-1] == new_X.shape[0] - 1
        for positions in parents.values():
            assert all(isinstance(pos, (int, np.integer)) for pos in positions)
            assert all(0 <= pos < X.shape[0] for pos in positions)


def test_smoten_provenance_on_categorical_data(data):
    X, y = data
    X_cat = np.random.RandomState(0).choice(list("ABCD"), size=(X.shape[0], 3))
    guard = ResamplingGuard()
    guarded = GuardedSampler(SMOTEN(random_state=0), guard=guard)
    with guard.fold():
        X_res, y_res = guarded.fit_resample(X_cat, y)
    provenance = guarded.provenance_
    assert provenance.support == "full"
    assert X_res.shape[0] == provenance.n_output_samples
    # every referenced parent is a valid *input* row identifier
    for positions in provenance.parent_sample_ids.values():
        assert all(0 <= pos < X_cat.shape[0] for pos in positions)


def test_provenance_to_dict_is_json_like_and_feature_free(data):
    import json

    X, y = data
    guard = ResamplingGuard()
    guarded = GuardedSampler(SMOTE(random_state=0), guard=guard)
    with guard.fold():
        guarded.fit_resample(X, y)
    payload = guarded.provenance_.to_dict()
    # must round-trip through JSON (no arrays / feature values inside)
    json.dumps(payload)
    assert payload["sampler"] == "SMOTE"
    assert payload["support"] == "full"
    assert isinstance(payload["parent_sample_ids"], dict)


# ---------------------------------------------------------------------------
# 2. Misuse: resampling a validation / test set raises an explicit,
#    actionable error naming the offending step.
# ---------------------------------------------------------------------------
def test_resampling_validation_set_raises(data):
    X, y = data
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=0)
    guard = ResamplingGuard()
    guard.register_validation(X_val, name="X_val")
    guarded = GuardedSampler(SMOTE(random_state=0), guard=guard)

    with pytest.raises(ResamplingLeakageError, match="X_val") as exc_info:
        guarded.fit_resample(X_val, y_val)
    # the message names both the role and the interception point
    assert "validation" in str(exc_info.value)
    assert "fit_resample" in str(exc_info.value)
    assert "SMOTE" in str(exc_info.value)


def test_resampling_test_set_raises(data):
    X, y = data
    guard = ResamplingGuard()
    guard.register_test(X, name="held_out")
    guarded = GuardedSampler(RandomOverSampler(random_state=0), guard=guard)
    with pytest.raises(ResamplingLeakageError, match="held_out"):
        guarded.fit_resample(X, y)


def test_unknown_array_defaults_to_train_and_is_allowed(data):
    # backward-compatible behaviour: unregistered arrays are treated as train
    X, y = data
    guard = ResamplingGuard()
    guarded = GuardedSampler(SMOTE(random_state=0), guard=guard)
    with guard.fold():
        X_res, y_res = guarded.fit_resample(X, y)
    assert X_res.shape[0] > X.shape[0]


def test_standalone_use_without_fold_still_records_provenance(data):
    # a GuardedSampler used outside of guard.fold() still collects lineage on
    # itself and on the guard, instead of silently downgrading support
    X, y = data
    guard = ResamplingGuard()
    guarded = GuardedSampler(SMOTE(random_state=0), guard=guard)
    X_res, _ = guarded.fit_resample(X, y)
    assert guarded.provenance_.support == "full"
    assert len(guard.provenance_records_) == 1
    assert guarded.provenance_ is guard.provenance_records_[0]
    assert X_res.shape[0] == guarded.provenance_.n_output_samples


def test_fit_on_protected_array_also_blocked(data):
    X, y = data
    guard = ResamplingGuard()
    guard.register_validation(X, name="X_val")
    guarded = GuardedSampler(SMOTE(random_state=0), guard=guard)
    with pytest.raises(ResamplingLeakageError, match="fit"):
        guarded.fit(X, y)


# ---------------------------------------------------------------------------
# 3. Cross-validation: each fold resamples independently; synthetic samples
#    (and their lineage) from one fold never reach the next.
# ---------------------------------------------------------------------------
def test_cross_validation_folds_are_independent(data):
    X, y = data
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    guard = ResamplingGuard()

    seen_records = 0
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        with guard.fold() as sink:
            guarded = GuardedSampler(SMOTE(random_state=42), guard=guard)
            X_res, y_res = guarded.fit_resample(X[train_idx], y[train_idx])

            # one record per fold, scoped to the fold context
            assert len(sink.records) == 1
            # every parent identifier resolves inside *this* training fold
            for positions in guarded.provenance_.parent_sample_ids.values():
                assert all(pos < train_idx.size for pos in positions)
            # synthetic outputs stay within the fold's training population
            assert X_res.shape[0] <= 2 * train_idx.size
        seen_records += 1

    assert seen_records == 4
    assert len(guard.provenance_records_) == 4
    # record ids are unique: lineage cannot be silently shared across folds
    ids = [record.record_id for record in guard.provenance_records_]
    assert len(set(ids)) == 4


def test_guarded_sampler_works_inside_pipeline(data):
    X, y = data
    guard = ResamplingGuard()
    pipe = Pipeline(
        [
            ("smote", GuardedSampler(SMOTE(random_state=42), guard=guard)),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    scores = cross_validate(pipe, X, y, cv=skf, error_score="raise")
    assert scores["test_score"].shape == (3,)
    # one provenance record per fold fit
    assert len(guard.provenance_records_) == 3


# ---------------------------------------------------------------------------
# 4. Target ratio: feasibility pre-check and actual-vs-target accounting.
# ---------------------------------------------------------------------------
def test_infeasible_undersampling_insufficient_samples(data):
    X, y = data
    with pytest.raises(InfeasibleSamplingTargetError) as exc_info:
        # class 1 has 270 rows, asking for 5000 of them is impossible
        check_sampling_ratio({1: 5000}, y, "under-sampling")
    assert exc_info.value.reason == "insufficient_samples"


def test_infeasible_oversampling_parameter_conflict(data):
    X, y = data
    with pytest.raises(InfeasibleSamplingTargetError) as exc_info:
        # over-sampling cannot shrink a class below its current size
        check_sampling_ratio({0: 5}, y, "over-sampling")
    assert exc_info.value.reason == "parameter_conflict"


def test_feasibility_report_without_raising(data):
    X, y = data
    report = check_sampling_ratio(
        {1: 5000}, y, "under-sampling", raise_on_infeasible=False
    )
    assert report.feasible is False
    assert report.reason == "insufficient_samples"


def test_actual_ratio_matches_target(data):
    X, y = data
    guard = ResamplingGuard()
    guarded = GuardedSampler(SMOTE(random_state=0), guard=guard)
    with guard.fold():
        X_res, y_res = guarded.fit_resample(X, y)
    report = check_sampling_ratio("auto", y, "over-sampling").record_actual(y_res)
    assert report.feasible is True
    # the auto target balances the minority class onto the majority
    assert report.deviations == {np.unique(y)[0]: 0}
    assert all(value == 0 for value in report.deviations.values())


# ---------------------------------------------------------------------------
# Reproducibility: fixed seed + identical split => identical resampling.
# ---------------------------------------------------------------------------
def test_fixed_seed_gives_identical_results(data):
    X, y = data
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    train_idx = next(iter(skf.split(X, y)))[0]

    def run():
        guard = ResamplingGuard()
        guarded = GuardedSampler(SMOTE(random_state=7), guard=guard)
        with guard.fold():
            X_res, y_res = guarded.fit_resample(X[train_idx], y[train_idx])
        return X_res, y_res, guarded.provenance_.to_dict()

    (X1, y1, p1), (X2, y2, p2) = run(), run()
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    # lineage is reproducible as well, except for the unique record id
    p1.pop("record_id"), p2.pop("record_id")
    assert p1 == p2


# ---------------------------------------------------------------------------
# Hard constraints.
# ---------------------------------------------------------------------------
def test_input_arrays_are_not_modified(data):
    X, y = data
    X_before, y_before = X.copy(), y.copy()
    guard = ResamplingGuard()
    with guard.fold():
        GuardedSampler(SMOTE(random_state=0), guard=guard).fit_resample(X, y)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


def test_switch_off_behaviour_is_unchanged(data):
    # using the raw sampler, without any guard, behaves exactly as before
    X, y = data
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    assert X_res.shape == (540, 10)
    assert np.bincount(y_res).tolist() == [270, 270]


def test_unsupported_sampler_is_marked_not_fabricated(data):
    X, y = data
    guard = ResamplingGuard()
    guarded = GuardedSampler(ADASYN(random_state=42), guard=guard)
    with guard.fold():
        guarded.fit_resample(X, y)
    assert guarded.provenance_.support == "not_supported"
    assert guarded.provenance_.parent_sample_ids is None
    assert guarded.provenance_.supported is False


def test_interrupted_sampling_leaves_no_record(data, monkeypatch):
    X, y = data
    guard = ResamplingGuard()
    guarded = GuardedSampler(SMOTE(random_state=0), guard=guard)

    def boom(self, X, y):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(SMOTE, "_fit_resample", boom)
    with guard.fold():
        with pytest.raises(RuntimeError, match="simulated interruption"):
            guarded.fit_resample(X, y)
    # no half-built provenance, no fold record published
    assert guarded.provenance_ is None
    assert guard.provenance_records_ == []


def test_registration_uses_identity_not_contents(data):
    # two arrays with equal contents but different identities are distinct:
    # guarding one must not block the other, and no data hashing is involved.
    X, y = data
    X_twin = X.copy()
    guard = ResamplingGuard()
    guard.register_validation(X, name="original")
    allowed = GuardedSampler(SMOTE(random_state=0), guard=guard)
    with guard.fold():
        X_res, _ = allowed.fit_resample(X_twin, y)
    assert X_res.shape[0] > X.shape[0]


def test_guarded_sampler_clone_roundtrip(data):
    from sklearn.base import clone

    guard = ResamplingGuard()
    guarded = GuardedSampler(SMOTE(random_state=5), guard=guard)
    cloned = clone(guarded)
    assert isinstance(cloned.sampler, SMOTE)
    assert cloned.guard is guard
    X, y = data
    with guard.fold():
        cloned.fit_resample(X, y)
    assert cloned.provenance_.sampler == "SMOTE"


def test_guard_rejects_unknown_role(data):
    guard = ResamplingGuard()
    with pytest.raises(ValueError, match="role"):
        guard.register(data[0], "production")
