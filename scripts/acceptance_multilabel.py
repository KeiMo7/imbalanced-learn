"""Acceptance checks for :class:`~imblearn.over_sampling.MultiLabelSMOTE`.

The script exercises the multi-label resampling requirements end to end and
writes a machine-readable summary, a human-readable summary and a Markdown
report into ``output/`` at the repository root.  Run it with::

    python scripts/acceptance_multilabel.py

It is deliberately dependency-free beyond numpy/scipy/scikit-learn so that the
results are reproducible on any checkout.
"""

# Authors: imbalanced-learn developers
# License: MIT

import json
import tracemalloc
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import sparse

from imblearn.over_sampling import MultiLabelSMOTE
from imblearn.over_sampling._multilabel import (
    group_by_label_set,
    label_sets,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def make_long_tail():
    """Long-tail dataset with a rare multi-label combination."""
    rng = np.random.RandomState(0)
    n, n_labels, n_features = 62, 4, 5
    X = rng.randn(n, n_features)
    y = np.zeros((n, n_labels), dtype=np.int64)
    y[0:30, 0] = 1  # (0,)      x30  majority
    y[30:50, 1] = 1  # (1,)      x20
    y[50:58, 0] = 1  # (0, 2)    x8   rare multi-label combination
    y[50:58, 2] = 1
    y[58, 3] = 1  # (3,)      x1  singleton combination
    # rows 59-61 stay all-zero
    return X, y


def label_set_counts(y):
    counts = OrderedDict()
    for label_set in label_sets(y):
        counts[label_set] = counts.get(label_set, 0) + 1
    return counts


def as_inputs(X, y, sparse_input):
    if sparse_input:
        return sparse.csr_matrix(X), sparse.csr_matrix(y)
    return X, y


def main():
    warnings.simplefilter("ignore", UserWarning)
    report = OrderedDict()
    report["seed"] = SEED
    X, y = make_long_tail()
    train_counts = label_set_counts(y)

    # ---- 1. rare combination lifted to the budget, dense and sparse -------
    grown = {}
    for mode, sparse_input in (("dense", False), ("sparse", True)):
        X_in, y_in = as_inputs(X, y, sparse_input)
        sampler = MultiLabelSMOTE(random_state=SEED)
        X_res, y_res = sampler.fit_resample(X_in, y_in)
        out_counts = label_set_counts(
            y_res.toarray() if sparse.issparse(y_res) else y_res
        )
        grew = {
            str(label_set): {
                "before": int(train_counts[label_set]),
                "after": int(out_counts[label_set]),
            }
            for label_set in train_counts
        }
        grown[mode] = {
            "input_sparse": sparse_input,
            "output_target_sparse": bool(sparse.issparse(y_res)),
            "n_rows_in": int(X.shape[0]),
            "n_rows_out": int(y_res.shape[0]),
            "counts_per_label_set": grew,
            "synthetic_per_label_set": {
                str(k): int(v) for k, v in sampler.n_synthetic_.items()
            },
        }
    report["1_rare_combination_grown"] = grown

    # ---- 2. every synthetic label set exists in training -------------------
    synthetic_checks = {}
    for mode, sparse_input in (("dense", False), ("sparse", True)):
        X_in, y_in = as_inputs(X, y, sparse_input)
        X_res, y_res = MultiLabelSMOTE(random_state=SEED).fit_resample(X_in, y_in)
        y_dense = y_res.toarray() if sparse.issparse(y_res) else y_res
        synthetic = y_dense[X.shape[0] :]
        synthetic_sets = set(label_sets(synthetic))
        assert synthetic_sets.issubset(set(train_counts)), synthetic_sets
        assert all(len(label_set) > 0 for label_set in synthetic_sets)
        assert set(np.unique(synthetic).tolist()).issubset({0, 1})
        assert int(synthetic.max(axis=0).sum()) <= y.shape[1]
        synthetic_checks[mode] = {
            "all_seen_in_training": True,
            "no_all_zero_synthetic": True,
            "no_out_of_range_label": True,
            "synthetic_label_sets": [
                list(label_set) for label_set in sorted(synthetic_sets)
            ],
        }
    report["2_synthetic_label_sets_valid"] = synthetic_checks

    # ---- 3. fixed seed: two runs are bit-for-bit identical -----------------
    determinism = {}
    for mode, sparse_input in (("dense", False), ("sparse", True)):
        X_in, y_in = as_inputs(X, y, sparse_input)
        first = MultiLabelSMOTE(random_state=SEED).fit_resample(X_in, y_in)
        second = MultiLabelSMOTE(random_state=SEED).fit_resample(X_in, y_in)
        if sparse_input:
            X_equal = bool((first[0] != second[0]).nnz == 0)
            y_equal = bool((first[1] != second[1]).nnz == 0)
            X_cmp = np.array_equal(first[0].toarray(), second[0].toarray())
            y_cmp = np.array_equal(first[1].toarray(), second[1].toarray())
        else:
            X_equal = bool(np.array_equal(first[0], second[0]))
            y_equal = bool(np.array_equal(first[1], second[1]))
            X_cmp, y_cmp = X_equal, y_equal
        determinism[mode] = {
            "X_identical": X_equal,
            "y_identical": y_equal,
            "X_elementwise_equal": bool(X_cmp),
            "y_elementwise_equal": bool(y_cmp),
        }
    report["3_determinism"] = determinism

    # ---- 4. budgets -------------------------------------------------------
    total = MultiLabelSMOTE(random_state=SEED, total_budget=10)
    Xr, yr = total.fit_resample(X, y)
    per_label = MultiLabelSMOTE(
        random_state=SEED, label_budget={2: 5}, enforce_label_budget="warn"
    )
    Xr2, yr2 = per_label.fit_resample(X, y)
    report["4_budgets"] = {
        "total_budget": {
            "cap": 10,
            "synthetic_generated": int(sum(total.n_synthetic_.values())),
            "rows_out": int(yr.shape[0]),
        },
        "label_budget": {
            "cap_for_label_2": 5,
            "synthetic_with_label_2": int(yr2[X.shape[0] :, 2].sum()),
            "reported_shortfall": {
                str(k): int(v) for k, v in per_label.label_budget_shortfall_.items()
            },
        },
    }

    # ---- 5. singleton policy ---------------------------------------------
    singleton = {}
    for strategy in ("keep", "merge", "skip"):
        sampler = MultiLabelSMOTE(random_state=SEED, singleton_strategy=strategy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            Xr, yr = sampler.fit_resample(X, y)
        synthetic_sets = set(label_sets(yr[X.shape[0] :]))
        singleton[strategy] = {
            "rows_out": int(yr.shape[0]),
            "synthetic_sets": [list(s) for s in sorted(synthetic_sets)],
            "n_synthetic_singleton": int(sampler.n_synthetic_[(3,)]),
        }
    report["5_singleton_policy"] = singleton

    # ---- 6. sparse memory scales with nnz, not n_labels ** 2 --------------
    rng = np.random.RandomState(7)
    n = 5_000
    scaling = {}
    for n_labels in (20, 200, 2000):
        rows = np.repeat(np.arange(n), 2)
        cols = rng.randint(0, n_labels, size=n * 2)
        yy = sparse.csr_matrix(
            (np.ones(n * 2, dtype=np.int8), (rows, cols)), shape=(n, n_labels)
        )
        yy.sum_duplicates()
        yy.data[:] = 1
        yy.sort_indices()
        groups, counts = group_by_label_set(yy)
        scaling[n_labels] = {
            "n_distinct_combinations": len(counts),
            "max_labels_per_tuple": max(len(s) for s in counts),
            "nnz": int(yy.nnz),
        }
    # A one-million-row, 50-label sparse target is grouped within a bounded
    # peak memory: label sets are read from CSR arrays, never densified.
    n_big, n_labels_big = 1_000_000, 50
    rows = np.repeat(np.arange(n_big), 3)
    cols = rng.randint(0, n_labels_big, size=n_big * 3)
    big = sparse.csr_matrix(
        (np.ones(n_big * 3, dtype=np.int8), (rows, cols)),
        shape=(n_big, n_labels_big),
    )
    big.sum_duplicates()
    big.data[:] = 1
    big.sort_indices()
    tracemalloc.start()
    big_groups, big_counts = group_by_label_set(big)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report["6_sparse_memory"] = {
        "scaling": scaling,
        "million_rows": {
            "n_rows": n_big,
            "n_labels": n_labels_big,
            "nnz": int(big.nnz),
            "n_distinct_combinations": len(big_counts),
            "max_labels_per_tuple": max(len(s) for s in big_counts),
            "peak_memory_mb": round(peak / 1e6, 1),
        },
    }

    # ---- 7. downstream consumption ---------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    X_res, y_res = MultiLabelSMOTE(random_state=SEED).fit_resample(
        sparse.csr_matrix(X), sparse.csr_matrix(y)
    )
    clf = OneVsRestClassifier(LogisticRegression(max_iter=200)).fit(X_res, y_res)
    ovr_shape = list(clf.predict(X[:3]).shape)
    dense_sampler = MultiLabelSMOTE(random_state=SEED, output_sparse=False)
    X_res_dense, y_dense = dense_sampler.fit_resample(
        sparse.csr_matrix(X), sparse.csr_matrix(y)
    )
    clf_dense = MultiOutputClassifier(LogisticRegression(max_iter=200)).fit(
        X_res_dense, y_dense
    )
    report["7_downstream"] = {
        "sparse_target_type": type(y_res).__name__,
        "sparse_target_accepted_by_one_vs_rest": ovr_shape,
        "output_sparse_false_is_dense": isinstance(y_dense, np.ndarray),
        "dense_target_predict_shape": list(clf_dense.predict(X[:3]).shape),
    }

    # ---- write the outputs ----------------------------------------------
    (OUTPUT_DIR / "multilabel_acceptance.json").write_text(json.dumps(report, indent=2))

    rare = grown["dense"]["counts_per_label_set"]["(0, 2)"]
    counts_equal = (
        grown["dense"]["counts_per_label_set"]
        == grown["sparse"]["counts_per_label_set"]
    )
    lines = [
        "MultiLabelSMOTE acceptance summary",
        "=" * 40,
        f"seed: {SEED}",
        "",
        "1) Rare combination (0, 2) grown from "
        f"{rare['before']} to {rare['after']} "
        "(dense and sparse agree; counts equal: "
        f"{counts_equal})",
        "   synthetic per label set: " f"{grown['dense']['synthetic_per_label_set']}",
        "2) Synthetic label sets all seen in training: "
        f"{synthetic_checks['dense']['all_seen_in_training']}; "
        "label sets: "
        f"{synthetic_checks['dense']['synthetic_label_sets']}",
        "3) Determinism (random_state=42, two independent runs):",
        f"   dense  X={determinism['dense']['X_identical']} "
        f"y={determinism['dense']['y_identical']}",
        f"   sparse X={determinism['sparse']['X_identical']} "
        f"y={determinism['sparse']['y_identical']}",
        "4) total_budget=10 -> "
        f"{report['4_budgets']['total_budget']['synthetic_generated']} synthetic; "
        "label_budget={2: 5} -> "
        f"{report['4_budgets']['label_budget']['synthetic_with_label_2']} "
        "synthetic rows carrying label 2, shortfall "
        f"{report['4_budgets']['label_budget']['reported_shortfall']}",
        "5) singleton (3,) synthetic counts: "
        f"keep={singleton['keep']['n_synthetic_singleton']}, "
        f"merge={singleton['merge']['n_synthetic_singleton']}, "
        f"skip={singleton['skip']['n_synthetic_singleton']}",
        "6) Sparse grouping: max labels per tuple stays "
        f"{max(v['max_labels_per_tuple'] for v in scaling.values())} as "
        "n_labels grows 20 -> 200 -> 2000; one million rows x 50 labels "
        "grouped with peak "
        f"{report['6_sparse_memory']['million_rows']['peak_memory_mb']} MB",
        "7) Downstream: sparse target accepted by OneVsRestClassifier "
        f"(predict shape {ovr_shape}); output_sparse=False gives a dense "
        "target (predict shape "
        f"{report['7_downstream']['dense_target_predict_shape']})",
    ]
    (OUTPUT_DIR / "multilabel_acceptance_summary.txt").write_text(
        "\n".join(lines) + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
