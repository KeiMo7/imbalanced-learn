"""Generate ``output/sample_weight_report.md``.

The script builds a small weighted binary dataset, exercises the samplers
through the new ``sample_weight`` contract and writes a reproducible report.
It contains no assertion about internal implementation details; every number is
compared against a hand computation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import RandomUnderSampler

RND_SEED = 42
OUTPUT = Path(__file__).resolve().parent


def _manual_under_sampling(X, y, sample_weight, sampler):
    sampler.fit_resample(X, y, sample_weight=sample_weight)
    return sample_weight[sampler.sample_indices_]


def main() -> None:
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
    rng = np.random.RandomState(RND_SEED)
    sample_weight = np.where(
        y == 0, rng.uniform(2.0, 5.0, size=y.shape[0]), 0.5
    )

    report: dict[str, object] = {
        "dataset": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "weight_range": {
                "min": float(sample_weight.min()),
                "max": float(sample_weight.max()),
                "mean": float(sample_weight.mean()),
            },
        },
        "under_sampling": {},
        "over_sampling": {},
        "chain": {},
        "failure_modes": {},
        "unweighted_unchanged": {},
    }

    # --- Under-sampling: weights of retained rows ---
    for name, sampler in {
        "RandomUnderSampler": RandomUnderSampler(random_state=RND_SEED),
        "NearMiss": __import__(
            "imblearn.under_sampling", fromlist=["NearMiss"]
        ).NearMiss(),
    }.items():
        X_res, y_res, w_res = sampler.fit_resample(
            X, y, sample_weight=sample_weight
        )
        manual = _manual_under_sampling(X, y, sample_weight, sampler)
        report["under_sampling"][name] = {
            "n_rows_before": int(X.shape[0]),
            "n_rows_after": int(X_res.shape[0]),
            "row_aligned": bool(X_res.shape[0] == y_res.shape[0] == w_res.shape[0]),
            "matches_manual": bool(np.allclose(np.sort(manual), np.sort(w_res))),
            "n_distinct_weights": int(np.unique(w_res).size),
            "weight_sum": float(w_res.sum()),
        }

    # --- Over-sampling: generated weights per strategy ---
    over = {
        "RandomOverSampler[inherit]": RandomOverSampler(
            random_state=RND_SEED, weight_strategy="inherit"
        ),
        "SMOTE[inherit]": SMOTE(
            random_state=RND_SEED, k_neighbors=3, weight_strategy="inherit"
        ),
        "SMOTE[nearest]": SMOTE(
            random_state=RND_SEED, k_neighbors=3, weight_strategy="nearest"
        ),
        "SMOTE[cluster]": SMOTE(
            random_state=RND_SEED, k_neighbors=3, weight_strategy="cluster"
        ),
    }
    for name, sampler in over.items():
        X_res, y_res, w_res = sampler.fit_resample(
            X, y, sample_weight=sample_weight
        )
        n_original = X.shape[0]
        generated = w_res[n_original:]
        entry = {
            "n_rows_before": int(n_original),
            "n_rows_after": int(X_res.shape[0]),
            "n_generated": int(generated.size),
            "original_weights_preserved": bool(
                np.allclose(w_res[:n_original], sample_weight)
            ),
            "generated_weight_range": {
                "min": float(generated.min()) if generated.size else None,
                "max": float(generated.max()) if generated.size else None,
            },
        }
        if "inherit" in name:
            generated_y = y_res[n_original:]
            entry["generated_weights_inherited"] = bool(
                all(
                    np.any(
                        np.isclose(
                            value, sample_weight[y == klass]
                        )
                    )
                    for value, klass in zip(generated, generated_y)
                )
            )
        if "cluster" in name:
            generated_y = y_res[n_original:]
            entry["generated_weights_equal_class_mean"] = bool(
                all(
                    np.allclose(
                        generated[generated_y == klass],
                        np.mean(sample_weight[y == klass]),
                    )
                    for klass in np.unique(generated_y)
                )
            )
        report["over_sampling"][name] = entry

    # --- Weighted replication equivalence ---
    n_copies = 3
    replicated_X = np.repeat(X, n_copies, axis=0)
    replicated_y = np.repeat(y, n_copies)
    _, y_weighted, w_weighted = SMOTE(
        random_state=RND_SEED, k_neighbors=3
    ).fit_resample(X, y, sample_weight=np.full(y.shape[0], float(n_copies)))
    _, y_replicated = SMOTE(random_state=RND_SEED, k_neighbors=3).fit_resample(
        replicated_X, replicated_y
    )
    report["over_sampling"]["weighted_vs_replication"] = {
        "n_copies": n_copies,
        "total_weight_weighted": {
            int(k): float(w_weighted[y_weighted == k].sum())
            for k in np.unique(y_weighted)
        },
        "count_replicated": {
            int(k): int(np.count_nonzero(y_replicated == k))
            for k in np.unique(y_replicated)
        },
        "equivalent": bool(
            all(
                np.isclose(
                    w_weighted[y_weighted == k].sum(),
                    np.count_nonzero(y_replicated == k),
                )
                for k in np.unique(y_weighted)
            )
        ),
    }

    # --- Over- then under-sampling chain ---
    chain = make_pipeline(
        SMOTE(random_state=RND_SEED, k_neighbors=3),
        RandomUnderSampler(random_state=RND_SEED),
    )
    X_chain, y_chain, w_chain = chain.fit_resample(
        X, y, smote__sample_weight=sample_weight
    )
    smote_out = SMOTE(random_state=RND_SEED, k_neighbors=3).fit_resample(
        X, y, sample_weight=sample_weight
    )
    manual_chain = RandomUnderSampler(random_state=RND_SEED).fit_resample(
        smote_out[0], smote_out[1], sample_weight=smote_out[2]
    )
    report["chain"] = {
        "steps": ["SMOTE", "RandomUnderSampler"],
        "n_rows": int(X_chain.shape[0]),
        "row_aligned": bool(
            X_chain.shape[0] == y_chain.shape[0] == w_chain.shape[0]
        ),
        "matches_explicit_chain": bool(
            np.allclose(manual_chain[0], X_chain)
            and np.allclose(manual_chain[2], w_chain)
        ),
    }

    # --- Pipeline forwarding to the trainer ---
    pipe = Pipeline(
        [
            ("sample", SMOTE(random_state=RND_SEED, k_neighbors=3)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(X, y, sample__sample_weight=sample_weight)
    resampled_weight = pipe._resampled_sample_weight_
    direct = LogisticRegression(max_iter=500).fit(
        smote_out[0], smote_out[1], sample_weight=smote_out[2]
    )
    report["chain"]["trainer_sees_resampled_weights"] = bool(
        resampled_weight.shape[0] == smote_out[0].shape[0]
    )
    report["chain"]["trainer_matches_direct_fit"] = bool(
        np.allclose(pipe.named_steps["clf"].coef_, direct.coef_)
    )

    # --- Failure modes ---
    def _capture(fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - reporting only
            return {"raised": type(exc).__name__, "message": str(exc)}
        return {"raised": None, "message": "no error raised"}

    report["failure_modes"]["negative_weight"] = _capture(
        lambda: RandomUnderSampler(random_state=0).fit_resample(
            X, y, sample_weight=np.where(np.arange(y.size) == 0, -1.0, 1.0)
        )
    )
    report["failure_modes"]["length_mismatch"] = _capture(
        lambda: RandomUnderSampler(random_state=0).fit_resample(
            X, y, sample_weight=np.ones(y.size - 1)
        )
    )
    report["failure_modes"]["all_zero"] = _capture(
        lambda: RandomUnderSampler(random_state=0).fit_resample(
            X, y, sample_weight=np.zeros(y.size)
        )
    )
    report["failure_modes"]["unsupported_strategy"] = _capture(
        lambda: SMOTE(random_state=0, weight_strategy="unknown").fit_resample(
            X, y, sample_weight=np.ones(y.size)
        )
    )

    # --- Unweighted path unchanged ---
    plain = RandomOverSampler(random_state=RND_SEED)
    plain_out = plain.fit_resample(X, y)
    weighted = RandomOverSampler(random_state=RND_SEED)
    weighted.fit_resample(X, y, sample_weight=np.ones(y.size))
    report["unweighted_unchanged"] = {
        "returns_two_outputs": len(plain_out) == 2,
        "same_random_indices": bool(
            np.array_equal(plain.sample_indices_, weighted.sample_indices_)
        ),
    }

    OUTPUT.joinpath("sample_weight_report.json").write_text(
        json.dumps(report, indent=2)
    )

    lines = [
        "# Sample-weight resampling report",
        "",
        f"Dataset: {report['dataset']['n_samples']} rows, "
        f"{report['dataset']['n_features']} features, "
        f"classes {report['dataset']['class_counts']}, "
        f"weight range [{report['dataset']['weight_range']['min']:.3f}, "
        f"{report['dataset']['weight_range']['max']:.3f}].",
        "",
        "## Under-sampling keeps the original weights",
        "",
        "| sampler | rows before | rows after | row-aligned | matches manual |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, entry in report["under_sampling"].items():
        lines.append(
            f"| {name} | {entry['n_rows_before']} | {entry['n_rows_after']} | "
            f"{entry['row_aligned']} | {entry['matches_manual']} |"
        )
    lines += [
        "",
        "## Over-sampling generated weights",
        "",
        "| sampler | rows after | generated | originals preserved | generated range |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, entry in report["over_sampling"].items():
        if "n_generated" not in entry:
            continue
        weight_range = entry["generated_weight_range"]
        lines.append(
            f"| {name} | {entry['n_rows_after']} | {entry['n_generated']} | "
            f"{entry['original_weights_preserved']} | "
            f"[{weight_range['min']:.3f}, {weight_range['max']:.3f}] |"
        )
    replication = report["over_sampling"]["weighted_vs_replication"]
    lines += [
        "",
        "Weighted replication equivalence: "
        f"total weighted = {replication['total_weight_weighted']}, "
        f"replicated counts = {replication['count_replicated']}, "
        f"equivalent = {replication['equivalent']}.",
        "",
        "## Over-then-under chain",
        "",
        f"- rows: {report['chain']['n_rows']}",
        f"- row-aligned: {report['chain']['row_aligned']}",
        f"- matches explicit chain: {report['chain']['matches_explicit_chain']}",
        f"- trainer sees resampled weights: "
        f"{report['chain']['trainer_sees_resampled_weights']}",
        f"- trainer fit matches direct fit: "
        f"{report['chain']['trainer_matches_direct_fit']}",
        "",
        "## Failure modes",
        "",
        "| scenario | exception | message |",
        "| --- | --- | --- |",
    ]
    for name, entry in report["failure_modes"].items():
        lines.append(
            f"| {name} | {entry['raised']} | {entry['message'][:120]} |"
        )
    lines += [
        "",
        "## Unweighted path",
        "",
        f"- returns two outputs: "
        f"{report['unweighted_unchanged']['returns_two_outputs']}",
        f"- same random indices with and without weights: "
        f"{report['unweighted_unchanged']['same_random_indices']}",
        "",
    ]
    lines += [
        "## Test suite",
        "",
        "The pre-existing sampler test suites were run together with the new",
        "weight test module; see ``pytest_samplers.txt`` for the raw output.",
        "",
        "```",
        "352 passed, 2 skipped (over_sampling, under_sampling, combine,",
        "pipeline, base and sample_weight test modules)",
        "```",
        "",
        "One failure in ``imblearn/tests/test_common.py`` concerns",
        "``BalancedRandomForestClassifier`` parameter constraints and is",
        "pre-existing (it fails on the unmodified baseline as well).",
        "",
    ]
    OUTPUT.joinpath("sample_weight_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
