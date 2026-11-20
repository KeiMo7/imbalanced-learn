"""Generate before/after output comparison for the memory feature.

The "before" results are produced from the pristine implementation stored in
git (``git show HEAD:<path>``), imported under isolated module names so that
the modified working-tree package is never shadowed. The "after" results come
from the current working tree.

The comparison is written to
``artifacts/memory_benchmark/before_after_comparison.json`` and a short
human-readable digest is printed. Run with the project virtualenv::

    .venv/bin/python maint_tools/memory_benchmark/compare_before_after.py
"""

# License: MIT

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "memory_benchmark" / "before_after_comparison.json"


def _load_module_from_source(name: str, source: str) -> types.ModuleType:
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = f"<git-head:{name}>.py"
    exec(compile(source, f"<git-head:{name}>", "exec"), module.__dict__)
    return module


def _git_head_module(rel_path: str, alias: str) -> types.ModuleType:
    import subprocess

    source = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"HEAD:{rel_path}"], text=True
    )
    return _load_module_from_source(alias, source)


def _old_samplers():
    """Rebuild the two historical samplers from git HEAD sources.

    The old modules import sibling packages through their real package paths,
    so only the two sampler modules themselves are replaced; every dependency
    comes from the installed (working tree) package.
    """
    import imblearn.over_sampling._random_over_sampler as new_ros
    import imblearn.under_sampling._prototype_selection._random_under_sampler as new_rus

    old_ros_mod = _git_head_module(
        "imblearn/over_sampling/_random_over_sampler.py",
        "_old_random_over_sampler",
    )
    old_rus_mod = _git_head_module(
        "imblearn/under_sampling/_prototype_selection/_random_under_sampler.py",
        "_old_random_under_sampler",
    )
    return (
        old_ros_mod.RandomOverSampler,
        old_rus_mod.RandomUnderSampler,
        new_ros.RandomOverSampler,
        new_rus.RandomUnderSampler,
    )


def _digest(arr: np.ndarray) -> dict:
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "first_row": np.asarray(flat[0]).tolist(),
        "last_row": np.asarray(flat[-1]).tolist(),
        "checksum": int(np.bitwise_xor.reduce(flat.view(np.uint64).ravel()))
        if flat.dtype.kind == "f"
        else int(np.sum(flat.view(np.uint64).ravel())),
    }


def main() -> int:
    OldROS, OldRUS, NewROS, NewRUS = _old_samplers()

    X, y = make_classification(
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.15, 0.35, 0.5],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=6,
        n_samples=500,
        random_state=7,
    )

    cases = [
        (
            "RandomOverSampler",
            OldROS(random_state=42),
            NewROS(random_state=42),
            NewROS(random_state=42, chunk_size=37),
        ),
        (
            "RandomUnderSampler",
            OldRUS(random_state=42),
            NewRUS(random_state=42),
            NewRUS(random_state=42, chunk_size=37),
        ),
        (
            "RandomUnderSampler-replacement",
            OldRUS(random_state=42, replacement=True),
            NewRUS(random_state=42, replacement=True),
            NewRUS(random_state=42, replacement=True, chunk_size=37),
        ),
    ]

    report = {"input": _digest(X), "cases": []}
    all_equal = True
    for name, old, new_full, new_chunk in cases:
        X_old, y_old = old.fit_resample(X, y)
        X_new, y_new = new_full.fit_resample(X, y)
        X_chk, y_chk = new_chunk.fit_resample(X, y)
        equal_full = np.array_equal(X_old, X_new) and np.array_equal(y_old, y_new)
        equal_chunk = np.array_equal(X_old, X_chk) and np.array_equal(y_old, y_chk)
        all_equal &= equal_full and equal_chunk
        case = {
            "name": name,
            "before": {"X": _digest(X_old), "y": _digest(y_old)},
            "after_full": {"X": _digest(X_new), "y": _digest(y_new)},
            "after_chunked": {"X": _digest(X_chk), "y": _digest(y_chk)},
            "full_path_identical": bool(equal_full),
            "chunked_path_identical": bool(equal_chunk),
            "sample_indices_equal": bool(
                np.array_equal(old.sample_indices_, new_full.sample_indices_)
                and np.array_equal(old.sample_indices_, new_chunk.sample_indices_)
            ),
        }
        report["cases"].append(case)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print(f"report written to {OUT.relative_to(ROOT)}")
    for case in report["cases"]:
        print(
            f"{case['name']}: full_identical={case['full_path_identical']} "
            f"chunked_identical={case['chunked_path_identical']} "
            f"indices_identical={case['sample_indices_equal']} "
            f"shape={case['before']['X']['shape']}"
        )
    return 0 if all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
