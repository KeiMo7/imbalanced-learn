"""Time and peak-memory benchmark of regular vs chunked resampling.

Each configuration runs in its own subprocess so that ``ru_maxrss`` reflects a
single, clean run. A "baseline" process only builds the input dataset; the
extra peak memory attributable to the sampler is the difference between the
resampling process peak and the baseline peak.

Results are written to ``artifacts/memory_benchmark/benchmark.json`` and a
markdown table to ``benchmark.md``. Run with::

    .venv/bin/python maint_tools/memory_benchmark/benchmark_chunking.py
"""

# License: MIT

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "artifacts" / "memory_benchmark" / "benchmark.json"
OUT_MD = ROOT / "artifacts" / "memory_benchmark" / "benchmark.md"

N_SAMPLES = 2_000_000
N_FEATURES = 20
SEED = 42
REPEATS = 3


def _build_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_classes=2,
        weights=[0.1, 0.9],
        n_informative=5,
        n_redundant=2,
        flip_y=0,
        n_features=N_FEATURES,
        n_clusters_per_class=1,
        n_samples=N_SAMPLES,
        random_state=SEED,
    )
    return X, y


def _run_worker(sampler_name: str, mode: str, chunk_size: int | None) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    X, y = _build_data()
    # warm the allocator and record the data-only peak baseline implicitly
    sampler_cls = (
        RandomOverSampler if sampler_name == "RandomOverSampler" else RandomUnderSampler
    )
    sampler = sampler_cls(random_state=SEED, chunk_size=chunk_size)

    start = time.perf_counter()
    X_res, y_res = sampler.fit_resample(X, y)
    elapsed = time.perf_counter() - start
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "elapsed_s": elapsed,
        "peak_rss_mib": peak_kib / 1024.0,
        "output_shape": list(X_res.shape),
        "input_mib": X.nbytes / (1024**2),
        "output_mib": X_res.nbytes / (1024**2),
        "mode": mode,
        "chunk_size": chunk_size,
        "sampler": sampler_name,
    }
    print("RESULT " + json.dumps(payload))


def _spawn(sampler_name: str, mode: str, chunk_size: int | None) -> dict:
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "worker",
            sampler_name,
            mode,
            str(chunk_size),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line.removeprefix("RESULT "))
    raise RuntimeError(f"worker produced no result: {proc.stdout}\n{proc.stderr}")


def _baseline_peak_mib() -> float:
    import subprocess

    code = (
        "import resource, sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from maint_tools.memory_benchmark.benchmark_chunking import _build_data; "
        "_build_data(); "
        "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    return float(proc.stdout.strip())


def _median(values: list[float]) -> float:
    return float(np.median(values))


def main() -> int:
    baseline_mib = _baseline_peak_mib()
    configs = [
        ("RandomOverSampler", "full", None),
        ("RandomOverSampler", "chunked", 100_000),
        ("RandomOverSampler", "chunked", 10_000),
        ("RandomUnderSampler", "full", None),
        ("RandomUnderSampler", "chunked", 100_000),
        ("RandomUnderSampler", "chunked", 10_000),
    ]
    rows = []
    for sampler_name, mode, chunk_size in configs:
        runs = [_spawn(sampler_name, mode, chunk_size) for _ in range(REPEATS)]
        peak = _median([r["peak_rss_mib"] for r in runs])
        rows.append(
            {
                "sampler": sampler_name,
                "mode": mode,
                "chunk_size": chunk_size,
                "elapsed_s_median": round(_median([r["elapsed_s"] for r in runs]), 3),
                "peak_rss_mib_median": round(peak, 1),
                "sampler_extra_mib": round(peak - baseline_mib, 1),
                "output_shape": runs[0]["output_shape"],
                "input_mib": round(runs[0]["input_mib"], 1),
                "output_mib": round(runs[0]["output_mib"], 1),
            }
        )
        print(rows[-1])

    report = {
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "seed": SEED,
        "repeats": REPEATS,
        "data_only_peak_rss_mib": round(baseline_mib, 1),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))

    headers = (
        "| sampler | mode | chunk_size | time (s) | peak RSS (MiB) | "
        "sampler extra (MiB) | output shape |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = [headers]
    for r in rows:
        lines.append(
            f"| {r['sampler']} | {r['mode']} | {r['chunk_size']} | "
            f"{r['elapsed_s_median']:.3f} | {r['peak_rss_mib_median']:.1f} | "
            f"{r['sampler_extra_mib']:.1f} | {tuple(r['output_shape'])} |\n"
        )
    lines.append(
        f"\nData-only process peak (baseline): "
        f"{report['data_only_peak_rss_mib']:.1f} MiB. "
        f"Input size: {rows[0]['input_mib']:.1f} MiB. "
        "Median of 3 subprocess runs; sampler extra = peak RSS - baseline.\n"
    )
    OUT_MD.write_text("".join(lines))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        chunk = None if sys.argv[4] == "None" else int(sys.argv[4])
        _run_worker(sys.argv[2], sys.argv[3], chunk)
    else:
        raise SystemExit(main())
