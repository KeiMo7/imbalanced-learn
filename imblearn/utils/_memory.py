"""Memory profiling and bounded-memory (chunked) resampling helpers.

This module gives every sampler a *declarative* and *measurable* memory
contract so that the peak memory cost of a resampling call can be estimated
before any large array is allocated. It also provides the shared machinery
used by the chunked code paths of the index-based samplers
(:class:`~imblearn.over_sampling.RandomOverSampler` and
:class:`~imblearn.under_sampling.RandomUnderSampler`).

Design goals
------------

* No new mandatory dependency: only :mod:`numpy` and the standard library
  are used.
* The profiling API never copies data; it only inspects array metadata.
* Chunking changes *where* the output array is filled, not the random number
  generation sequence, so chunked and full paths give bit-identical results
  with the same seed.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from scipy import sparse
from sklearn.utils._param_validation import Interval

#: Number of rows above which a non-chunked call triggers a large-data
#: advisory warning (when the memory guard is enabled).
LARGE_DATA_THRESHOLD = 500_000

#: Parameter constraints shared by every sampler accepting the memory
#: management keyword arguments.
MEMORY_PARAM_CONSTRAINTS: dict = {
    "chunk_size": [Interval(Integral, 1, None, closed="left"), None],
    "memory_guard": ["boolean"],
}

# Reusable docstring fragments (kept identical across samplers).
_chunk_size_docstring = """chunk_size : int or None, default=None
        Number of rows materialised per chunk when building the resampled
        arrays. When ``None`` (default) the sampler uses the historical
        implementation and its memory behaviour is unchanged.

        When set to a positive integer, the output array is pre-allocated to
        its final size and filled chunk by chunk, so the *temporary* memory
        footprint is bounded by ``chunk_size * n_features`` instead of
        growing with the full input size. The selection indices are still
        computed globally with the same random number generator calls, hence
        results are identical to the non-chunked path.

        Only dense in-memory arrays support the chunked path; sparse matrices
        and heterogeneous (object) arrays raise an explicit error and keep
        using the regular path.

        .. versionadded:: 0.15""".strip()

_memory_guard_docstring = """memory_guard : bool, default=False
        When ``True``, the sampler estimates its peak extra memory usage
        before resampling:

        - a large dataset processed without ``chunk_size`` emits an
          :class:`~python:warnings.UserWarning` carrying the estimate;
        - when the estimate exceeds the available memory (or a
          ``MemoryError`` is raised mid-run), the sampler degrades to
          *no resampling*: it returns the original ``X`` and ``y`` untouched,
          emits a :class:`~python:warnings.UserWarning` with actionable
          advice and sets the ``degraded_`` attribute to ``True``.

        When ``False`` (default) memory errors propagate as usual.

        .. versionadded:: 0.15""".strip()


@dataclass
class MemoryProfile:
    """Memory characteristics of a sampler for a given input.

    All sizes are expressed in bytes and describe *extra* memory on top of
    the input arrays, which are never counted as sampler-owned allocations.

    Parameters
    ----------
    copies_input : bool
        Whether the implementation copies the full input array as part of its
        regular (non-chunked) path.

    chunked_supported : bool
        Whether a bounded-memory chunked path is available for this input
        type.

    peak_extra_bytes_full : int
        Estimated peak extra bytes used by the historical full path.

    peak_extra_bytes_chunked : int or None
        Estimated peak extra bytes used by the chunked path for the given
        ``chunk_size``, or ``None`` when chunking is not applicable.

    output_bytes : int
        Size of the resampled output arrays (``X_resampled`` and
        ``y_resampled``). This allocation is unavoidable in both paths.

    temp_extra_bytes_chunked : int or None
        Temporary, non-output bytes of the chunked path (per-chunk scratch
        space). It depends only on ``chunk_size`` and the row width, never on
        the total number of input rows, so it stays constant as the dataset
        grows. ``None`` when chunking is not applicable.

    scaling : str
        Human-readable relation between the sampling ratio and the extra
        memory, e.g. ``"O((1 + ratio) * n * p)"``.

    notes : tuple of str
        Free-form implementation notes (copy points, limitations).
    """

    copies_input: bool
    chunked_supported: bool
    peak_extra_bytes_full: int
    peak_extra_bytes_chunked: int | None
    output_bytes: int
    scaling: str
    temp_extra_bytes_chunked: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """Return a readable multi-line description of the profile."""
        lines = [
            f"copies_input={self.copies_input}",
            f"chunked_supported={self.chunked_supported}",
            f"peak_extra_bytes(full)={self.peak_extra_bytes_full}",
            f"peak_extra_bytes(chunked)={self.peak_extra_bytes_chunked}",
            f"temp_extra_bytes(chunked)={self.temp_extra_bytes_chunked}",
            f"output_bytes={self.output_bytes}",
            f"scaling={self.scaling}",
        ]
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)


def _array_nbytes(X) -> int:
    """Return the data bytes of an array-like without copying it."""
    if sparse.issparse(X):
        sp = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
        if hasattr(X, "indptr") and hasattr(X, "indices"):
            return sp
        return int(sp)
    return int(np.asarray(X).nbytes)


def _row_nbytes(X) -> int:
    """Bytes occupied by one row of ``X`` (dense arrays only)."""
    if sparse.issparse(X):
        # Average sparsity-based cost; used only for documentation estimates.
        return max(1, _array_nbytes(X) // max(1, X.shape[0]))
    return (
        int(np.asarray(X).strides[0])
        if hasattr(X, "strides")
        else (int(np.asarray(X).nbytes // max(1, X.shape[0])))
    )


def _dtype_nbytes(y) -> int:
    arr = np.asarray(y)
    return max(1, arr.dtype.itemsize)


def _supports_chunking(X) -> bool:
    """Chunking requires a dense, fixed-dtype numeric/string ndarray.

    Object arrays (heterogeneous data) and sparse matrices are excluded: they
    cannot be pre-allocated and filled row-wise at a constant dtype.
    """
    if sparse.issparse(X):
        return False
    if not isinstance(X, np.ndarray):
        return False
    return X.dtype.kind in "biufcUS"


def coerce_dense_chunkable(X):
    """Return an ndarray view suitable for the chunked path.

    A homogeneous pandas DataFrame is converted with ``to_numpy`` (one
    explicit, documented copy); an object-dtype or sparse input raises so the
    caller falls back to the regular path semantics with a clear message.
    Ndarrays already supported by :func:`_supports_chunking` are returned
    unchanged.
    """
    if sparse.issparse(X):
        raise ValueError(
            "The chunked path only supports dense in-memory arrays with a "
            "fixed dtype. Sparse matrices should be resampled with "
            "`chunk_size=None`."
        )
    if hasattr(X, "to_numpy") and not isinstance(X, np.ndarray):
        # pandas DataFrame / Series: homogeneous dtypes only
        dtypes = getattr(X, "dtypes", None)
        if dtypes is not None and getattr(dtypes, "nunique", lambda: 1)() > 1:
            raise ValueError(
                "The chunked path only supports arrays with a single fixed "
                "dtype. Heterogeneous DataFrames should be resampled with "
                "`chunk_size=None`."
            )
        X = X.to_numpy()
    if not _supports_chunking(X):
        raise ValueError(
            "The chunked path only supports dense in-memory arrays with a "
            "fixed dtype. Heterogeneous (object) arrays should be resampled "
            "with `chunk_size=None`."
        )
    return X


def validate_chunk_size(chunk_size):
    """Validate the ``chunk_size`` parameter at startup.

    Invalid values are rejected before any data is touched, as required by the
    bounded-memory contract.
    """
    if chunk_size is None:
        return None
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
        raise TypeError(
            "`chunk_size` must be a positive integer or None. Got "
            f"{type(chunk_size).__name__}."
        )
    if chunk_size < 1:
        raise ValueError(f"`chunk_size` must be strictly positive. Got {chunk_size}.")
    return int(chunk_size)


def available_memory_bytes() -> int | None:
    """Return the currently available physical memory in bytes.

    Uses :mod:`psutil` when present and falls back to ``/proc/meminfo`` on
    Linux. Returns ``None`` when no source of information is available: in
    that case pre-flight checks are skipped instead of failing.
    """
    try:
        import psutil  # optional dependency, not required at install time

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "rb") as handle:
            for line in handle:
                if line.startswith(b"MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        pass
    return None


def estimate_index_bytes(n_rows: int, n_output: int) -> int:
    """Bytes used by the integer bookkeeping arrays.

    Both samplers keep an ``int64`` (platform default ``int``) index array
    covering the output rows plus transient per-class index arrays.
    """
    int_size = np.dtype(int).itemsize
    return int((n_rows + n_output) * int_size)


def _format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "unknown"
    num_bytes = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0 or unit == "TiB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TiB"


class MemoryGuard:
    """Pre-flight estimation and graceful degradation for a sampler call.

    Parameters
    ----------
    sampler : sampler instance
        The sampler about to call ``_fit_resample``.

    X, y : arrays
        Validated input arrays.

    n_output : int
        Expected number of output rows.

    chunk_size : int or None
        Chunk size requested by the user.

    profile_factory : callable
        ``f(X, y, n_output, chunk_size) -> MemoryProfile`` bound to the
        concrete sampler.
    """

    def __init__(self, sampler, X, y, n_output, chunk_size, profile_factory):
        self.sampler = sampler
        self.X = X
        self.y = y
        self.n_output = int(n_output)
        self.chunk_size = chunk_size
        self.profile = profile_factory(X, y, self.n_output, chunk_size)
        self.degraded = False
        self.message = ""

    def preflight(self) -> bool:
        """Run the pre-flight checks.

        Returns ``True`` when resampling should proceed, ``False`` when the
        call must degrade to returning the input unchanged.
        """
        n_rows = self.X.shape[0]
        if self.chunk_size is None and n_rows > LARGE_DATA_THRESHOLD:
            warnings.warn(
                "Large dataset detected "
                f"({n_rows} rows > {LARGE_DATA_THRESHOLD}) and no "
                "`chunk_size` is set: the regular path may allocate about "
                f"{_format_bytes(self.profile.peak_extra_bytes_full)} of "
                "extra memory (output arrays included). Pass an explicit "
                "`chunk_size` to bound the temporary footprint, or reduce "
                "the sampling ratio.",
                UserWarning,
                stacklevel=3,
            )

        available = available_memory_bytes()
        if available is not None:
            planned = (
                self.profile.peak_extra_bytes_chunked
                if self.chunk_size is not None
                and self.profile.peak_extra_bytes_chunked is not None
                else self.profile.peak_extra_bytes_full
            )
            # Keep a safety margin: do not consume more than 90% of the
            # currently available memory.
            if planned > int(0.9 * available):
                self._degrade(
                    "Estimated peak extra memory "
                    f"{_format_bytes(planned)} exceeds 90% of the available "
                    f"memory ({_format_bytes(available)}). Resampling was "
                    "skipped. Advice: (1) set `chunk_size` to a few tens of "
                    "thousands of rows, (2) lower the sampling ratio, or (3) "
                    "free memory / run on a machine with more RAM."
                )
                return False
        return True

    def handle_memory_error(self, error: MemoryError) -> bool:
        """Degrade after a mid-run :class:`MemoryError`.

        The sampler is expected to drop every partially allocated array
        before calling this, so the only remaining allocations are the
        original inputs.
        """
        self._degrade(
            "A MemoryError was raised while resampling. All partially "
            "allocated arrays were released and the original data was "
            "returned unchanged. Advice: (1) retry with an explicit "
            "`chunk_size` (e.g. 10000), (2) reduce the sampling ratio, or "
            "(3) free memory before retrying."
        )
        return False

    def _degrade(self, message: str) -> None:
        self.degraded = True
        self.message = message
        warnings.warn(message, UserWarning, stacklevel=3)


def fill_by_indices_chunked(X, indices, output, chunk_size, row_writer=None):
    """Fill ``output`` with ``X[indices]`` one chunk at a time.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Dense source array.

    indices : ndarray of shape (n_output,)
        Integer selection indices, computed globally so that the random
        sequence is independent of the chunk size.

    output : ndarray of shape (n_output, n_features)
        Pre-allocated destination array. Filled in-place, never reallocated.

    chunk_size : int
        Maximum number of rows handled per chunk.

    row_writer : callable or None
        Optional ``f(destination, source)`` hook used when the output rows
        need a transformation (e.g. the smoothed bootstrap). When ``None`` a
        plain copy is performed.

    Returns
    -------
    output : ndarray
        The same array, filled chunk by chunk.
    """
    n_output = indices.shape[0]
    for start in range(0, n_output, chunk_size):
        end = min(start + chunk_size, n_output)
        chunk_idx = indices[start:end]
        source = X[chunk_idx]
        if row_writer is None:
            output[start:end] = source
        else:
            row_writer(output[start:end], source, chunk_idx, start, end)
        # Drop the temporary view/copy before the next iteration.
        del source
    return output
