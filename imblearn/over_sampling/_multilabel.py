"""Utilities for multi-label indicator targets.

A multi-label target is a binary indicator matrix ``y`` of shape
``(n_samples, n_labels)`` where ``y[i, j] == 1`` means that sample ``i``
carries label ``j``.  A single sample may carry several labels at once.

The samplers built on top of this module reason about the *label set* of a
sample, i.e. the full combination of active labels, instead of handling each
label column independently.  Nothing here densifies a sparse target: label
sets are read from the CSR ``indptr``/``indices`` arrays, so the memory
footprint stays linear in the number of stored ones and never grows with the
square of the number of labels.
"""

# Authors: imbalanced-learn developers
# License: MIT

from collections import OrderedDict

import numpy as np
from scipy import sparse
from sklearn.utils.multiclass import type_of_target
from sklearn_compat.utils.validation import check_array

#: Label set used for rows without any active label.
EMPTY_LABEL_SET: tuple = ()

# Above this number of padded cells (n_rows x max row nnz) the vectorised
# grouping falls back to one byte key per row to keep the peak memory bounded.
# A million rows with three labels each needs ~3M cells, i.e. ~24 MB as int64.
_PADDED_GROUPING_MAX_CELLS = 16_000_000


def check_multilabel_target(y, *, zero_label_rows_threshold=None):
    """Validate that ``y`` is a binary multi-label indicator matrix.

    Dense arrays and every scipy sparse format are accepted.  A sparse target
    is converted to CSR (only when needed) and is never densified.

    Parameters
    ----------
    y : {ndarray, sparse matrix} of shape (n_samples, n_labels)
        The multi-label indicator matrix to validate.

    zero_label_rows_threshold : int, float or None, default=None
        Policy regarding rows carrying no label at all.

        - ``None``: any number of all-zero rows is accepted;
        - an ``int``: maximum number of all-zero rows allowed;
        - a ``float`` in the range ``[0, 1]``: maximum fraction of all-zero
          rows allowed.

    Returns
    -------
    y_validated : {ndarray, sparse matrix} of shape (n_samples, n_labels)
        The validated target.  A sparse matrix is returned as a CSR matrix, a
        dense target as a numpy array.

    Raises
    ------
    ValueError
        If the target is not a 2D indicator matrix, contains values other than
        0 and 1, has a label column that is always zero, or breaks the
        all-zero-row policy.  The error message reports the offending
        positions and counts whenever possible.
    """
    # Inspect the raw number of dimensions first: a 2D integer matrix with a
    # value of e.g. 2 is a malformed indicator matrix, not a multiclass
    # target, and should be reported as an illegal value.
    raw_ndim = _raw_ndim(y)
    if raw_ndim != 2:
        try:
            type_y = type_of_target(y)
        except (TypeError, ValueError):  # pragma: no cover - ragged input
            type_y = "unknown"
        raise ValueError(
            "Multi-label samplers expect a binary indicator matrix of shape "
            f"(n_samples, n_labels). Got target of type {type_y!r} with "
            f"{raw_ndim} dimension(s) instead. Use a single-label sampler "
            "(e.g. SMOTE) for 1D targets; reshape a binarized target from "
            "sklearn.preprocessing.MultiLabelBinarizer to 2D before "
            "resampling."
        )

    # A pandas DataFrame with a sparse dtype is not reported as sparse by
    # `scipy.sparse.issparse`; turn it into a scipy sparse matrix so that the
    # sparse path below (which never densifies) handles it too.
    if _is_pandas_sparse_frame(y):
        y = y.sparse.to_coo()

    if sparse.issparse(y):
        if not np.issubdtype(y.dtype, np.number) and y.dtype != np.bool_:
            raise ValueError(
                "Sparse multi-label targets must have a numeric or boolean "
                f"dtype, got {y.dtype!r}. Convert the indicator matrix first, "
                "for example with `y.astype(np.int8)`."
            )
        try:
            y_checked = check_array(
                y,
                accept_sparse=["csr", "csc", "coo", "lil", "dok", "bsr", "dia"],
                ensure_2d=True,
                ensure_all_finite=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The sparse multi-label target could not be validated. Make "
                "sure it is a 2D numeric indicator matrix with values in "
                "{0, 1}; `y.tocsr()` gives a canonical sparse format."
            ) from exc
        # `isspmatrix_csr` is False for the scipy sparse *array* classes, so
        # check the `format` attribute instead.
        if getattr(y_checked, "format", None) != "csr":
            y_checked = y_checked.tocsr()
        # Explicit zeros and duplicated entries are legal inputs but would
        # corrupt the label-set extraction; normalise them on a private copy.
        y_checked = y_checked.copy()
        y_checked.sum_duplicates()
        y_checked.eliminate_zeros()
        y_checked.sort_indices()
        data = y_checked.data
    else:
        try:
            y_checked = check_array(y, ensure_2d=True, ensure_all_finite=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The multi-label target could not be validated. Make sure it "
                "is a 2D numeric indicator matrix with values in {0, 1}. "
                "Non-finite values (NaN or inf) are not supported."
            ) from exc
        data = y_checked.ravel()

    if y_checked.ndim != 2:
        raise ValueError(
            "Multi-label samplers expect a 2D indicator matrix of shape "
            f"(n_samples, n_labels). Got {y_checked.ndim}D input instead."
        )

    unique_values = (
        np.unique(data) if data.size else np.array([], dtype=y_checked.dtype)
    )
    illegal = unique_values[(unique_values != 0) & (unique_values != 1)]
    if illegal.size:
        positions = _illegal_value_positions(y_checked, illegal)
        raise ValueError(
            "The multi-label indicator matrix can only contain 0 and 1. "
            f"Found {positions['n_illegal']} illegal entries with values "
            f"{positions['values'].tolist()}. First offending positions "
            f"(row, column): {positions['locations']}. Binarize the target "
            "(e.g. with sklearn.preprocessing.MultiLabelBinarizer) before "
            "resampling."
        )

    n_samples, n_labels = y_checked.shape
    if n_labels < 2:
        raise ValueError(
            "A multi-label indicator matrix needs at least 2 label columns. "
            f"Got {n_labels} column(s). For a single label use a 1D target "
            "with a single-label sampler such as SMOTE."
        )

    # Label columns that are never active cannot be learned nor synthesised.
    n_active_per_label = np.asarray(_axis_sum(y_checked, axis=0)).ravel()
    dead_labels = np.flatnonzero(n_active_per_label == 0)
    if dead_labels.size:
        raise ValueError(
            f"Label column(s) {dead_labels.tolist()} never take a positive "
            "value in the training target. Every label column must be active "
            "for at least one sample; remove the empty columns before "
            "resampling."
        )

    zero_mask = np.asarray(_row_nnz(y_checked) == 0).ravel()
    zero_row_count = int(np.count_nonzero(zero_mask))
    if zero_label_rows_threshold is not None and zero_row_count:
        limit = _zero_row_limit(zero_label_rows_threshold, n_samples)
        if zero_row_count > limit:
            zero_rows = np.flatnonzero(zero_mask)[:5].tolist()
            raise ValueError(
                f"Found {zero_row_count} all-zero label row(s) "
                f"({zero_row_count / n_samples:.2%} of the data) but the "
                f"configured limit is {limit}. First all-zero rows: "
                f"{zero_rows}. Set `zero_label_rows_threshold` to a larger "
                "value, assign labels to these samples, or remove them."
            )

    return y_checked


def _is_pandas_sparse_frame(y):
    """Whether ``y`` is a pandas DataFrame backed by a sparse dtype."""
    if not hasattr(y, "sparse") or not hasattr(y, "dtypes"):
        return False
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is an optional dependency
        return False
    return isinstance(y, pd.DataFrame) and all(
        isinstance(dtype, pd.SparseDtype) for dtype in y.dtypes
    )


def _raw_ndim(y):
    """Number of dimensions without densifying sparse inputs."""
    if sparse.issparse(y):
        return y.ndim
    if isinstance(y, np.ndarray):
        return y.ndim
    try:
        return np.asarray(y).ndim
    except (TypeError, ValueError):
        # Ragged nested sequences cannot be converted to an array.  They are
        # definitively not a valid indicator matrix; report 0 dimensions.
        return 0


def _zero_row_limit(threshold, n_samples):
    """Convert an int/float all-zero-row threshold into an absolute count."""
    if isinstance(threshold, (int, np.integer)):
        if threshold < 0:
            raise ValueError(
                "`zero_label_rows_threshold` must be non-negative when given "
                f"as an integer. Got {threshold}."
            )
        return int(threshold)
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "`zero_label_rows_threshold` as a float must be in the range "
            f"[0, 1]. Got {threshold}."
        )
    return int(np.floor(threshold * n_samples))


def _illegal_value_positions(y, illegal_values):
    """Locate a handful of entries holding non-binary values."""
    if sparse.issparse(y):
        # Explicit zeros were removed by the validation step; only stored
        # entries can be illegal here.
        coo = y.tocoo()
        mask = np.isin(coo.data, illegal_values)
        rows = coo.row[mask][:5]
        cols = coo.col[mask][:5]
        n_illegal = int(mask.sum())
    else:
        mask = np.isin(y, illegal_values)
        rows, cols = np.nonzero(mask)
        rows, cols = rows[:5], cols[:5]
        n_illegal = int(mask.sum())
    return {
        "n_illegal": n_illegal,
        "values": illegal_values,
        "locations": list(zip(rows.tolist(), cols.tolist())),
    }


def _axis_sum(y, axis):
    """Sum along an axis for both dense and sparse targets."""
    if sparse.issparse(y):
        return np.asarray(y.sum(axis=axis))
    return y.sum(axis=axis)


def _row_nnz(y):
    """Number of stored/non-zero entries per row."""
    if sparse.issparse(y) and getattr(y, "format", None) == "csr":
        return np.diff(y.indptr)
    if sparse.issparse(y):
        return np.asarray(y.getnnz(axis=1)).ravel()
    return np.count_nonzero(y, axis=1)


def _as_csr(y):
    """Return a canonical CSR matrix/array of a validated target.

    The returned object is a private copy: explicit zeros and duplicated
    entries are removed and the indices are sorted.  Dense targets are
    converted without passing through ``n_labels ** 2`` memory.
    """
    if sparse.issparse(y):
        if getattr(y, "format", None) == "csr" and y.has_sorted_indices:
            csr = y.copy()
        else:
            csr = y.tocsr(copy=True)
        csr.sum_duplicates()
        csr.eliminate_zeros()
        csr.sort_indices()
        return csr
    # Building the CSR counterpart of a dense indicator only stores the ones.
    csr = sparse.csr_matrix(np.asarray(y))
    csr.eliminate_zeros()
    csr.sort_indices()
    return csr


def label_sets(y):
    """Extract the label set (tuple of active label indices) of every row.

    Parameters
    ----------
    y : {ndarray, sparse matrix} of shape (n_samples, n_labels)
        A validated binary indicator matrix.

    Returns
    -------
    sets : list of tuple
        ``sets[i]`` is the sorted tuple of column indices active in row ``i``.
        An all-zero row maps to the empty tuple.

    Notes
    -----
    For a CSR matrix the tuples are built from ``indptr``/``indices`` without
    calling ``toarray``; the cost is :math:`O(\\mathrm{nnz}(y))`.
    """
    csr = _as_csr(y)
    return [
        tuple(csr.indices[start:stop].tolist())
        for start, stop in zip(csr.indptr[:-1], csr.indptr[1:])
    ]


def group_by_label_set(y):
    """Group sample indices by label set and count each combination.

    Parameters
    ----------
    y : {ndarray, sparse matrix} of shape (n_samples, n_labels)
        A validated binary indicator matrix.

    Returns
    -------
    groups : OrderedDict[tuple, ndarray]
        Mapping from label set to the sorted array of row indices carrying
        exactly this combination.  Insertion order follows the first
        appearance of each combination.
    counts : OrderedDict[tuple, int]
        Number of samples per label set, in the same order.
    """
    return _group_by_label_set_csr(_as_csr(y))


def _group_by_label_set_csr(csr):
    """Grouping implementation working on a canonical CSR target."""
    n_samples = csr.shape[0]
    row_nnz = np.diff(csr.indptr)
    max_nnz = int(row_nnz.max()) if n_samples else 0

    if max_nnz == 0:
        empty = np.arange(n_samples, dtype=np.intp)
        groups = OrderedDict([(EMPTY_LABEL_SET, empty)])
        counts = OrderedDict([(EMPTY_LABEL_SET, n_samples)])
        return groups, counts

    if n_samples * max_nnz <= _PADDED_GROUPING_MAX_CELLS:
        group_ids, group_label_sets = _groups_from_padded(csr, row_nnz, max_nnz)
    else:
        group_ids, group_label_sets = _groups_from_keys(csr)

    # Re-index the groups so that they are ordered by first appearance.  Both
    # paths can return internal ids in an arbitrary order.
    order = np.argsort(group_ids, kind="stable")
    sorted_ids = group_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    first_rows = order[starts]
    # ``first_rows[i]`` is the first row of internal group ``i``; rank the
    # groups by that row so ``final_ids`` follows first appearance.
    rank_by_group = np.empty(first_rows.size, dtype=np.intp)
    rank_by_group[np.argsort(first_rows, kind="stable")] = np.arange(
        first_rows.size, dtype=np.intp
    )
    final_ids = rank_by_group[group_ids]

    # A single stable sort splits the rows per final group id.  Stability keeps
    # the rows of each group sorted by their original index.
    order = np.argsort(final_ids, kind="stable")
    sorted_ids = final_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    sizes = np.diff(np.r_[starts, sorted_ids.size])

    groups = OrderedDict()
    counts = OrderedDict()
    for start, size in zip(starts.tolist(), sizes.tolist()):
        rows = order[start : start + size]
        label_set = group_label_sets[int(group_ids[rows[0]])]
        groups[label_set] = rows.astype(np.intp, copy=False)
        counts[label_set] = size
    return groups, counts


def _groups_from_padded(csr, row_nnz, max_nnz):
    """Vectorised grouping: pad every row's label set to a fixed width.

    ``-1`` is the padding sentinel; labels are non-negative so it cannot
    collide with a real label.  Rows are ordered lexicographically and the
    boundaries between equal rows delimit the groups.
    """
    n_samples = csr.shape[0]
    indptr = csr.indptr
    padded = np.full((n_samples, max_nnz), -1, dtype=np.int64)
    row_ids = np.repeat(np.arange(n_samples, dtype=np.intp), row_nnz)
    if row_ids.size:
        starts = np.repeat(indptr[:-1], row_nnz)
        columns = np.arange(row_ids.size, dtype=np.intp) - starts
        padded[row_ids, columns] = csr.indices
    order = np.lexsort([padded[:, column] for column in range(max_nnz - 1, -1, -1)])
    ordered = padded[order]
    is_new_group = np.ones(n_samples, dtype=bool)
    if n_samples > 1:
        is_new_group[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    starts = np.flatnonzero(is_new_group)
    sizes = np.diff(np.r_[starts, n_samples])
    group_ids = np.empty(n_samples, dtype=np.intp)
    group_ids[order] = np.repeat(np.arange(starts.size, dtype=np.intp), sizes)
    group_label_sets = [
        tuple(int(label) for label in ordered[start] if label >= 0) for start in starts
    ]
    return group_ids, group_label_sets


def _groups_from_keys(csr):
    """Memory-bounded grouping: one ``bytes`` key per row.

    Used when the padded representation would exceed the memory budget.  Both
    the keys and the group table stay linear in the number of stored ones, so a
    wide label matrix never allocates ``n_samples * n_labels`` cells.
    """
    n_samples = csr.shape[0]
    indptr = csr.indptr
    indices = csr.indices
    lookup = {}
    group_ids = np.empty(n_samples, dtype=np.intp)
    group_label_sets = []
    for row in range(n_samples):
        start, stop = indptr[row], indptr[row + 1]
        key = indices[start:stop].tobytes()
        internal_id = lookup.get(key)
        if internal_id is None:
            internal_id = len(group_label_sets)
            lookup[key] = internal_id
            group_label_sets.append(tuple(int(label) for label in indices[start:stop]))
        group_ids[row] = internal_id
    return group_ids, group_label_sets


def indicator_block(
    label_set, n_rows, n_labels, *, sparse_output=False, dtype=np.int64
):
    """Build an ``n_rows x n_labels`` block whose rows all carry ``label_set``.

    The sparse representation stores exactly ``n_rows * len(label_set)`` ones,
    so the memory used is proportional to the labels actually carried and not
    to ``n_rows * n_labels``.
    """
    label_set = np.asarray(sorted({int(label) for label in label_set}), dtype=np.intp)
    if sparse_output:
        n_entries = n_rows * label_set.size
        if label_set.size == 0:
            indices = np.empty(0, dtype=np.intp)
            data = np.empty(0, dtype=dtype)
        else:
            indices = np.tile(label_set, n_rows)
            data = np.ones(n_entries, dtype=dtype)
        if label_set.size:
            indptr = np.arange(0, n_entries + 1, label_set.size, dtype=np.intp)
        else:
            indptr = np.zeros(n_rows + 1, dtype=np.intp)
        # `sparse.csr_matrix` and `sparse.csr_array` share this constructor.
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(n_rows, n_labels), dtype=dtype
        )
    block = np.zeros((n_rows, n_labels), dtype=dtype)
    if label_set.size:
        block[:, label_set] = 1
    return block
