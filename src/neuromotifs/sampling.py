"""Sampling networks from a probability matrix and counting their triplet motifs.

Counts are the 13 connected triplet motifs in the order of Gal et al. 2017 (nn.4576), computed by
``netsci``. Tables follow the schema of the bundled ``motifs.csv.gz``: ``motif_1`` … ``motif_13``,
``nodes``, ``edges``, ``reciprocal_edges``, ``seed``, plus any labels (``model``, ``celltype``, …).
"""

import netsci.metrics.motifs as nsm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

MOTIF_COLUMNS = [f"motif_{i}" for i in range(1, 14)]


def motif_counts(A) -> np.ndarray:
    """The 13 connected triplet-motif counts of a binary network, nn.4576 order."""
    return nsm.motifs(np.asarray(A, dtype=int), algorithm="matmul")[nsm.triad_order_nn4576]


def motif_counts_row(A, **labels) -> dict:
    """One table row for a network: motif counts, size, and the given labels."""
    A = np.asarray(A, dtype=int)
    return {
        **dict(zip(MOTIF_COLUMNS, motif_counts(A))),
        "nodes": len(A),
        "edges": int(A.sum()),
        "reciprocal_edges": int((A * A.T).sum() // 2),
        **labels,
    }


def sample_motif_counts(P, n_samples, base_seed=0, n_jobs=1, **labels) -> pd.DataFrame:
    """Motif counts of ``n_samples`` networks drawn from ``P``, one row each.

    Network ``i`` is drawn with ``np.random.default_rng(base_seed + i)`` and the seed is recorded,
    so any row can be regenerated on its own. ``n_jobs`` is passed to ``joblib.Parallel``.
    """
    seeds = base_seed + np.arange(n_samples)
    rows = Parallel(n_jobs=n_jobs)(delayed(_sample_row)(P, seed, labels) for seed in seeds)
    return pd.DataFrame(rows)


def _sample_row(P, seed, labels) -> dict:
    A = np.random.default_rng(seed).binomial(1, P)
    return motif_counts_row(A, seed=int(seed), **labels)
