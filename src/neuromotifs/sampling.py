"""Sampling networks from a probability matrix and counting their triplet motifs.

Counts are the 13 connected triplet motifs in the order of Gal et al. 2017 (nn.4576), computed by
``netsci``. Tables follow the schema of the bundled ``motifs.csv.gz``: ``motif_1`` … ``motif_13``,
``nodes``, ``edges``, ``reciprocal_edges``, plus any labels (``seed``, ``model``, ``celltype``, …).
"""

import netsci.metrics.motifs as nsm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

MOTIF_COLUMNS = [f"motif_{i}" for i in range(1, 14)]


def motif_counts(A) -> np.ndarray:
    """The 13 connected triplet-motif counts of a binary network, nn.4576 order."""
    return nsm.motifs(np.asarray(A, dtype=int), algorithm="matmul")[
        nsm.triad_order_nn4576
    ]


def motif_table(networks, **labels) -> pd.DataFrame:
    """One row per network: its motif counts and size, plus the given labels.

    ``networks`` is one adjacency matrix or a stack of them ``(n_networks, n, n)``. A scalar label
    is repeated on every row; a sequence label gives one value per row.
    """
    networks = np.asarray(networks, dtype=int)
    if networks.ndim == 2:
        networks = networks[None]
    return pd.DataFrame([_row(A) for A in networks]).assign(**labels)


def sample_motif_counts(P, n_samples, base_seed=0, n_jobs=1, **labels) -> pd.DataFrame:
    """The motif table of ``n_samples`` networks drawn from ``P``, with each network's seed.

    Network ``i`` is drawn with ``np.random.RandomState(base_seed + i).binomial(1, P)``, so any row
    can be regenerated on its own from its ``seed``: this is exactly how the bundled
    ``motifs.csv.gz`` rows were drawn, and the legacy stream is frozen across NumPy versions.
    Drawing and counting run per network under ``joblib.Parallel(n_jobs)``.
    """
    seeds = base_seed + np.arange(n_samples)
    rows = Parallel(n_jobs=n_jobs)(delayed(_sample_row)(P, seed) for seed in seeds)
    return pd.DataFrame(rows).assign(seed=seeds, **labels)


def _row(A) -> dict:
    return {
        **dict(zip(MOTIF_COLUMNS, motif_counts(A))),
        "nodes": len(A),
        "edges": int(A.sum()),
        "reciprocal_edges": int((A * A.T).sum() // 2),
    }


def _sample_row(P, seed) -> dict:
    return _row(np.random.RandomState(seed).binomial(1, P))
