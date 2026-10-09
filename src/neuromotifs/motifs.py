"""Triplet motifs in the paper's convention.

``netsci`` counts all 16 triads; the paper reports the 13 connected ones, in the order of Gal et al.
2017 (nn.4576), under the names ``motif_1`` … ``motif_13``, which is the schema of the bundled
``motifs.csv.gz`` (with ``nodes``, ``edges``, ``reciprocal_edges`` and labels such as ``seed``,
``model``, ``celltype``).
"""

import netsci.metrics.motifs as nsm
import numpy as np

MOTIF_COLUMNS = [f"motif_{i}" for i in range(1, 14)]


def motif_counts(A) -> dict:
    """The 13 connected triplet-motif counts of a binary network, with its size.

    Keys: ``motif_1`` … ``motif_13`` (nn.4576 order), ``nodes``, ``edges``, ``reciprocal_edges``.
    One dict per network; ``pd.DataFrame([motif_counts(A) for A in networks])`` is a table in the
    schema of ``motifs.csv.gz``.
    """
    A = np.asarray(A, dtype=int)
    counts = nsm.motifs(A, algorithm="matmul")[nsm.triad_order_nn4576]
    return {
        **dict(zip(MOTIF_COLUMNS, counts)),
        "nodes": len(A),
        "edges": int(A.sum()),
        "reciprocal_edges": int((A * A.T).sum() // 2),
    }
