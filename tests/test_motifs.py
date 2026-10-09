import numpy as np
import pandas as pd

import netsci.metrics.motifs as nsm

from neuromotifs import MOTIF_COLUMNS, load_motifs, load_nmc, motif_counts


def test_motif_counts_match_paper():
    A = load_nmc("L5_MC").A
    bb = load_motifs().query("celltype == 'L5_MC' and model == 'bb'").iloc[0]
    row = motif_counts(A)
    assert list(row) == MOTIF_COLUMNS + ["nodes", "edges", "reciprocal_edges"]
    assert [row[c] for c in MOTIF_COLUMNS] == bb[MOTIF_COLUMNS].tolist()
    # the paper's convention on top of netsci: the 13 connected triads, nn.4576 order
    assert [row[c] for c in MOTIF_COLUMNS] == nsm.motifs(A.astype(int))[nsm.triad_order_nn4576].tolist()
    assert (row["nodes"], row["edges"], row["reciprocal_edges"]) == (395, 891, bb["reciprocal_edges"])

    # framed, the row is in the schema of motifs.csv.gz
    table = pd.DataFrame([row]).assign(model="bb")
    assert table.iloc[0][MOTIF_COLUMNS].tolist() == bb[MOTIF_COLUMNS].tolist()


def test_motif_counts_of_a_feed_forward_loop():
    A = np.zeros((4, 4), dtype=int)
    A[0, 1] = A[0, 2] = A[1, 2] = 1  # one feed-forward loop: motif 4
    row = motif_counts(A)
    assert row["motif_4"] == 1 and sum(row[c] for c in MOTIF_COLUMNS) == 1
    assert (row["edges"], row["reciprocal_edges"]) == (3, 0)
    assert motif_counts(A.T)["motif_4"] == 1  # reversing every edge keeps a feed-forward loop
