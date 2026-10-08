from pathlib import Path

import numpy as np
import pytest

from neuromotifs import load_motifs, load_nmc, motif_counts, sample_motif_counts
from neuromotifs.sampling import MOTIF_COLUMNS, motif_counts_row

REFERENCE = Path(__file__).parent / "data" / "L5_MC.dd.P.npz"


@pytest.fixture(scope="module")
def paper_l5_mc():
    return load_motifs().query("celltype == 'L5_MC'")


def test_motif_counts_match_paper(paper_l5_mc):
    A = load_nmc("L5_MC").A
    bb = paper_l5_mc.query("model == 'bb'").iloc[0]
    assert np.array_equal(motif_counts(A), bb[MOTIF_COLUMNS].values)

    row = motif_counts_row(A, model="bb")
    assert (row["nodes"], row["edges"], row["reciprocal_edges"]) == (
        395,
        891,
        bb["reciprocal_edges"],
    )
    assert row["model"] == "bb"


def test_sample_table_schema_and_seeds():
    P = np.full((30, 30), 0.1)
    np.fill_diagonal(P, 0)
    table = sample_motif_counts(
        P, n_samples=4, base_seed=100, model="er", celltype="toy"
    )
    assert list(table.columns) == MOTIF_COLUMNS + [
        "nodes",
        "edges",
        "reciprocal_edges",
        "seed",
        "model",
        "celltype",
    ]
    assert table["seed"].tolist() == [100, 101, 102, 103]
    assert (table["nodes"] == 30).all() and table["edges"].between(40, 140).all()
    again = sample_motif_counts(
        P, n_samples=4, base_seed=100, n_jobs=2, model="er", celltype="toy"
    )
    assert table.equals(again)


def test_er_ratio_of_feedforward_to_cycle():
    """Independent symmetric edges give E[#4] = 3 E[#7] exactly; a sample should sit near 3."""
    n, p = 150, 0.05
    P = np.full((n, n), p)
    np.fill_diagonal(P, 0)
    table = sample_motif_counts(P, n_samples=300, base_seed=0)
    assert table["motif_4"].mean() / table["motif_7"].mean() == pytest.approx(
        3, rel=0.1
    )


@pytest.mark.slow
def test_dd_samples_reproduce_paper_distribution(paper_l5_mc):
    """200 draws from the reference dd matrix match the paper's 1000-sample dd distribution."""
    P = np.load(REFERENCE)["P"].astype(float)
    paper = paper_l5_mc.query("model == 'dd'")[MOTIF_COLUMNS]
    table = sample_motif_counts(P, n_samples=200, base_seed=1234, n_jobs=-2)[
        MOTIF_COLUMNS
    ]
    frequent = paper.mean() >= 5
    z = (table.mean() - paper.mean()) / np.sqrt(
        paper.var() / len(paper) + table.var() / len(table)
    )
    assert z[frequent].abs().max() < 4
