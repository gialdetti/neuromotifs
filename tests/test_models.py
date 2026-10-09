from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score

from neuromotifs import MOTIF_COLUMNS, GeometricModel, draw_network, load_motifs, load_nmc, make_positions, motif_counts
from neuromotifs.models import ORDERS, order_name, paper_classifier

REFERENCE = Path(__file__).parent / "data" / "L5_MC.dd.P.npz"


@pytest.fixture(scope="module")
def l5_mc():
    return load_nmc("L5_MC")


@pytest.fixture(scope="module")
def toy():
    """A small random connectome: positions in a box, edges more likely for nearby pairs."""
    positions = make_positions(60, random_state=0)

    d = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    P = 0.4 * np.exp(-d / 150)
    np.fill_diagonal(P, 0)
    A = np.random.default_rng(1).binomial(1, P).astype(np.int8)

    return positions, A


def test_order_names():
    assert [order_name(k) for k in range(1, 6)] == list(ORDERS)
    assert order_name("od") == "od"
    for bad in ["bb", 0, 6, "pd"]:
        with pytest.raises(ValueError):
            order_name(bad)


def test_paper_classifier_configuration():
    params = paper_classifier().get_params()
    assert (params["learning_rate"], params["n_estimators"], params["max_depth"]) == (
        0.01,
        500,
        5,
    )
    assert params["random_state"] == 1234


def test_er_order(l5_mc):
    model = GeometricModel("er").fit(l5_mc.positions, l5_mc.A)
    n, p = len(l5_mc.A), l5_mc.A.sum() / (395 * 394)
    assert model.p_ == pytest.approx(p)

    P = model.predict_proba(l5_mc.positions)
    assert P.shape == (n, n) and not P.diagonal().any()
    assert np.allclose(P[~np.eye(n, dtype=bool)], p)


def test_sklearn_conventions(toy):
    positions, A = toy
    model = GeometricModel(
        order=2, classifier=HistGradientBoostingClassifier(max_iter=20)
    )
    assert clone(model).get_params()["order"] == 2

    with pytest.raises(NotFittedError):
        model.predict_proba(positions)

    assert model.fit(positions, A) is model
    assert model.feature_names_ == ORDERS["dd"]
    assert model.classifier is not model.classifier_  # cloned before fitting


def test_fit_recovers_distance_dependence(toy):
    positions, A = toy
    model = GeometricModel("dd", classifier=HistGradientBoostingClassifier(max_iter=50))
    P = model.fit(positions, A).predict_proba(positions)
    off = ~np.eye(len(A), dtype=bool)
    assert roc_auc_score(A[off], P[off]) > 0.7
    assert P[off].mean() == pytest.approx(A[off].mean(), rel=0.1)


def test_sample_is_one_draw_per_seed(toy):
    positions, A = toy
    model = GeometricModel("er").fit(positions, A)
    assert np.array_equal(model.positions_, positions)

    stack = model.sample(n_samples=3, random_state=10)
    assert stack.shape == (3, 60, 60) and stack.dtype == np.int8
    assert not stack[:, np.eye(60, dtype=bool)].any()
    assert np.array_equal(stack, model.sample(n_samples=3, random_state=10))
    assert np.array_equal(stack[2], model.sample(random_state=12)[0])  # seed 10 + 2

    P = model.predict_proba(positions)
    assert np.array_equal(stack, np.stack([draw_network(P, seed) for seed in (10, 11, 12)]))
    assert model.sample(n_samples=2, positions=positions[:10]).shape == (2, 10, 10)


@pytest.mark.slow
def test_dd_fit_reproduces_paper_probabilities(l5_mc):
    """The paper's classifier, refitted today, gives the 2021 probability matrix (sklearn 0.21)."""
    reference = np.load(REFERENCE)["P"]
    P = (
        GeometricModel("dd")
        .fit(l5_mc.positions, l5_mc.A)
        .predict_proba(l5_mc.positions)
    )
    assert np.abs(P - reference).max() < 1e-3
    assert np.corrcoef(P.ravel(), reference.ravel())[0, 1] > 0.9999


def test_draw_network_is_seeded_and_binary():
    P = np.full((30, 30), 0.1)
    np.fill_diagonal(P, 0)
    A = draw_network(P, seed=100)
    assert A.shape == (30, 30) and A.dtype == np.int8 and set(np.unique(A)) <= {0, 1}
    assert not A.diagonal().any() and 40 <= A.sum() <= 140
    assert np.array_equal(A, draw_network(P, seed=100))
    assert np.array_equal(A, np.random.RandomState(100).binomial(1, P))  # the paper's call


def test_er_ratio_of_feedforward_to_cycle():
    """Independent symmetric edges give E[#4] = 3 E[#7] exactly; a sample should sit near 3."""
    n, p = 150, 0.05
    P = np.full((n, n), p)
    np.fill_diagonal(P, 0)
    counts = [motif_counts(draw_network(P, seed)) for seed in range(300)]
    m4 = np.mean([c["motif_4"] for c in counts])
    m7 = np.mean([c["motif_7"] for c in counts])
    assert m4 / m7 == pytest.approx(3, rel=0.1)


@pytest.mark.slow
def test_dd_draws_reproduce_paper_distribution():
    """200 draws from the reference dd matrix match the paper's 1000-sample dd distribution."""
    import pandas as pd

    P = np.load(REFERENCE)["P"].astype(float)
    paper = load_motifs().query("celltype == 'L5_MC' and model == 'dd'")[MOTIF_COLUMNS]
    drawn = pd.DataFrame([motif_counts(draw_network(P, seed)) for seed in range(1234, 1434)])[MOTIF_COLUMNS]
    frequent = paper.mean() >= 5
    z = (drawn.mean() - paper.mean()) / np.sqrt(paper.var() / len(paper) + drawn.var() / len(drawn))
    assert z[frequent].abs().max() < 4
