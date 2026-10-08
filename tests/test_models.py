from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score

from neuromotifs import GeometricModel, load_nmc, make_positions
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


def test_sample_shapes_and_seed(toy):
    positions, A = toy
    model = GeometricModel("er").fit(positions, A)

    one = model.sample(positions, random_state=0)
    assert one.shape == (60, 60) and one.dtype == np.int8
    assert np.array_equal(one, model.sample(positions, random_state=0))

    many = model.sample(positions, n_samples=3, random_state=0)
    assert many.shape == (3, 60, 60) and not many[:, np.eye(60, dtype=bool)].any()


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
