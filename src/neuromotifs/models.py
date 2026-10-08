"""The geometric model hierarchy: one estimator, five orders.

Each order fits the probability of a connection ``from -> to`` as a function of a subset of the
pairwise features (:mod:`neuromotifs.features`), with a scikit-learn classifier on all ordered
pairs of a connectome:

=========  =====  =====================================================================
``order``  rank   features
=========  =====  =====================================================================
``er``     1      none: a constant, the connectome's sparsity (Erdős–Rényi)
``dd``     2      ``offset_rho`` (distance-dependent)
``ddz``    3      ``+ sign_dz`` (bipolar distance-dependent)
``od``     4      ``+ offset_az, offset_el`` (offset-dependent)
``ld``     5      ``+ from_rho, from_az, from_el`` (position-dependent)
=========  =====  =====================================================================

The fitted model gives an ``n x n`` probability matrix for any positions, and networks are sampled
from it as independent Bernoulli draws. The estimator follows scikit-learn's conventions
(``get_params``, ``clone``, fitted attributes end in ``_``) but is not a scikit-learn classifier:
``X`` is a positions array and ``y`` an adjacency matrix.
"""

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted

from .features import pairwise_features

ORDERS = {
    "er": [],
    "dd": ["offset_rho"],
    "ddz": ["offset_rho", "sign_dz"],
    "od": ["offset_rho", "sign_dz", "offset_az", "offset_el"],
    "ld": [
        "offset_rho",
        "sign_dz",
        "offset_az",
        "offset_el",
        "from_rho",
        "from_az",
        "from_el",
    ],
}
"""Feature set of each model order, in increasing order."""


def paper_classifier() -> GradientBoostingClassifier:
    """The classifier of the paper's Figures 2–3: a gradient-boosted tree ensemble, fixed seed."""
    return GradientBoostingClassifier(
        learning_rate=0.01, n_estimators=500, max_depth=5, random_state=1234
    )


class GeometricModel(BaseEstimator):
    """A geometric null model of connectivity, of a given order.

    Parameters
    ----------
    order : str or int
        One of :data:`ORDERS` (``"er"``, ``"dd"``, ``"ddz"``, ``"od"``, ``"ld"``) or its rank 1–5.
    classifier : scikit-learn classifier, optional
        Must implement ``fit`` and ``predict_proba``. Cloned before fitting. Default:
        :func:`paper_classifier`. Ignored for ``order="er"``.

    Attributes
    ----------
    feature_names_ : list of str
        The features the order uses.
    classifier_ : fitted classifier (orders 2–5)
    p_ : float (order 1)
        The fitted connection probability.
    """

    def __init__(self, order="ld", classifier=None):
        self.order = order
        self.classifier = classifier

    def fit(self, positions, A):
        """Fit on a connectome: ``positions`` (n, 3) and its binary adjacency ``A`` (n, n)."""
        features = pairwise_features(positions)
        y = np.asarray(A)[features["from"], features["to"]] != 0
        self.feature_names_ = ORDERS[order_name(self.order)]
        if self.feature_names_:
            classifier = (
                self.classifier if self.classifier is not None else paper_classifier()
            )
            self.classifier_ = clone(classifier).fit(features[self.feature_names_], y)
        else:
            self.p_ = float(y.mean())
        return self

    def predict_proba(self, positions) -> np.ndarray:
        """The ``n x n`` matrix of connection probabilities, zero diagonal."""
        check_is_fitted(self)
        features = pairwise_features(positions)
        if self.feature_names_:
            p = self.classifier_.predict_proba(features[self.feature_names_])[:, 1]
        else:
            p = self.p_
        P = np.zeros((len(positions), len(positions)))
        P[features["from"], features["to"]] = p
        return P

    def sample(self, positions, n_samples=None, random_state=None) -> np.ndarray:
        """Networks drawn from the fitted probabilities, as independent Bernoulli edges.

        Returns one ``(n, n)`` int8 matrix, or ``(n_samples, n, n)`` when ``n_samples`` is given.
        Draws use the legacy ``numpy.random.RandomState`` stream, which NumPy keeps frozen, so a
        seed regenerates the same network in any NumPy version (the paper's samples were drawn this way).
        """
        P = self.predict_proba(positions)
        size = None if n_samples is None else (n_samples, *P.shape)
        return np.random.RandomState(random_state).binomial(1, P, size=size).astype(np.int8)


def order_name(order) -> str:
    """Normalise an order given by name or by rank (1-5) to its name."""
    names = list(ORDERS)
    if isinstance(order, str) and order in ORDERS:
        return order
    if isinstance(order, (int, np.integer)) and 1 <= order <= len(names):
        return names[order - 1]
    raise ValueError(
        f"order must be one of {names} or a rank 1-{len(names)}, got {order!r}"
    )
