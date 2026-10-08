import numpy as np
import pytest

from neuromotifs import make_positions, pairwise_features, pairwise_indices
from neuromotifs.features import pairwise_positions, enrich_pairwise_geometry

POSITION_COLUMNS = ["from", "to", "from_x", "from_y", "from_z", "to_x", "to_y", "to_z"]
GEOMETRY_COLUMNS = [
    "dx",
    "dy",
    "dz",
    "sign_dz",
    "offset_rho",
    "offset_az",
    "offset_el",
    "from_rho",
    "from_az",
    "from_el",
]
FROM, TO, D = (
    ["from_x", "from_y", "from_z"],
    ["to_x", "to_y", "to_z"],
    ["dx", "dy", "dz"],
)


@pytest.fixture(scope="module")
def positions():
    return make_positions(40, random_state=0)


def test_positions_table(positions):
    n = len(positions)
    pairs = pairwise_positions(positions)
    assert list(pairs.columns) == POSITION_COLUMNS
    assert len(pairs) == n * (n - 1)
    assert (pairs["from"] != pairs["to"]).all()

    i, j = pairwise_indices(n)
    assert np.array_equal(pairs["from"].values, i) and np.array_equal(
        pairs["to"].values, j
    )
    assert np.allclose(pairs[FROM].values, positions[i])
    assert np.allclose(pairs[TO].values, positions[j])


def test_features_table(positions):
    f = pairwise_features(positions)
    assert list(f.columns) == POSITION_COLUMNS + GEOMETRY_COLUMNS
    assert f.equals(enrich_pairwise_geometry(pairwise_positions(positions)))

    # row-major: scattering a column back into a matrix round-trips
    n = len(positions)
    M = np.zeros((n, n))
    M[f["from"], f["to"]] = f["dx"]
    assert np.allclose(M, positions[:, 0][None, :] - positions[:, 0][:, None])


def test_geometric_identities(positions):
    f = pairwise_features(positions)
    assert np.allclose(f[D].values, f[TO].values - f[FROM].values)
    assert set(np.unique(f["sign_dz"])) <= {-1.0, 1.0}

    for (x, y, z), prefix in [(D, "offset"), (FROM, "from")]:
        rho, az, el = (f[f"{prefix}_{c}"] for c in ["rho", "az", "el"])
        assert np.allclose(np.linalg.norm(f[[x, y, z]], axis=1), rho)
        assert np.allclose(np.arctan2(f[y], f[x]), az)
        assert np.allclose(np.arctan2(np.hypot(f[x], f[y]), f[z]), el)


def test_antisymmetry(positions):
    f = pairwise_features(positions).set_index(["from", "to"])
    swapped = f.swaplevel().loc[f.index]
    assert np.allclose(f[D].values, -swapped[D].values)
    assert np.allclose(f["offset_rho"].values, swapped["offset_rho"].values)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        pairwise_features(np.zeros((5, 2)))
    bad = make_positions(5, random_state=1)
    bad[0, 2] = np.nan
    with pytest.raises(ValueError):
        pairwise_features(bad)
