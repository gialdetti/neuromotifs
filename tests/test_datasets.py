import netsci.metrics.motifs as nsm
import numpy as np
import pytest

from neuromotifs import Connectome, load_motifs, load_nmc
from neuromotifs.datasets import MODEL_ORDER, bundled_path, _portal_file

MOTIF_COLUMNS = [f"motif_{i}" for i in range(1, 14)]


@pytest.fixture
def portal_file():
    """The portal H5, resolved as the loader does ($NEUROMOTIFS_DATA, then ~/neuromotifs_data)."""
    try:
        return _portal_file()
    except FileNotFoundError:
        pytest.skip("portal file not available on this machine")


def test_load_motifs_schema():
    motifs = load_motifs()
    assert set(MOTIF_COLUMNS) <= set(motifs.columns)
    assert motifs.model.cat.ordered and list(motifs.model.cat.categories) == MODEL_ORDER

    sizes = motifs.groupby(["celltype", "model"], observed=True).size()
    assert sizes.loc["L5_MC", "dd"] == 1000


def test_bundled_l5_mc_matches_paper():
    connectome = load_nmc("L5_MC")
    assert isinstance(connectome, Connectome)
    assert len(connectome.A) == 395 and connectome.A.sum() == 891
    assert connectome.positions.shape == (395, 3) and connectome.A.dtype == np.int8
    assert not connectome.A.diagonal().any()

    bb = load_motifs().query("celltype == 'L5_MC' and model == 'bb'")
    counts = nsm.motifs(connectome.A.astype(int), algorithm="matmul")
    assert np.array_equal(counts[nsm.triad_order_nn4576], bb[MOTIF_COLUMNS].values[0])


def test_model_frame_is_z_vertical():
    # The portal stores (x, y, z) with y the cortical vertical axis; the model frame puts it on z.
    connectome = load_nmc("L5_MC")
    with bundled_path("L5_MC.npz").open("rb") as f:
        raw = np.load(f)["locations"]
    assert np.array_equal(connectome.positions, raw[:, [0, 2, 1]])


def test_portal_file_equals_bundled_extract(portal_file):
    bundled, portal = load_nmc("L5_MC"), load_nmc("L5_MC", path=portal_file)
    assert np.array_equal(bundled.A, portal.A)
    assert np.allclose(bundled.positions, portal.positions)


def test_missing_portal_file_message(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROMOTIFS_DATA", str(tmp_path))
    monkeypatch.setattr("neuromotifs.datasets.NMC_DEFAULT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="nmc-portal"):
        load_nmc("L5_TTPC2")


def test_connectome_is_immutable():
    with pytest.raises(ValueError):
        load_nmc("L5_MC").A[0, 1] = 1
