"""Datasets: I/O only.

Naming follows scikit-learn: ``load_*`` reads bundled or local files, ``fetch_*`` downloads and
caches published artefacts, ``make_*`` generates synthetic data.

The connectome is the Blue Brain neocortical microcircuit (Markram et al. 2015, Cell), instance
``mc2``, distributed by the NMC portal (https://bbp.epfl.ch/nmc-portal) as
``cons_locs_pathways_mc2_Column.h5`` under an academic-use licence, so it is not redistributed here.
A small extract (``L5_MC``) is bundled so that the package works offline.
"""

import os
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

MODEL_ORDER = ["er", "dd", "ddz", "od", "ld", "bb"]
"""Model names in increasing order; ``bb``, the data itself, last."""

NMC_FILENAME = "cons_locs_pathways_mc2_Column.h5"
NMC_ENV = "NEUROMOTIFS_DATA"
NMC_DEFAULT_DIR = Path("~/neuromotifs_data")


@dataclass(frozen=True)
class Connectome:
    """A cell-type-specific subcircuit: soma positions and binary connectivity.

    ``positions`` (n, 3) are in the model frame, ``x, y`` horizontal and ``z`` the cortical vertical
    axis, in micrometres. ``A[i, j] == 1`` means a connection from neuron ``i`` to neuron ``j``.
    """

    positions: np.ndarray
    A: np.ndarray
    mtype: str
    source: str

    @classmethod
    def from_portal(cls, locations, cmat, mtype, source):
        """Build from the portal's arrays: its vertical axis ``y`` becomes ``z``; ``cMat`` is binarised."""
        positions = np.asarray(locations, dtype=float)[:, [0, 2, 1]]
        A = (np.asarray(cmat) != 0).astype(np.int8)
        if A.shape != (len(positions), len(positions)):
            raise ValueError(
                f"cMat shape {A.shape} does not match {len(positions)} locations"
            )
        positions.setflags(write=False)
        A.setflags(write=False)
        return cls(positions, A, mtype, source)


def load_motifs(path=None) -> pd.DataFrame:
    """The paper's motif counts: 5 m-types x {er, dd, ddz, od, ld} x 1000 samples, plus ``bb``.

    Without ``path``, the copy bundled with the package is loaded.
    """
    source = bundled_path("motifs.csv.gz") if path is None else Path(path)
    with as_file(source) as p:
        df = pd.read_csv(p)
    return df.assign(model=pd.Categorical(df.model, MODEL_ORDER, ordered=True))


def load_nmc(mtype="L5_MC", path=None) -> Connectome:
    """One m-type's subcircuit of the NMC ``mc2`` instance.

    Without ``path``, a bundled extract is used when one exists for ``mtype``; otherwise the portal
    file is looked for in ``$NEUROMOTIFS_DATA`` and then in ``~/neuromotifs_data/``.
    """
    if path is not None:
        return _load_nmc_from_path(mtype, Path(path).expanduser())
    if bundled_path(f"{mtype}.npz").is_file():
        return _load_nmc_from_bundle(mtype)
    return _load_nmc_from_path(mtype, _portal_file())


def _load_nmc_from_bundle(mtype) -> Connectome:
    with bundled_path(f"{mtype}.npz").open("rb") as f:
        z = np.load(f)
        source = f"bundled extract of {z['source']}"
        return Connectome.from_portal(z["locations"], z["cMat"], mtype, source)


def _load_nmc_from_path(mtype, path: Path) -> Connectome:
    with h5py.File(path, "r") as h5:
        locations = h5[f"populations/{mtype}/locations"][:]
        cmat = h5[f"connectivity/{mtype}/{mtype}/cMat"][:]
    return Connectome.from_portal(locations, cmat, mtype, path.name)


def make_positions(n=100, size=(400.0, 400.0, 1000.0), random_state=None) -> np.ndarray:
    """Uniformly random soma positions in a box, for tests and examples."""
    return np.random.default_rng(random_state).random((n, 3)) * np.asarray(
        size, dtype=float
    )


def bundled_path(name):
    """A file bundled under ``neuromotifs/data/nmc``, as an ``importlib.resources`` traversable."""
    return files("neuromotifs.data").joinpath("nmc", name)


def _portal_file() -> Path:
    for folder in (os.environ.get(NMC_ENV), NMC_DEFAULT_DIR):
        if folder and (Path(folder).expanduser() / NMC_FILENAME).is_file():
            return Path(folder).expanduser() / NMC_FILENAME
    raise FileNotFoundError(
        f"{NMC_FILENAME} not found in ${NMC_ENV} or {NMC_DEFAULT_DIR}. Download it from the NMC portal "
        f"(https://bbp.epfl.ch/nmc-portal, academic use) into one of those folders, or pass path=..."
    )
