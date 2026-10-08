# Bundled data

Small files the installed package needs at run time. They are read through `importlib.resources`,
never by path. Anything large or generated (the full connectome, fitted probability matrices,
sampled networks) lives outside the package and outside git; see *Not in this folder* below.

## `nmc/motifs.csv.gz` — the paper's motif counts

The source data of Figures 2–3: triplet-motif counts of the Blue Brain NMC subcircuits and of the
networks sampled from the geometric model hierarchy. 25,005 rows, one per network.

| Column | Meaning |
|---|---|
| `motif_1` … `motif_13` | counts of the 13 connected triplet motifs, in the order of Gal et al. 2017 (*Nat Neurosci*, nn.4576), i.e. `netsci.metrics.motifs(A)[netsci.metrics.motifs.triad_order_nn4576]` |
| `nodes`, `edges`, `reciprocal_edges` | size of the network |
| `celltype` | `L5_MC`, `L5_TTPC1`, `L5_TTPC2`, `L6_IPC`, `L6_LBC` |
| `model` | `bb` the NMC network itself (one row per cell type); `er`, `dd`, `ddz`, `od`, `ld` the model orders, 1,000 samples each |
| `seed` | the sampling seed of that network (`NaN` for `bb`) |
| `hcid`, `structural` | NMC instance (`2` = `mc2`) and connectome stage (`False` = the functional, post-pruning connectome) |
| `regressor`, `regressor_*` | the fitted classifier and its parameters: `GradientBoostingClassifier`, `learning_rate=0.01`, `n_estimators=500`, `max_depth=5`, `random_state=1234` |

Loaded by `neuromotifs.load_motifs()`, which makes `model` an ordered categorical
(`er < dd < ddz < od < ld < bb`).

## `nmc/L5_MC.npz` — a connectome extract for offline use

The `L5_MC` subcircuit (395 neurons, 891 connections) of the NMC `mc2` instance, copied verbatim
from the portal file so that `load_nmc("L5_MC")`, the tests and the examples work without a
download. Keys keep the portal's names and frame:

| Key | Content |
|---|---|
| `locations` | (395, 3) float64 soma positions in the portal frame (`x, y, z`, `y` the cortical vertical axis, µm) |
| `cMat` | (395, 395) int8 binary connectivity, presynaptic neurons along the first axis |
| `mtype`, `source`, `source_sha256` | `L5_MC`; `cons_locs_pathways_mc2_Column.h5`; its SHA-256 `fe25f48a…f18199` |

`load_nmc` converts both this extract and the portal file through the same code, moving the vertical
axis to `z`. The extract was made with:

```python
import h5py, numpy as np

with h5py.File("cons_locs_pathways_mc2_Column.h5") as h5:
    np.savez_compressed(
        "L5_MC.npz",
        locations=h5["populations/L5_MC/locations"][:],
        cMat=h5["connectivity/L5_MC/L5_MC/cMat"][:].astype(np.int8),
        mtype="L5_MC",
        source="cons_locs_pathways_mc2_Column.h5",
        source_sha256="<sha256 of the file>",
    )
```

## Not in this folder

**The full connectome.** `cons_locs_pathways_mc2_Column.h5` (34.7 MB; all 55 m-types, positions and
binary connectivity; Markram et al. 2015, *Cell*) is distributed by the
[NMC portal](https://bbp.epfl.ch/nmc-portal) for academic use and is not redistributed here.
Download it and either pass its path to `load_nmc(mtype, path=...)` or place it in the folder named
by `$NEUROMOTIFS_DATA` or in `~/neuromotifs_data/`.

**Fitted probability matrices and sampled networks.** These are outputs of the package, not inputs;
they are published as release assets and obtained through `fetch_*` functions, cached in the user's
cache directory.
