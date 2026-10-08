# neuromotifs
Python tools to load neuronal microcircuit geometry, generate geometry-aware null models, and quantify over/under-expression of 3-node motifs.

> Paper: *Neuron Morphological Asymmetry Explains Fundamental Network Stereotypy Across Neocortex* (Gal et al.)

## Highlights
- Motif counting for directed triplets (#1-#13)
- Geometry-driven random graph generators (1st-5th order) mirroring the paper’s models
- Reproducibility notebooks for Figures 1-4
- Simple CLI: `neuromotifs motifs`, `neuromotifs generate`, `neuromotifs fit`

## Quickstart
```python
from neuromotifs import GeometricModel, load_nmc, motif_table, sample_motif_counts

# a spatial network (node position and adjacency matrix)
connectome = load_nmc("L5_MC")

# a geometrical model, 2nd order
model = GeometricModel("dd").fit(connectome.positions, connectome.A)

# n x n connection probabilities
P = model.predict_proba(connectome.positions)

# the connectome's 13 triplet-motif counts
observed = motif_table(connectome.A, model="bb")

# one network per seed, one row each
sampled = sample_motif_counts(P, n_samples=100, base_seed=1300, model="dd")
```
Orders: `er` (constant), `dd` (distance), `ddz` (bipolar distance), `od` (offset), `ld` (location).
Any scikit-learn classifier can replace the paper's: `GeometricModel("od", classifier=HistGradientBoostingClassifier())`.
The paper's own counts ship with the package, `load_motifs()`; the example notebook below checks the pipeline against them.

## Data
- `data/nmc/` contains tiny demonstrators only.
- For full datasets, see `data/README.md` for scripted download instructions.

## Installation
```bash
pip install neuromotifs
# or, for dev
pip install -e .[dev]
```

## Help and Support

### Examples

| Notebook | Run |
| -- | :--: |
| [Generative pipeline: connectome → model → samples → motifs](https://github.com/gialdetti/neuromotifs/blob/main/examples/generative-pipeline.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gialdetti/neuromotifs/blob/main/examples/generative-pipeline.ipynb) |
| [Motif expression levels](https://github.com/gialdetti/neuromotifs/blob/main/examples/motif-expression-levels.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gialdetti/neuromotifs/blob/main/examples/motif-expression-levels.ipynb) |

## Citing
Please cite the paper and this package (see `CITATION.cff`).
