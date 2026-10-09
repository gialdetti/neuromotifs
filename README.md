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
from neuromotifs import load_nmc, GeometricModel
import netsci.metrics.motifs as nsm

# a spatial network: soma positions and the adjacency matrix A
connectome = load_nmc("L5_MC")

# a geometric model, 2nd order (distance-dependent), fitted with the paper's classifier
model = GeometricModel("dd").fit(connectome.positions, connectome.A)

# 3 networks drawn from the model, seeds 1300, 1301, 1302 (the paper's)
A_sampled = model.sample(n_samples=3, random_state=1300)

# triplet-motif counts of the connectome and of the samples (netsci counts the motifs)
motifs_bb = nsm.motifs(connectome.A)
motifs_sampled = [nsm.motifs(A) for A in A_sampled]
```
Next:
- **Orders**: `er`, `dd`, `ddz`, `od`, `ld` (constant, distance, bipolar distance, offset, position); `classifier=` swaps in any scikit-learn classifier.
- **The paper's data**: `load_motifs()` ships its motif counts; `motif_counts(A)` names counts the same way; the example notebook checks the pipeline against them.
- **Under the hood**: `model.predict_proba(positions)` gives the `n × n` connection probabilities; `draw_network(P, seed)` draws one network from any such matrix, fitted here or archived.

## Data
- `data/nmc/` contains tiny demonstrators only.
- For full datasets, see `data/README.md` for scripted download instructions.

## Installation
Python 3.10 or newer; a virtual environment is recommended.

Latest version, from GitHub:
```bash
python -m pip install git+https://github.com/gialdetti/neuromotifs.git
```
Append `@<tag-or-commit>` to the URL to pin a version. `python -m pip install neuromotifs` installs the
PyPI release, which is 0.1.0a2 and predates the API above, until the next release.

For development:
```bash
git clone https://github.com/gialdetti/neuromotifs.git && cd neuromotifs
python -m pip install -e ".[dev]"
python -m pytest            # add --runslow for the two regression tests against the paper (about a minute)
```

## Help and Support

### Examples

| Notebook | Run |
| -- | :--: |
| [Generative pipeline: connectome → model → samples → motifs](https://github.com/gialdetti/neuromotifs/blob/main/examples/generative-pipeline.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gialdetti/neuromotifs/blob/main/examples/generative-pipeline.ipynb) |
| [Motif expression levels](https://github.com/gialdetti/neuromotifs/blob/main/examples/motif-expression-levels.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gialdetti/neuromotifs/blob/main/examples/motif-expression-levels.ipynb) |

## Citing
Please cite the paper and this package (see `CITATION.cff`).
