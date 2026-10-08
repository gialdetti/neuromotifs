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
# TBD
```

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

| Theme | MyBinder | Colab |
| -- | :--: | :--: |
| [Motif expression levels](https://nbviewer.org/github/gialdetti/neuromotifs/blob/main/examples/motif-expression-levels.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gialdetti/neuromotifs/main?filepath=examples/motif-expression-levels.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gialdetti/neuromotifs/blob/main/examples/motif-expression-levels.ipynb) |
| [Generative pipeline: connectome → model → samples → motifs](https://nbviewer.org/github/gialdetti/neuromotifs/blob/main/examples/generative-pipeline.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gialdetti/neuromotifs/main?filepath=examples/generative-pipeline.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gialdetti/neuromotifs/blob/main/examples/generative-pipeline.ipynb) |

## Citing
Please cite the paper and this package (see `CITATION.cff`).
