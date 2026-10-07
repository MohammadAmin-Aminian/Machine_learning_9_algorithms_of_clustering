# Clustering Algorithms Tutorial

[![Regression tests](https://github.com/MohammadAmin-Aminian/Machine_learning_9_algorithms_of_clustering/actions/workflows/tests.yml/badge.svg)](https://github.com/MohammadAmin-Aminian/Machine_learning_9_algorithms_of_clustering/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE.txt)

**A reproducible teaching project comparing nine unsupervised clustering algorithms across two case studies: customer segmentation and a synthetic geophysical feature lab.**

> **Provenance:** this repository is a fork and educational extension. The original MIT license credits **Mohamadhasan Sarvandani (2023)**. That attribution is preserved. The current repository adds reproducible preprocessing, tests/CI, corrected evaluation, package/CLI support, and an original synthetic geophysical clustering exercise. See [NOTICE.md](NOTICE.md).

## Why this repository exists

Clustering tutorials often show several algorithms without explaining why they disagree, how preprocessing changes the geometry, or how to evaluate an unsupervised result responsibly.

This project uses the **same nine algorithms** in controlled workflows so that students can compare:

- centroid-based clustering;
- hierarchical clustering;
- graph-based clustering;
- density-based clustering;
- probabilistic mixture models;
- mode-seeking methods;
- scalable tree-based clustering.

The emphasis is not “which algorithm wins?” but **what assumptions each method makes, what failure modes look like, and what an evaluation metric can and cannot establish**.

## Algorithms

| Family | Algorithm |
|---|---|
| Centroid | K-means |
| Hierarchical | Agglomerative clustering |
| Graph | Spectral clustering |
| Density | DBSCAN |
| Message passing | Affinity propagation |
| Density / reachability | OPTICS |
| Probabilistic | Gaussian mixture model |
| Mode seeking | Mean shift |
| Hierarchical / scalable | BIRCH |

## Two learning tracks

### 1. Customer-segmentation tutorial

The original customer dataset is retained as a familiar entry point. The maintained workflow now includes:

- median imputation for missing income;
- explicit one-hot encoding for categorical variables;
- shared standardized inputs across algorithms;
- deterministic random seeds;
- bounded spectral-neighbor behaviour on small datasets;
- adaptive OPTICS minimum-sample handling;
- explicit density-method noise accounting;
- silhouette scores only when mathematically defined;
- regression tests across all nine methods.

This remains an **exploratory educational example**, not a validated business segmentation model.

### 2. Synthetic geophysical clustering lab

The repository now includes an original teaching exercise in [`clustering_lab.py`](clustering_lab.py) and [`docs/GEOPHYSICAL_TUTORIAL.md`](docs/GEOPHYSICAL_TUTORIAL.md).

It generates a labelled synthetic feature set inspired by seismic/OBS signal analysis:

- dominant frequency;
- spectral slope;
- RMS amplitude;
- vertical/pressure coherence.

Three deliberately separated populations are generated. Because their true generating labels are known, students can compare normal unsupervised metrics with **Adjusted Rand Index (ARI)**.

This allows an important distinction:

- **silhouette score** asks whether clusters are geometrically compact/separated;
- **ARI** asks whether a clustering recovered the known synthetic classes.

Real field data usually do **not** provide such ground truth.

The synthetic classes are didactic constructs, not validated earthquake/noise/infragravity classifiers.

## Quick start

Python 3.10 or newer:

```bash
git clone https://github.com/MohammadAmin-Aminian/Machine_learning_9_algorithms_of_clustering.git
cd Machine_learning_9_algorithms_of_clustering
python -m pip install -e '.[dev]'
python -m pytest -q
```

### Run the geophysical lab

```bash
clustering-lab
```

Example with a CSV report:

```bash
clustering-lab \
    --samples-per-class 120 \
    --seed 42 \
    --output results/geophysical_benchmark.csv
```

The table reports:

- recovered cluster count;
- number of points labelled as noise;
- silhouette score;
- Adjusted Rand Index against the known synthetic labels.

Existing output files are never silently overwritten.

### Run the customer notebook

```bash
jupyter notebook clustering_algo.ipynb
```

The notebook defaults to a deterministic subset to keep quadratic methods practical.

## Core Python API

```python
from clustering import models, prepare_data, summarize
from clustering_lab import (
    benchmark_geophysical_models,
    synthetic_seismic_features,
)

features, truth = synthetic_seismic_features(samples_per_class=120, seed=42)
results = benchmark_geophysical_models(samples_per_class=120, seed=42)
print(results)
```

## Reproducibility

The maintained implementation makes stochastic choices explicit and keeps preprocessing shared across algorithms.

For deterministic local numerical behaviour where BLAS threading may matter:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
```

GitHub Actions:

1. installs the repository as a package;
2. runs the complete test suite;
3. executes the geophysical CLI;
4. builds source and wheel distributions.

## Validation

Tests cover:

- missing-value imputation and row alignment;
- categorical one-hot encoding;
- all-noise and single-cluster edge cases;
- bounded Spectral/OPTICS behaviour on small datasets;
- all nine algorithms on the bundled customer data;
- reproducibility of the synthetic geophysical dataset;
- physical bounds of generated frequency/amplitude/coherence features;
- all nine algorithms on the geophysical exercise;
- ARI validity and recovery of at least one strongly separated synthetic partition.

Run:

```bash
python -m pytest -q
```

## Teaching questions

The geophysical lab is designed to support questions such as:

1. Why does standardization matter for distance-based clustering?
2. Why can DBSCAN call physically plausible observations “noise”?
3. Why can silhouette and ARI rank algorithms differently?
4. What happens when cluster populations overlap?
5. What happens when an irrelevant high-variance feature is added?
6. Why does success on a synthetic labelled dataset not imply field-data validity?

More exercises are in [`docs/GEOPHYSICAL_TUTORIAL.md`](docs/GEOPHYSICAL_TUTORIAL.md).

## Scientific and statistical limitations

- Clustering does not create physical labels by itself.
- Silhouette score measures geometric separation, not scientific truth.
- ARI is only available here because the synthetic generator provides known labels.
- Hyperparameters such as DBSCAN `eps` are dataset-dependent.
- Different algorithms encode different notions of a “cluster.”
- Results from synthetic data cannot be promoted to an operational seismic classifier without independently labelled validation data.

## Repository structure

```text
clustering.py                    maintained nine-algorithm implementation
clustering_lab.py                original synthetic geophysical teaching lab
clustering_algo.ipynb            customer-segmentation notebook
marketing_campaign.csv           tutorial dataset
docs/GEOPHYSICAL_TUTORIAL.md     teaching notes and exercises
tests/                           customer + geophysical regression tests
pyproject.toml                   installable package and CLI metadata
NOTICE.md                        fork provenance and added-work summary
LICENCE.txt                      preserved MIT license
```

## Data and references

Customer dataset: [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

General method reference: [scikit-learn clustering documentation](https://scikit-learn.org/stable/modules/clustering.html)

Dataset use remains subject to the source dataset's own terms.

## Provenance and authorship

The repository remains visibly and intentionally a **fork**.

The existing MIT license credits **Mohamadhasan Sarvandani** as the original copyright holder. It has not been replaced or obscured.

**Maintained educational extension:** Mohammad Amin Aminian

Original additions in this fork include the reproducibility/test infrastructure and the synthetic geophysical clustering lab. See [NOTICE.md](NOTICE.md) for a concise change/provenance record.

## Related research software

This tutorial is separate from the research-software portfolio. For original geophysical research software, see:

- [ComPy](https://github.com/MohammadAmin-Aminian/ComPy) — seafloor compliance processing, DPG calibration and inversion.
- [OBS Transient Cleaner](https://github.com/MohammadAmin-Aminian/Transients) — periodic OBS transient removal.
- [ComPy Inversion Tuner](https://github.com/MohammadAmin-Aminian/Optimization) — inversion-control optimization.
- [RHUM-RUM Geospatial Mapper](https://github.com/MohammadAmin-Aminian/Map) — OBS/bathymetry/tectonic mapping.
- [VRE Seismic Enhancement](https://github.com/MohammadAmin-Aminian/vre-seismic-enhancement) — seismic-resolution enhancement.
- [Gabor Seismic Filter](https://github.com/MohammadAmin-Aminian/gabor-seismic-filter) — orientation-selective seismic filtering.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development and validation workflow.
