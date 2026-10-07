# Clustering Algorithms — Educational Tutorial

[![Regression tests](https://github.com/MohammadAmin-Aminian/Machine_learning_9_algorithms_of_clustering/actions/workflows/tests.yml/badge.svg)](https://github.com/MohammadAmin-Aminian/Machine_learning_9_algorithms_of_clustering/actions/workflows/tests.yml)

**A teaching-oriented machine-learning project comparing nine unsupervised clustering methods on the same customer dataset.**

> **Portfolio context:** This repository is an educational/tutorial project rather than a research contribution. I developed it to demonstrate and compare clustering workflows, preprocessing choices, validation, and the practical differences between common unsupervised-learning algorithms. My research software and geophysical projects are maintained separately on my GitHub profile.

## Learning objectives

The notebook provides a reproducible comparison of:

- K-means
- Agglomerative clustering
- Spectral clustering
- DBSCAN
- Affinity propagation
- OPTICS
- Gaussian mixture models
- Mean shift
- BIRCH

The goal is to show how different clustering families behave when applied to the **same standardized feature matrix**, and to illustrate important practical issues such as categorical encoding, scaling, noise labels, stochastic reproducibility, and silhouette-score limitations.

## Run

```bash
git clone https://github.com/MohammadAmin-Aminian/Machine_learning_9_algorithms_of_clustering.git
cd Machine_learning_9_algorithms_of_clustering
python -m pip install -r requirements.txt
jupyter notebook clustering_algo.ipynb
```

Run the notebook from top to bottom. The bundled `marketing_campaign.csv` is tab-separated. The notebook defaults to a deterministic 400-row subset to keep quadratic methods practical; change `SAMPLE_SIZE` to `None` for the full dataset.

This is an **exploratory teaching example**, not a validated customer-segmentation model. Small datasets use bounded spectral graph neighborhoods and embedding dimensions, and OPTICS adjusts its minimum sample count to the available observations.

## Implementation and reproducibility

- Median income imputation is applied to the modeling data.
- Categorical features are one-hot encoded rather than assigned arbitrary distances.
- All algorithms use identical standardized inputs.
- Plot labels retain row alignment.
- Noise points are excluded from cluster counts and silhouette scores.
- Random seeds and K-means restarts are explicit.
- Warnings remain visible rather than being globally suppressed.

Density thresholds and cluster counts are examples and should be tuned for each dataset. Silhouette score is undefined for a single cluster or an all-noise result and is reported as `NaN`. It should not be interpreted as evidence of scientific or business validity. Age uses the fixed reference year 2023 to reproduce the original analysis.

## Validation

```bash
python -m pytest -q
```

The tests run all nine algorithms on a fixed 120-row sample, check row/label alignment, and verify silhouette-score bounds where defined. The smaller sample keeps CI practical; it is not intended as a full-dataset benchmark.

For more reproducible local numerical behavior:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
```

## Data and references

Dataset: [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)  
Reference: [scikit-learn clustering documentation](https://scikit-learn.org/stable/modules/clustering.html)

See `LICENCE.txt` for the existing project license; dataset usage follows its source terms.

## Provenance and license

This repository is maintained as an educational adaptation/tutorial project. The existing MIT license in `LICENCE.txt` credits **Mohamadhasan Sarvandani** as the original copyright holder. That attribution is preserved.

The current repository adds reproducibility, preprocessing corrections, validation, documentation and test coverage. It should therefore be read as a maintained teaching adaptation rather than a claim of sole authorship of all original material.

**Maintainer:** Mohammad Amin Aminian

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and bug reports.
