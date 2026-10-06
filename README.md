# Nine clustering algorithms — version 2

A reproducible customer-segmentation tutorial comparing K-means, agglomerative,
spectral, DBSCAN, affinity propagation, OPTICS, Gaussian mixtures, mean shift,
and BIRCH. All methods now use the same standardized feature matrix.

## Run

```bash
git clone https://github.com/MohammadAmin-Aminian/Machine_learning_9_algorithms_of_clustering.git
cd Machine_learning_9_algorithms_of_clustering
python -m pip install -r requirements.txt
jupyter notebook clustering_algo.ipynb
```

Run the notebook from top to bottom. The bundled `marketing_campaign.csv` is
 tab-separated. The notebook defaults to a deterministic 400-row subset to keep
quadratic methods practical; change `SAMPLE_SIZE` to `None` for the full dataset.
The example is exploratory, not a validated customer segmentation model.

## Version 2 changes

- Median income imputation is applied to the actual modeling data.
- Categorical features are one-hot encoded rather than assigned arbitrary distances.
- Every algorithm uses identical standardized inputs; plot labels retain row alignment.
- Noise points are excluded from cluster counts and silhouette scores.
- Random seeds and K-means restarts are explicit. Warnings remain visible.
- Stale notebook outputs and misleading unrelated license badges were removed.

Density thresholds and cluster counts are examples and should be tuned for each
sample. Silhouette is undefined for a single cluster or all-noise result, reported
as `NaN`. It is not evidence of scientific or business validity. Age uses the fixed
reference year 2023 to reproduce the original analysis; this dataset predates it.

## Tests and provenance

```bash
python -m pytest -q
```

Dataset source: [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).
Algorithms: [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html).
See `LICENCE.txt` for the existing project license; dataset usage follows its source terms.
Author: Mohammad Amin Aminian.
