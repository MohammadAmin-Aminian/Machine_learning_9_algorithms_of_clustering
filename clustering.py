"""Reproducible preprocessing and comparison of nine clustering algorithms."""

import numpy as np
import pandas as pd
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    SpectralClustering,
)
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def prepare_data(data, reference_year=2023):
    """Return display features and standardized, one-hot encoded model features."""
    data = data.copy().reset_index(drop=True)
    data["Income"] = pd.to_numeric(data["Income"], errors="raise")
    if not data["Income"].notna().any():
        raise ValueError("Income must contain at least one observed value")
    data["Income"] = data["Income"].fillna(data["Income"].median())
    data["Age"] = reference_year - data["Year_Birth"]
    data["Children"] = data["Kidhome"] + data["Teenhome"]
    data["Is_Parent"] = (data["Children"] > 0).astype(int)
    data = data.rename(
        columns={
            "MntWines": "Wines",
            "MntFruits": "Fruits",
            "MntMeatProducts": "Meat",
            "MntFishProducts": "Fish",
            "MntSweetProducts": "Sweets",
            "MntGoldProds": "Gold",
        }
    )
    data = data.drop(
        columns=[
            "Marital_Status",
            "Dt_Customer",
            "Z_CostContact",
            "Z_Revenue",
            "Year_Birth",
            "ID",
            "Kidhome",
            "Teenhome",
            "AcceptedCmp1",
            "AcceptedCmp2",
            "AcceptedCmp3",
            "AcceptedCmp4",
            "AcceptedCmp5",
        ],
        errors="ignore",
    )
    encoded = pd.get_dummies(data, dtype=float)
    if len(encoded) < 3 or not np.isfinite(encoded.to_numpy(dtype=float)).all():
        raise ValueError("Need at least three rows of finite features")
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(encoded),
        columns=encoded.columns,
        index=data.index,
    )
    return data, scaled


class AdaptiveSpectralClustering(SpectralClustering):
    """Bound graph neighbors and embedding size by the fitted sample count."""

    def fit(self, X, y=None):
        samples = len(X)
        if samples < 2 or self.n_clusters > samples:
            raise ValueError(
                "Need at least two observations and no more clusters than observations"
            )
        neighbors, components = self.n_neighbors, self.n_components
        self.n_neighbors = min(neighbors, samples)
        self.n_components = min(components or self.n_clusters, samples - 1)
        try:
            return super().fit(X, y)
        finally:
            self.n_neighbors, self.n_components = neighbors, components


class AdaptiveOPTICS(OPTICS):
    """Allow the tutorial's three-row minimum without oversized neighborhoods."""

    def fit(self, X, y=None):
        minimum = self.min_samples
        if isinstance(minimum, int):
            self.min_samples = min(minimum, len(X))
        try:
            return super().fit(X, y)
        finally:
            self.min_samples = minimum


def models(n_clusters=3, seed=42):
    return {
        "KMeans": KMeans(n_clusters=n_clusters, n_init=10, random_state=seed),
        "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
        "Spectral": AdaptiveSpectralClustering(
            n_clusters=n_clusters,
            random_state=seed,
            affinity="nearest_neighbors",
            n_neighbors=10,
        ),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
        "AffinityPropagation": AffinityPropagation(damping=0.84, random_state=seed),
        "OPTICS": AdaptiveOPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1),
        "GaussianMixture": GaussianMixture(n_components=n_clusters, random_state=seed),
        "MeanShift": MeanShift(),
        "Birch": Birch(n_clusters=n_clusters),
    }


def summarize(features, labels):
    """Exclude density-method noise (-1); undefined silhouette is NaN."""
    labels = np.asarray(labels)
    features = np.asarray(features)
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("One label is required per observation")
    keep = labels != -1
    clusters = len(np.unique(labels[keep]))
    score = np.nan
    if 1 < clusters < keep.sum():
        score = silhouette_score(
            features[keep], labels[keep], sample_size=1000, random_state=42
        )
    return {"clusters": clusters, "noise": int((~keep).sum()), "silhouette": score}
