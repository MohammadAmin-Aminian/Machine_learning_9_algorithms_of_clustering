"""Educational clustering lab with a synthetic geophysical feature dataset.

The synthetic dataset is intentionally simple and labelled so students can compare
unsupervised cluster assignments with known generating classes. It is not field
data and should not be interpreted as a validated seismic classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from clustering import models, summarize

FEATURE_NAMES = (
    "dominant_frequency_hz",
    "spectral_slope",
    "rms_amplitude",
    "vertical_pressure_coherence",
)


def synthetic_seismic_features(samples_per_class: int = 120, seed: int = 42):
    """Generate three labelled, geophysically inspired feature populations.

    Classes are deliberately separated to make algorithm behaviour interpretable:
    low-frequency coherent noise, earthquake-like transients, and higher-frequency
    local/noise-like windows. Values are synthetic and have no survey-specific
    physical calibration.
    """
    if not isinstance(samples_per_class, int) or samples_per_class < 10:
        raise ValueError("samples_per_class must be an integer >= 10")

    rng = np.random.default_rng(seed)
    # mean vectors: frequency, spectral slope, RMS amplitude, Z/P coherence
    means = np.array(
        [
            [0.018, -2.1, 0.35, 0.92],
            [0.075, -0.8, 1.40, 0.55],
            [0.220, -0.2, 0.60, 0.25],
        ],
        dtype=float,
    )
    scales = np.array(
        [
            [0.004, 0.20, 0.08, 0.035],
            [0.015, 0.25, 0.20, 0.090],
            [0.035, 0.25, 0.12, 0.080],
        ],
        dtype=float,
    )
    blocks = []
    labels = []
    for label, (mean, scale) in enumerate(zip(means, scales)):
        block = rng.normal(mean, scale, size=(samples_per_class, len(FEATURE_NAMES)))
        block[:, 0] = np.clip(block[:, 0], 0.001, None)
        block[:, 2] = np.clip(block[:, 2], 0.001, None)
        block[:, 3] = np.clip(block[:, 3], 0.0, 1.0)
        blocks.append(block)
        labels.extend([label] * samples_per_class)

    frame = pd.DataFrame(np.vstack(blocks), columns=FEATURE_NAMES)
    truth = np.asarray(labels, dtype=int)
    return frame, truth


def benchmark_geophysical_models(samples_per_class: int = 120, seed: int = 42):
    """Run all nine tutorial models and report unsupervised quality metrics."""
    features, truth = synthetic_seismic_features(samples_per_class, seed)
    scaled = StandardScaler().fit_transform(features)

    rows = []
    for name, model in models(n_clusters=3, seed=seed).items():
        labels = model.fit_predict(scaled)
        metrics = summarize(scaled, labels)
        keep = labels != -1
        ari = np.nan
        if keep.sum() >= 2 and len(np.unique(labels[keep])) >= 2:
            ari = adjusted_rand_score(truth[keep], labels[keep])
        rows.append(
            {
                "algorithm": name,
                "clusters": metrics["clusters"],
                "noise": metrics["noise"],
                "silhouette": metrics["silhouette"],
                "adjusted_rand_index": ari,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["adjusted_rand_index", "silhouette"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark nine clustering algorithms on a synthetic geophysical feature dataset."
    )
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    results = benchmark_geophysical_models(args.samples_per_class, args.seed)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.output:
        if args.output.exists():
            parser.error("output already exists; choose a new path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
