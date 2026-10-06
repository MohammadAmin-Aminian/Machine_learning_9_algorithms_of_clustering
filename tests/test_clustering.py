import numpy as np
import pandas as pd
from clustering import models, prepare_data, summarize


def test_imputation_alignment_and_encoding():
    raw = pd.DataFrame(
        {
            "Income": [10, np.nan, 30],
            "Year_Birth": [1990] * 3,
            "Kidhome": [0, 1, 0],
            "Teenhome": [0] * 3,
            "Education": ["A", "B", "A"],
        },
        index=[8, 9, 10],
    )
    display, features = prepare_data(raw)
    assert display.Income.tolist() == [10, 20, 30]
    assert display.index.equals(features.index)
    assert "Education_A" in features
    assert np.isfinite(features).all().all()


def test_noise_and_undefined_scores():
    assert summarize([[0], [1], [2]], [-1, -1, -1])["clusters"] == 0
    assert np.isnan(summarize([[0], [1], [2]], [0, 0, 0])["silhouette"])
    assert summarize([[0], [1], [2]], [0, 0, -1])["noise"] == 1
    assert len(models()) == 9


def test_small_dataset_spectral_and_optics():
    for size in [3, 5, 9]:
        features = np.random.default_rng(42).normal(size=(size, 3))
        for name in ["Spectral", "OPTICS"]:
            model = models()[name]
            labels = model.fit_predict(features)
            assert len(labels) == size
            assert np.isfinite(labels).all()


def test_all_nine_algorithms_on_bundled_data():
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "marketing_campaign.csv"
    raw = pd.read_csv(source, sep="\t").sample(n=120, random_state=42)
    display, features = prepare_data(raw)
    assert display.index.equals(features.index)
    for name, model in models(seed=42).items():
        labels = model.fit_predict(features)
        assert labels.shape == (len(display),), name
        assert np.isfinite(labels).all(), name
        metrics = summarize(features, labels)
        assert 0 <= metrics["noise"] <= len(display), name
        if np.isfinite(metrics["silhouette"]):
            assert -1 <= metrics["silhouette"] <= 1, name
