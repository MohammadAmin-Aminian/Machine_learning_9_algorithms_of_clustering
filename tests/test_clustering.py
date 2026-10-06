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
