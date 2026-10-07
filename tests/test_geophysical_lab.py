import numpy as np

from clustering_lab import benchmark_geophysical_models, synthetic_seismic_features


def test_synthetic_geophysical_dataset_is_reproducible_and_bounded():
    first, labels1 = synthetic_seismic_features(samples_per_class=20, seed=7)
    second, labels2 = synthetic_seismic_features(samples_per_class=20, seed=7)
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy())
    np.testing.assert_array_equal(labels1, labels2)
    assert first.shape == (60, 4)
    assert set(labels1) == {0, 1, 2}
    assert (first["dominant_frequency_hz"] > 0).all()
    assert (first["rms_amplitude"] > 0).all()
    assert first["vertical_pressure_coherence"].between(0, 1).all()


def test_all_nine_models_run_on_geophysical_lab():
    results = benchmark_geophysical_models(samples_per_class=30, seed=11)
    assert len(results) == 9
    assert set(results.columns) == {
        "algorithm",
        "clusters",
        "noise",
        "silhouette",
        "adjusted_rand_index",
    }
    assert results["algorithm"].nunique() == 9
    finite_ari = results["adjusted_rand_index"].dropna()
    assert ((finite_ari >= -1) & (finite_ari <= 1)).all()


def test_separated_dataset_has_at_least_one_strong_recovery():
    results = benchmark_geophysical_models(samples_per_class=50, seed=42)
    assert results["adjusted_rand_index"].max() > 0.80
