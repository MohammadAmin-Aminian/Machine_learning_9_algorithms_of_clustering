# Teaching notes: clustering with geophysical features

This exercise uses a **synthetic labelled dataset** to make unsupervised-learning
behaviour measurable without pretending that cluster labels are ground truth in
a real seismic survey.

## Learning goals

Students should be able to explain:

1. why feature scaling changes distance-based clustering;
2. why centroid, hierarchical, density, graph and probabilistic methods can
   produce different partitions of the same data;
3. why silhouette score measures geometric separation rather than scientific
   correctness;
4. how density methods represent noise with label `-1`;
5. why Adjusted Rand Index (ARI) is available in a simulation but usually not
   for genuinely unsupervised field data;
6. why synthetic success does not establish performance on real seismic data.

## Synthetic features

The four features are geophysically inspired:

- dominant frequency;
- spectral slope;
- RMS amplitude;
- vertical/pressure coherence.

Three generating populations are produced with different means and variances.
They are intentionally well separated so that students can first verify expected
algorithm behaviour before making the exercise harder.

## Suggested exercises

- Increase overlap between generating populations and compare ARI degradation.
- Remove feature standardization and identify which algorithms change most.
- Change DBSCAN `eps` and explain the cluster/noise trade-off.
- Compare silhouette and ARI rankings and explain disagreements.
- Add an irrelevant high-variance feature and quantify the effect.
- Replace the synthetic data with responsibly sourced real features, but do not
  treat algorithmic clusters as physical event classes without independent labels.

## Scientific caution

The classes in this tutorial are didactic constructs, not validated earthquake,
noise or infragravity classifiers. The repository demonstrates clustering
methodology and evaluation, not an operational seismic classification system.
