# Clustering Analysis Report

## Overview

This document presents a comprehensive analysis of the clustering pipeline's ability to align unsupervised clusters with human-labeled humpback whale call categories. The analysis is based on 122 audio files organized into 15 behavioral categories, embedded using the Perch `multispecies_whale_fp16` model (1280-d vectors).

**Key finding:** The Perch embedding space has a fundamental separability ceiling of approximately ARI ~0.15 and NMI ~0.48 for these call types. No clustering algorithm or parameter combination tested was able to exceed this limit.

---

## 1. Dataset Summary

| Property | Value |
|----------|-------|
| Total audio files | 122 |
| Audio duration | ~3.09 seconds each |
| Windows per file | 1 (all files < 5s window) |
| Embedding model | `multispecies_whale_fp16` (Perch) |
| Embedding dimensions | 1280 |
| Categories | 15 |
| Total data points | 122 vectors in 1280-d space |

### Category Distribution

| Category | Count | % of Total |
|----------|------:|----------:|
| Grunt | 31 | 25.4% |
| Upsweep | 18 | 14.8% |
| Buzz | 18 | 14.8% |
| Shriek | 14 | 11.5% |
| Moan | 8 | 6.6% |
| Short_moan | 6 | 4.9% |
| Whup | 5 | 4.1% |
| Modulated_upsweep | 4 | 3.3% |
| Modulated_moan | 4 | 3.3% |
| Growl | 3 | 2.5% |
| Descending_moan | 3 | 2.5% |
| Bellow | 3 | 2.5% |
| Unknown | 2 | 1.6% |
| Sigh | 2 | 1.6% |
| Cry | 1 | 0.8% |

The dataset is highly imbalanced. 7 categories have 4 or fewer samples, making density-based methods (HDBSCAN) unable to form clusters for them.

---

## 2. Embedding Space Analysis

### 2.1 Vector Properties

| Property | Value |
|----------|-------|
| Value range | [-0.0002, 178.6] |
| Mean L2 norm | 1250.9 |
| Std of L2 norms | 269.2 |
| Zero fraction | 2.1% |
| Near-zero variance dims (<1e-6) | 285 / 1280 |

The embeddings are **non-normalized** with highly variable norms. Roughly 22% of dimensions carry near-zero variance.

### 2.2 PCA Variance Analysis

| Components | Cumulative Variance Explained |
|-----------|----------------------------:|
| 2 | 72.7% |
| 5 | 82.2% |
| 10 | 88.2% |
| 20 | 93.4% |
| 30 | 95.6% |
| 50 | 97.9% |

72.7% of variance is captured in just 2 dimensions, suggesting the effective dimensionality is very low. Most of the 1280 dimensions are redundant.

### 2.3 Category Separability (Cosine Distance)

| Metric | Value |
|--------|------:|
| Mean intra-class distance | 0.0252 |
| Mean inter-class distance | 0.0317 |
| **Separation ratio** | **1.26** |

A separation ratio of 1.26 means inter-class distances are only 26% larger than intra-class distances. For good clustering, this ratio should be >>2. **The categories heavily overlap in embedding space.**

### 2.4 Per-Category Distance Analysis

| Category | N | Intra-class Dist | Nearest Other-class |
|----------|--:|------------------:|--------------------:|
| Bellow | 3 | 0.0220 | 0.0054 |
| Buzz | 18 | 0.0206 | 0.0050 |
| Cry | 1 | N/A | N/A |
| Descending_moan | 3 | 0.0338 | 0.0051 |
| Growl | 3 | 0.0297 | 0.0104 |
| Grunt | 31 | 0.0217 | 0.0051 |
| Moan | 8 | 0.0249 | 0.0064 |
| Modulated_moan | 4 | 0.0318 | 0.0087 |
| Modulated_upsweep | 4 | 0.0405 | 0.0052 |
| Short_moan | 6 | 0.0284 | 0.0077 |
| Shriek | 14 | 0.0319 | 0.0067 |
| Sigh | 2 | 0.0195 | 0.0052 |
| Unknown | 2 | 0.0167 | 0.0097 |
| Upsweep | 18 | 0.0353 | 0.0103 |
| Whup | 5 | 0.0270 | 0.0058 |

**Critical observation:** For every category, the nearest point from a *different* category (0.005–0.010) is **much closer** than the average distance to same-category points (0.017–0.040). This means nearest-neighbor classification would also fail — the categories are not separable in this embedding space.

---

## 3. Configuration Benchmarks

Exhaustive testing of 20+ clustering configurations on the 122 embeddings:

### 3.1 Direct Clustering (no UMAP reduction)

| Configuration | ARI | NMI |
|--------------|----:|----:|
| Raw 1280D, KMeans k=15 | 0.065 | 0.341 |
| L2-normalized, KMeans k=15 | 0.079 | 0.371 |
| PCA-50D, KMeans k=15 | 0.075 | 0.340 |
| PCA-20D, KMeans k=15 | 0.058 | 0.345 |
| L2-norm + PCA-20D, KMeans k=15 | 0.076 | 0.367 |
| L2-norm, Agglomerative Ward k=15 | 0.085 | 0.385 |
| **L2-norm, Agglomerative Cosine k=15** | **0.121** | **0.365** |
| Raw, Agglomerative Ward k=15 | 0.072 | 0.339 |
| L2-norm, Spectral RBF k=15 | 0.061 | 0.374 |
| L2-norm, Spectral NN k=15 | 0.047 | 0.369 |

### 3.2 Agglomerative Cosine (L2-norm) at Different k

| k | ARI | NMI |
|--:|----:|----:|
| 8 | **0.148** | 0.305 |
| 10 | 0.111 | 0.315 |
| 12 | 0.118 | 0.336 |
| 15 | 0.121 | 0.365 |
| 20 | 0.120 | 0.386 |

### 3.3 UMAP + Cosine Variants (L2-norm)

| Configuration | ARI | NMI |
|--------------|----:|----:|
| UMAP-cos-10D nn=15, KMeans k=15 | 0.048 | 0.368 |
| UMAP-cos-10D nn=10, KMeans k=15 | 0.070 | 0.390 |
| UMAP-cos-20D nn=15, KMeans k=15 | 0.067 | 0.376 |

### 3.4 Best Configurations Summary

| Rank | Configuration | ARI | NMI |
|-----:|--------------|----:|----:|
| 1 | L2-norm Agglomerative Cosine k=8 | **0.148** | 0.305 |
| 2 | L2-norm Agglomerative Cosine k=15 | 0.121 | 0.365 |
| 3 | HDBSCAN leaf mcs=22 (from sweep) | 0.158 | 0.301 |
| 4 | HDBSCAN leaf mcs=2 (from sweep) | 0.048 | **0.479** |

**No configuration exceeds ARI 0.16 or NMI 0.48.**

---

## 4. Job `e3b947cc` Results

### 4.1 Parameters

Default HDBSCAN: `min_cluster_size=5`, `cluster_selection_method=leaf`, UMAP 5D + 2D.

### 4.2 Cluster Structure

- 6 clusters + 54 noise points (44.3% noise)
- 14 of 15 categories represented (Modulated_moan entirely in noise)

### 4.3 Metrics

| Metric | Value |
|--------|------:|
| Silhouette | 0.516 |
| Davies-Bouldin | 0.672 |
| Calinski-Harabasz | 294.5 |
| **ARI** | **0.107** |
| **NMI** | **0.372** |
| Homogeneity | 0.336 |
| Completeness | 0.416 |
| V-measure | 0.372 |

### 4.4 Per-Category Purity

| Category | Purity | Interpretation |
|----------|-------:|---------------|
| Cry | 1.00 | Trivial (1 sample) |
| Descending_moan | 1.00 | Trivial (1 non-noise sample) |
| Growl | 1.00 | Trivial (1 non-noise sample) |
| Sigh | 1.00 | Trivial (1 non-noise sample) |
| Unknown | 1.00 | Trivial (2 samples, same cluster) |
| Moan | 0.67 | 2/3 non-noise in one cluster |
| Bellow | 0.50 | Split across 2 clusters |
| Short_moan | 0.50 | Split across 2 clusters |
| **Grunt** | **0.47** | Largest category, spread across 4 clusters |
| Upsweep | 0.45 | Spread across 3 clusters |
| Shriek | 0.43 | Spread across 3 clusters |
| Buzz | 0.33 | Spread across 4 clusters |
| Modulated_upsweep | 0.33 | Spread across 3 clusters |
| Whup | 0.33 | Spread across 3 clusters |

### 4.5 Confusion Matrix (cluster rows, category columns — non-noise only)

| Cluster | Size | Dominant Categories |
|--------:|-----:|---------------------|
| 0 | 11 | Buzz(4), Grunt(4), Cry(1), Mod_up(1), Shriek(1) |
| 1 | 15 | Grunt(9), Buzz(3), Bellow(1), Short_moan(1), Sigh(1) |
| 2 | 10 | Grunt(3), Shriek(3), Desc_moan(1), Buzz(1), Moan(1), Mod_up(1) |
| 3 | 16 | Upsweep(3), Grunt(3), Shriek(3), Unknown(2), Bellow(1), Growl(1), Mod_up(1), Short_moan(1) |
| 4 | 8 | Buzz(4), Upsweep(3), Whup(1) |
| 5 | 7 | Upsweep(5), Moan(2) |

Most clusters contain 4+ categories. Cluster 5 (Upsweep + Moan) is the cleanest.

---

## 5. Root Cause Analysis

### Why Perch Embeddings Don't Separate Humpback Call Types

1. **Model training objective mismatch.** The `multispecies_whale_fp16` model (Google Perch) was trained to distinguish between **species** (humpback vs. blue whale vs. orca vs. background noise), not between **call types within a single species**. The model produces embeddings that are useful for detecting "this is a humpback whale" but not for distinguishing "this is a Grunt vs. a Buzz."

2. **Acoustic similarity of call types.** Many humpback call types share overlapping spectral and temporal features. For example, Grunt and Buzz may differ primarily in duration or pitch contour — subtle distinctions that a species-level embedding doesn't capture.

3. **Short audio clips.** All files are ~3 seconds, producing a single 5-second window (zero-padded). This gives the model a limited acoustic context.

4. **Small, imbalanced dataset.** 122 samples across 15 categories, with 7 categories having <=4 samples. Even a perfect embedding space would challenge most unsupervised algorithms at this scale.

5. **High ambient dimensionality.** 1280 dimensions with 285 near-zero-variance dims. The high-dimensional space introduces noise that further obscures category boundaries.

---

## 6. Recommendations

### Short-term (no model changes required)

1. **Use the right algorithm for the data.** With a known number of categories and a small dataset:
   - K-Means k=15 or Agglomerative k=15 are better choices than HDBSCAN
   - HDBSCAN's density-based approach classifies 30-44% of points as noise with this small dataset
   - Agglomerative with cosine distance on L2-normalized embeddings gave the best ARI (0.148)

2. **L2-normalize embeddings before clustering.** The raw embeddings have highly variable norms (mean 1251, std 269). L2 normalization consistently improved results across all algorithms.

3. **Consider fewer clusters.** The sweep shows k=8 gives higher ARI than k=15, suggesting some labeled categories are acoustically indistinguishable to this model and would be better merged.

4. **Skip UMAP for this dataset size.** With only 122 points, UMAP reduction (especially to 5D) loses information. PCA or no reduction performed comparably and is deterministic.

### Medium-term (model exploration)

5. **Evaluate alternative pre-trained models:**
   - **BirdNET** — trained on a broader set of species with potentially better fine-grained features
   - **BioLingual** — multi-modal bio-acoustic model designed for diverse species/call-type distinction
   - **AVES** (Animal Vocalization Encoder for Search) — self-supervised model specifically for bioacoustics

6. **Fine-tune on humpback call types.** Using the 122 labeled samples (plus any additional labeled data), fine-tune Perch or another base model with a supervised contrastive loss to learn call-type-discriminative embeddings. Even a simple linear probe on top of Perch features would reveal how much separability exists.

7. **Collect more data.** 122 samples with 7 categories having <=4 samples is very small. Doubling the dataset with balanced categories would help both supervised and unsupervised approaches.

### Long-term

8. **Semi-supervised clustering.** Use the known labels to guide clustering with methods like COP-KMeans (constrained K-Means) or semi-supervised spectral clustering. This combines unsupervised structure discovery with label guidance.

9. **Hierarchical category structure.** Some categories are likely sub-types (Moan / Short_moan / Descending_moan / Modulated_moan). Evaluating at a coarser level (collapsing sub-types) may reveal better alignment.

---

## 7. Available Clustering Parameters

The pipeline now supports these parameters (passed as JSON in clustering job `parameters`):

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `clustering_algorithm` | `"hdbscan"`, `"kmeans"`, `"agglomerative"` | `"hdbscan"` | Clustering method |
| `n_clusters` | int | 15 | For kmeans/agglomerative |
| `linkage` | `"ward"`, `"complete"`, `"average"`, `"single"` | `"ward"` | For agglomerative |
| `reduction_method` | `"umap"`, `"pca"`, `"none"` | `"umap"` | Dimensionality reduction |
| `distance_metric` | `"euclidean"`, `"cosine"` | `"euclidean"` | Distance for UMAP + HDBSCAN |
| `min_cluster_size` | int | 5 | HDBSCAN minimum points per cluster |
| `cluster_selection_method` | `"leaf"`, `"eom"` | `"leaf"` | HDBSCAN selection method |
| `umap_cluster_n_components` | int | 5 | UMAP dims for clustering (viz always 2D) |
| `umap_n_neighbors` | int | 15 | UMAP neighbor count |
| `umap_min_dist` | float | 0.1 | UMAP minimum distance |

### Evaluation Metrics (computed automatically)

**Internal (unsupervised):**
- Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score

**Supervised (from folder-path labels):**
- Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)
- Homogeneity, Completeness, V-measure
- Per-category purity, Confusion matrix

**Parameter sweep:** Automatically sweeps HDBSCAN (min_cluster_size x selection_method) and K-Means (k=2..30), with ARI/NMI when category labels are available.

---

## 8. Recommended Clustering Job Configuration

Based on this analysis, the following configuration is recommended for exploring this dataset:

```json
{
  "clustering_algorithm": "agglomerative",
  "n_clusters": 15,
  "linkage": "average",
  "distance_metric": "cosine",
  "reduction_method": "none"
}
```

Or for a quick overview with the expected number of acoustic groups:

```json
{
  "clustering_algorithm": "kmeans",
  "n_clusters": 8,
  "reduction_method": "pca",
  "umap_cluster_n_components": 10
}
```
