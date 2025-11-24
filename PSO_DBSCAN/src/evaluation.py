"""
evaluation.py

Pure evaluation utilities for clustering algorithms.
Implements the 4 functions defined in the project specification.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score as skl_silhouette,
    calinski_harabasz_score,
    davies_bouldin_score
)


# -----------------------------------
# 1. Silhouette Score (Safe Version)
# -----------------------------------
def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes silhouette score safely.
    Works even with DBSCAN where -1 = noise.
    Returns None if silhouette cannot be computed.
    """
    # remove noise for silhouette
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]

    # must have 2 or more clusters
    if len(np.unique(labels_valid)) < 2:
        return None

    try:
        return float(skl_silhouette(X_valid, labels_valid))
    except Exception:
        return None


# -----------------------------------
# 2. Davies-Bouldin Index
# -----------------------------------
def davies_bouldin_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes Davies–Bouldin index.
    Lower is better.
    Returns None when only 1 cluster or invalid labels.
    """
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]

    if len(np.unique(labels_valid)) < 2:
        return None

    try:
        return float(davies_bouldin_score(X_valid, labels_valid))
    except Exception:
        return None


# -----------------------------------
# 3. Calinski–Harabasz Index
# -----------------------------------
def calinski_harabasz_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Higher is better.
    Returns None if cannot compute.
    """
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]

    if len(np.unique(labels_valid)) < 2:
        return None

    try:
        return float(calinski_harabasz_score(X_valid, labels_valid))
    except Exception:
        return None


# -----------------------------------
# 4. Cluster Statistics
# -----------------------------------
def cluster_statistics(labels: np.ndarray) -> dict:
    """
    Returns basic statistics for clusters:
      - number of clusters (excluding -1)
      - cluster sizes
      - noise count
    """
    labels = np.array(labels)
    clusters = labels[labels != -1]
    unique_clusters, counts = np.unique(clusters, return_counts=True)

    stats = {
        "num_clusters": len(unique_clusters),
        "cluster_sizes": dict(zip(unique_clusters.tolist(), counts.tolist())),
        "noise_count": int(np.sum(labels == -1))
    }

    return stats
