"""
clustering.py

DBSCAN implementation using scikit-learn + post-processing utilities.
"""

import numpy as np
from sklearn.cluster import DBSCAN as SKDBSCAN


class DBSCAN:
    def __init__(self, metric: str = 'euclidean'):
        """
        Wrapper around sklearn DBSCAN.
        """
        self.labels_ = None
        self.metric = metric
        self._model = None


    def fit(self, X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """
        Run DBSCAN on dataset X.
        Returns labels for each point.
        
        -1 = noise
        0, 1, 2, ... = clusters
        """
        self._model = SKDBSCAN(eps=eps, min_samples=min_samples, metric=self.metric)
        self.labels_ = self._model.fit_predict(X)
        return self.labels_


# ============================================================

# Post Processing Utilities

# ============================================================

class PostProcessor:
    """
    Utility class for cleaning cluster outputs.
    """

    @staticmethod
    def remove_noise(X: np.ndarray, labels: np.ndarray):
        """
        Remove points where label == -1.
        Returns:
            X_clean, labels_clean
        """
        mask = labels != -1
        return X[mask], labels[mask]

    @staticmethod
    def map_labels(labels: np.ndarray):
        """
        Remap cluster labels to 0..K-1.
        Noise (-1) remains -1.
        """
        unique = sorted([c for c in np.unique(labels) if c != -1])
        mapping = {old: new for new, old in enumerate(unique)}

        new_labels = labels.copy()
        for old, new in mapping.items():
            new_labels[labels == old] = new

        return new_labels

