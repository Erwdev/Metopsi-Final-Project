"""
Embedding Module - PCA & UMAP (both using libraries)
Reduces dimensionality of scaled data for visualization and clustering
"""

import numpy as np
from typing import Optional
from sklearn.decomposition import PCA
from umap import UMAP



class PCAEmbedder:
    """
    Principal Component Analysis wrapper using sklearn
    Simplifies interface for the pipeline
    """
    
    def __init__(self, n_components: int = 2, random_state: Optional[int] = None):

        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        
    def fit(self, X: np.ndarray) -> 'PCAEmbedder':

        self.pca.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:

        return self.pca.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        return self.pca.fit_transform(X)
    
    def get_explained_variance_ratio(self) -> np.ndarray:

        return self.pca.explained_variance_ratio_
    
    def get_components(self) -> np.ndarray:

        return self.pca.components_

class UMAPEmbedder:
    """
    UMAP (Uniform Manifold Approximation and Projection) wrapper
    Non-linear dimensionality reduction good for clustering visualization
    """
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        random_state: Optional[int] = None
    ):

        self.umap = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray, **kwargs) -> 'UMAPEmbedder':
        if kwargs:
            self.umap.set_params(**kwargs)
        self.umap.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.umap.transform(X)
    
    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if kwargs:
            self.umap.set_params(**kwargs)
        return self.umap.fit_transform(X)

