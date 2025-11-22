"""
K-Means Clustering Implementation - Modular Version
Dataset: Medical Check-Up (MCU) Patient Data
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import logging


class KMeansConfig:
    """Configuration class for K-Means parameters"""
    
    def __init__(self, n_clusters: int = 3, max_iterations: int = 100, tolerance: float = 1e-4, random_state: Optional[int] = 42, verbose: bool = True):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.verbose = verbose


class DataPreprocessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        if logger.handlers:
            return logger
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV data
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def extract_numeric_features(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract numeric columns for clustering
        
        Args:
            df: Input dataframe
            exclude_cols: Columns to exclude (e.g., ['BADGE', 'ID'])
            
        Returns:
            Tuple of (numeric_df, feature_names)
        """
        if exclude_cols is None:
            exclude_cols = ['BADGE']
        
        numeric_df = df.select_dtypes(include=[np.number]).drop(columns=exclude_cols, errors='ignore')
        feature_names = numeric_df.columns.tolist()
        
        self.logger.info(f"Extracted {len(feature_names)} numeric features")
        return numeric_df, feature_names
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            df: Input dataframe
            strategy: 'mean', 'median', or 'drop'
            
        Returns:
            DataFrame with handled missing values
        """
        if df.isnull().sum().sum() == 0:
            self.logger.info("No missing values detected")
            return df
        
        if strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif strategy == 'drop':
            df = df.dropna()
        
        self.logger.info(f"Handled missing values using strategy: {strategy}")
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Normalize data to 0-1 range (Min-Max normalization)
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (normalized_array, normalization_params)
        """
        data_array = df.values.astype(float)
        
        min_vals = data_array.min(axis=0)
        max_vals = data_array.max(axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (data_array - min_vals) / range_vals
        
        params = {'min': min_vals, 'max': max_vals}
        
        self.logger.info("Data normalized using Min-Max normalization")
        return normalized, params


class KMeansClusterer:
    """K-Means clustering implementation"""
    
    def __init__(self, config: KMeansConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.centroids = None
        self.labels = None
        self.inertia_history = []
        self.converged = False
        self.n_iterations = 0
        
    def _initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        """
        Initialize centroids randomly from data points
        
        Args:
            data: Input data array (n_samples, n_features)
            
        Returns:
            Initial centroids (n_clusters, n_features)
        """
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
        
        indices = np.random.choice(data.shape[0], self.config.n_clusters, replace=False)
        centroids = data[indices].copy()
        
        self.logger.info(f"Centroids initialized using {self.config.n_clusters} random data points")
        return centroids
    
    def _assign_clusters(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each data point to nearest centroid
        
        Args:
            data: Input data (n_samples, n_features)
            centroids: Current centroids (n_clusters, n_features)
            
        Returns:
            Cluster labels (n_samples,)
        """
        distances = self._compute_distances(data, centroids)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _compute_distances(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between data and centroids
        
        Args:
            data: Input data (n_samples, n_features)
            centroids: Centroids (n_clusters, n_features)
            
        Returns:
            Distance matrix (n_samples, n_clusters)
        """
        return np.array([np.linalg.norm(data - c, axis=1) for c in centroids]).T
    
    def _update_centroids(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids as mean of each cluster
        
        Args:
            data: Input data (n_samples, n_features)
            labels: Cluster labels (n_samples,)
            
        Returns:
            Updated centroids (n_clusters, n_features)
        """
        new_centroids = np.array([
            data[labels == k].mean(axis=0) if np.sum(labels == k) > 0
            else data[np.random.choice(data.shape[0])]
            for k in range(self.config.n_clusters)
        ])
        return new_centroids
    
    def _compute_inertia(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute inertia (sum of squared distances to nearest centroid)
        
        Args:
            data: Input data
            labels: Cluster labels
            
        Returns:
            Inertia value
        """
        return sum(
            np.sum(np.linalg.norm(data[labels == k] - self.centroids[k], axis=1) ** 2)
            for k in range(self.config.n_clusters)
            if np.sum(labels == k) > 0
        )
    
    def fit(self, data: np.ndarray, initial_centroids: np.ndarray = None) -> 'KMeansClusterer':
        """
        Fit K-Means model to data
        
        Args:
            data: Input data array (n_samples, n_features)
            initial_centroids: Optional initial centroids to start from
            
        Returns:
            Self (for method chaining)
        """
        if initial_centroids is not None:
            self.centroids = initial_centroids.copy()
        else:
            self.centroids = self._initialize_centroids(data)
        self.inertia_history = []
        
        for iteration in range(self.config.max_iterations):
            # Assign clusters
            labels = self._assign_clusters(data, self.centroids)
            
            # Compute inertia
            inertia = self._compute_inertia(data, labels)
            self.inertia_history.append(inertia)
            
            # Update centroids
            new_centroids = self._update_centroids(data, labels)
            
            # Check convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            
            if self.config.verbose and iteration % 10 == 0:
                self.logger.info(
                    f"Iteration {iteration}: Inertia = {inertia:.4f}, "
                    f"Centroid shift = {centroid_shift:.6f}"
                )
            
            self.centroids = new_centroids
            self.n_iterations = iteration + 1
            
            if centroid_shift < self.config.tolerance:
                self.converged = True
                self.logger.info(f"Converged at iteration {iteration + 1}")
                break
        
        self.labels = labels
        
        if not self.converged:
            self.logger.warning(f"Did not converge after {self.config.max_iterations} iterations")
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Args:
            data: New data array (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self._assign_clusters(data, self.centroids)
    
    def get_cluster_statistics(self, data: np.ndarray) -> Dict:
        """
        Get statistics for each cluster
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with cluster statistics
        """
        return {
            f'cluster_{k}': {
                'size': np.sum(self.labels == k),
                'percentage': f"{100 * np.sum(self.labels == k) / len(data):.2f}%",
                'centroid': self.centroids[k],
            }
            for k in range(self.config.n_clusters)
        }


class KMeansPipeline:
    """Complete K-Means clustering pipeline"""
    
    def __init__(self, config: KMeansConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        
        self.preprocessor = DataPreprocessor(verbose=verbose)
        self.clusterer = KMeansClusterer(config)
        
        self.data = None
        self.normalized_data = None
        self.norm_params = None
        self.feature_names = None
        self.results = None
    
    def run(
        self,
        filepath: str,
        exclude_cols: List[str] = None,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Run complete pipeline
        
        Args:
            filepath: Path to data file
            exclude_cols: Columns to exclude
            normalize: Whether to normalize data
            
        Returns:
            DataFrame with cluster assignments
        """
        # Load data
        self.data = self.preprocessor.load_data(filepath)
        
        # Extract numeric features
        numeric_data, self.feature_names = self.preprocessor.extract_numeric_features(
            self.data, exclude_cols
        )
        
        # Handle missing values
        numeric_data = self.preprocessor.handle_missing_values(numeric_data)
        
        # Normalize if requested
        if normalize:
            self.normalized_data, self.norm_params = self.preprocessor.normalize_data(numeric_data)
            fitting_data = self.normalized_data
        else:
            fitting_data = numeric_data.values
        
        # Fit clustering
        self.clusterer.fit(fitting_data)
        
        # Create results
        self.results = self.data.copy()
        self.results['cluster'] = self.clusterer.labels
        
        return self.results
    
    def get_summary(self) -> Dict:
        """Get pipeline execution summary"""
        summary = {
            'n_samples': len(self.data),
            'n_features': len(self.feature_names),
            'n_clusters': self.config.n_clusters,
            'converged': self.clusterer.converged,
            'n_iterations': self.clusterer.n_iterations,
            'final_inertia': self.clusterer.inertia_history[-1] if self.clusterer.inertia_history else None,
        }
        
        if self.clusterer.labels is not None:
            summary['cluster_statistics'] = self.clusterer.get_cluster_statistics(
                self.normalized_data if self.normalized_data is not None else self.data
            )
        
        return summary

