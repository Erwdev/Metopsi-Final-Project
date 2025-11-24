"""
Particle Swarm Optimization (PSO) for WCSS Optimization
Mengoptimalkan centroid/parameter K-Means untuk WCSS terbaik dengan K yang fixed
"""

import numpy as np
from typing import Callable, Tuple, Dict, List
import logging


class PSOWCSSOptimizer:
    """PSO untuk mengoptimalkan WCSS dengan K tertentu (fine-tuning centroid)"""
    
    def __init__(
        self,
        data: np.ndarray,
        n_clusters: int,
        initial_centroids: np.ndarray = None,
        n_particles: int = 20,
        n_iterations: int = 50,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        verbose: bool = True
    ):
        """
        Inisialisasi PSO untuk optimasi WCSS
        
        Args:
            data: Training data (n_samples, n_features)
            n_clusters: Jumlah cluster (FIXED)
            initial_centroids: Centroid awal dari K-Means (optional, kalau None pakai random)
            n_particles: Jumlah particle swarm
            n_iterations: Jumlah iterasi PSO
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
            verbose: Print info
        """
        self.data = data
        self.n_clusters = n_clusters
        self.initial_centroids = initial_centroids
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        
        self.logger = self._setup_logger()
        
        # State tracking
        self.best_centroids = None
        self.best_wcss = float('inf')
        self.history = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging - use root logger to avoid duplicates"""
        logger = logging.getLogger(__name__)
        logger.handlers.clear()
        logger.propagate = False
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger
    
    def _compute_wcss(self, centroids: np.ndarray) -> float:
        """
        Compute WCSS untuk centroid tertentu
        
        Args:
            centroids: Centroid positions (n_clusters, n_features)
            
        Returns:
            WCSS value
        """
        # Assign points to nearest centroid
        distances = np.array([np.linalg.norm(self.data - c, axis=1) for c in centroids]).T
        labels = np.argmin(distances, axis=1)
        
        # Compute WCSS
        wcss = sum(
            np.sum(np.linalg.norm(self.data[labels == k] - centroids[k], axis=1) ** 2)
            for k in range(self.n_clusters)
            if np.sum(labels == k) > 0
        )
        
        return wcss
    
    def optimize(self, seed: int = 42) -> Tuple[np.ndarray, float]:
        """
        Run PSO optimization untuk mencari centroid terbaik (meminimalkan WCSS)
        
        Args:
            seed: Random seed
            
        Returns:
            Tuple of (best_centroids, best_wcss)
        """
        np.random.seed(seed)
        
        n_features = self.data.shape[1]
        n_params = self.n_clusters * n_features
        
        # Get data bounds untuk initialize particles
        data_min = self.data.min(axis=0)
        data_max = self.data.max(axis=0)
        
        # Initialize particles
        positions = []
        if self.initial_centroids is not None:
            # Mulai dari initial centroids + moderate random perturbation untuk explore neighborhood
            base_pos = self.initial_centroids.flatten()
            for i in range(self.n_particles):
                perturbation = np.random.uniform(-0.05, 0.05, n_params) * (data_max - data_min).repeat(self.n_clusters)
                pos = base_pos + perturbation
                positions.append(np.clip(pos, np.tile(data_min, self.n_clusters), np.tile(data_max, self.n_clusters)))
            positions = np.array(positions)
        else:
            # Random initialization jika tidak ada initial centroids
            positions = np.array([
                np.random.uniform(
                    np.tile(data_min, self.n_clusters),
                    np.tile(data_max, self.n_clusters)
                )
                for _ in range(self.n_particles)
            ])
        
        velocities = np.random.uniform(-1, 1, (self.n_particles, n_params))
        
        # Personal best
        personal_best_positions = positions.copy()
        personal_best_values = np.array([
            self._compute_wcss(pos.reshape(self.n_clusters, n_features))
            for pos in positions
        ])
        
        # Global best
        best_idx = np.argmin(personal_best_values)
        self.best_centroids = personal_best_positions[best_idx].reshape(self.n_clusters, n_features).copy()
        self.best_wcss = personal_best_values[best_idx]
        
        self.logger.info(f"Starting PSO WCSS optimization with K={self.n_clusters}")
        self.logger.info(f"Particles: {self.n_particles}, Iterations: {self.n_iterations}")
        self.logger.info(f"Initial best WCSS: {self.best_wcss:.6f}")
        
        # Main loop
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                r1 = np.random.rand(n_params)
                r2 = np.random.rand(n_params)
                
                # Update velocity
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                    self.c2 * r2 * (self.best_centroids.flatten() - positions[i])
                )
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Enforce bounds
                positions[i] = np.clip(
                    positions[i],
                    np.tile(data_min, self.n_clusters),
                    np.tile(data_max, self.n_clusters)
                )
                
                # Evaluate WCSS
                centroids = positions[i].reshape(self.n_clusters, n_features)
                wcss = self._compute_wcss(centroids)
                
                # Update personal best
                if wcss < personal_best_values[i]:
                    personal_best_values[i] = wcss
                    personal_best_positions[i] = positions[i].copy()
                
                # Update global best
                if wcss < self.best_wcss:
                    self.best_wcss = wcss
                    self.best_centroids = centroids.copy()
            
            self.history.append(self.best_wcss)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                self.logger.info(
                    f"Iteration {iteration + 1}/{self.n_iterations} - "
                    f"Best WCSS: {self.best_wcss:.6f}"
                )
        
        self.logger.info(f"PSO completed. Final WCSS: {self.best_wcss:.6f}")
        
        return self.best_centroids, self.best_wcss
    
    def get_history(self) -> List[float]:
        """Get optimization history"""
        return self.history
