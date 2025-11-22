"""
Optimization module - Elbow Method untuk mencari K optimal
PSO untuk mengoptimalkan WCSS dengan K tertentu
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging
from k_mean import KMeansConfig, KMeansClusterer
from pso import PSOWCSSOptimizer


class WCSSCalculator:
    """Menghitung WCSS (Within-Cluster Sum of Squares)"""
    
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: Data array (n_samples, n_features)
        """
        self.data = data
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, n_clusters: int, random_state: int = 42) -> float:
        """
        Hitung WCSS untuk jumlah cluster tertentu
        
        Args:
            n_clusters: Jumlah cluster
            random_state: Seed
            
        Returns:
            WCSS value
        """
        config = KMeansConfig(
            n_clusters=int(n_clusters),
            max_iterations=100,
            tolerance=1e-4,
            random_state=random_state,
            verbose=False
        )
        
        clusterer = KMeansClusterer(config)
        clusterer.fit(self.data)
        
        return clusterer.inertia_history[-1] if clusterer.inertia_history else float('inf')


class ElbowOptimizer:
    """Elbow Method untuk optimasi jumlah cluster K"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = self._setup_logger()
        self.wcss_values = []
        self.k_values = []
        self.elbow_point = None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        if logger.handlers:
            return logger
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger
    
    def find_elbow(
        self,
        data: np.ndarray,
        k_range: Tuple[int, int] = (2, 10)
    ) -> Dict:
        """
        Cari elbow point menggunakan Elbow Method
        
        Args:
            data: Input data
            k_range: Range K (min, max)
            
        Returns:
            Dictionary dengan hasil elbow
        """
        self.k_values = list(range(k_range[0], k_range[1] + 1))
        self.wcss_values = []
        
        calculator = WCSSCalculator(data)
        
        self.logger.info(f"Computing WCSS for K in range {k_range}")
        
        for k in self.k_values:
            wcss = calculator.calculate(k)
            self.wcss_values.append(wcss)
            self.logger.info(f"K = {k}: WCSS = {wcss:.6f}")
        
        # Find elbow point menggunakan maksimum curvature
        self.elbow_point = self._find_knee_point()
        
        self.logger.info(f"Suggested K (Elbow point): {self.elbow_point}")
        
        return {
            'k_values': self.k_values,
            'wcss_values': self.wcss_values,
            'elbow_point': self.elbow_point
        }
    
    def _find_knee_point(self) -> int:
        """
        Find knee/elbow point menggunakan curvature analysis
        
        Returns:
            K value di elbow point
        """
        if len(self.wcss_values) < 3:
            return self.k_values[0]
        
        # Normalized WCSS
        wcss_array = np.array(self.wcss_values)
        wcss_normalized = (wcss_array - wcss_array.min()) / (wcss_array.max() - wcss_array.min())
        
        # Normalized K
        k_array = np.array(self.k_values)
        k_normalized = (k_array - k_array.min()) / (k_array.max() - k_array.min())
        
        # Calculate distances dari diagonal line
        distances = []
        for i in range(len(self.k_values)):
            # Distance dari point ke diagonal line
            x0, y0 = k_normalized[i], wcss_normalized[i]
            x1, y1 = k_normalized[0], wcss_normalized[0]
            x2, y2 = k_normalized[-1], wcss_normalized[-1]
            
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            
            distance = numerator / denominator if denominator != 0 else 0
            distances.append(distance)
        
        # Elbow point adalah yang paling jauh dari diagonal
        elbow_idx = np.argmax(distances) + self.k_values[0]
        
        return elbow_idx


class PSOOptimizer:
    """PSO untuk fine-tuning WCSS dengan K tertentu (optimasi centroid)"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = self._setup_logger()
        
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
    
    def optimize_wcss_for_k(
        self,
        data: np.ndarray,
        n_clusters: int,
        initial_centroids: np.ndarray = None,
        n_particles: int = 20,
        n_iterations: int = 50
    ) -> Dict:
        """
        Optimasi WCSS menggunakan PSO dengan jumlah cluster tetap
        PSO mencari centroid terbaik untuk meminimalkan WCSS
        
        Args:
            data: Input data
            n_clusters: Jumlah cluster (FIXED)
            initial_centroids: Centroid awal dari K-Means (untuk better starting point)
            n_particles: Jumlah particle PSO
            n_iterations: Jumlah iterasi PSO
            
        Returns:
            Dictionary dengan hasil optimasi
        """
        self.logger.info(f"Starting PSO WCSS optimization for K={n_clusters}")
        
        # Buat PSO optimizer untuk WCSS
        pso = PSOWCSSOptimizer(
            data=data,
            n_clusters=n_clusters,
            initial_centroids=initial_centroids,
            n_particles=n_particles,
            n_iterations=n_iterations,
            verbose=self.verbose
        )
        
        # Jalankan optimasi
        best_centroids, best_wcss = pso.optimize()
        
        result = {
            'n_clusters': n_clusters,
            'best_centroids': best_centroids,
            'best_wcss': best_wcss,
            'pso_history': pso.get_history()
        }
        
        self.logger.info(f"PSO optimization completed. Best WCSS: {best_wcss:.6f}")
        
        return result


class OptimizationPipeline:
    """Pipeline lengkap untuk optimasi K dan WCSS"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = self._setup_logger()
        self.elbow_optimizer = ElbowOptimizer(verbose=verbose)
        self.pso_optimizer = PSOOptimizer(verbose=verbose)
        self.results = {}
    
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
    
    def run_elbow_method(
        self,
        data: np.ndarray,
        k_range: Tuple[int, int] = (2, 10)
    ) -> Dict:
        """
        Jalankan Elbow Method
        
        Args:
            data: Input data
            k_range: Range K
            
        Returns:
            Dictionary dengan hasil Elbow
        """
        self.logger.info("=" * 60)
        self.logger.info("RUNNING ELBOW METHOD")
        self.logger.info("=" * 60)
        
        elbow_result = self.elbow_optimizer.find_elbow(data, k_range)
        self.results['elbow'] = elbow_result
        
        return elbow_result
    
    def run_pso_wcss_optimization(
        self,
        data: np.ndarray,
        n_clusters: int,
        n_particles: int = 20,
        n_iterations: int = 50
    ) -> Dict:
        """
        Jalankan PSO untuk optimasi WCSS dengan K tertentu
        
        Args:
            data: Input data
            n_clusters: Jumlah cluster (FIXED)
            n_particles: Jumlah particle
            n_iterations: Jumlah iterasi
            
        Returns:
            Dictionary dengan hasil PSO
        """
        self.logger.info("=" * 60)
        self.logger.info(f"RUNNING PSO WCSS OPTIMIZATION (K={n_clusters})")
        self.logger.info("=" * 60)
        
        pso_result = self.pso_optimizer.optimize_wcss_for_k(
            data, n_clusters, n_particles, n_iterations
        )
        self.results['pso_wcss'] = pso_result
        
        return pso_result
    
    def get_summary(self) -> Dict:
        """Get optimization summary"""
        summary = {}
        
        if 'elbow' in self.results:
            elbow = self.results['elbow']
            summary['elbow_method'] = {
                'optimal_k': elbow['elbow_point'],
                'wcss_at_elbow': elbow['wcss_values'][
                    elbow['k_values'].index(elbow['elbow_point'])
                ]
            }
        
        if 'pso_wcss' in self.results:
            pso = self.results['pso_wcss']
            summary['pso_wcss_optimization'] = {
                'n_clusters': pso['n_clusters'],
                'wcss': pso['best_wcss']
            }
        
        return summary
