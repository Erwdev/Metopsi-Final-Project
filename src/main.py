"""
Main Pipeline - Simple Orchestrator
Hanya call functions dari module lain, no logic di sini
"""

from pathlib import Path
import logging

from k_mean import KMeansConfig, DataPreprocessor, KMeansClusterer
from optimization import OptimizationPipeline
from feature_engineering import FeatureSelector, setup_logger


class ClusteringPipeline:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = setup_logger(__name__, verbose)
        
        # Import dari modul lain
        self.feature_selector = FeatureSelector(verbose=verbose)
        self.optimization = OptimizationPipeline(verbose=verbose)
        self.preprocessor = DataPreprocessor(verbose=verbose)
        
        # State holder
        self.data = None
        self.selected_data = None
        self.selected_features = None
        self.normalized_data = None  # Cache normalized data untuk consistency
        self.scaler_info = None      # Cache min/max untuk consistency
        self.clusterer = None
        self.results = None
    
    def load_data(self, filepath: str):
        """Call preprocessor.load_data()"""
        self.data = self.preprocessor.load_data(filepath)
        return self.data
    
    def select_features(self, category=None, categories=None, custom_features=None):
        """Call feature_selector methods"""
        if category:
            self.selected_data, self.selected_features = self.feature_selector.select_by_category(
                self.data, category
            )
        elif categories:
            self.selected_data, self.selected_features = self.feature_selector.select_multiple_categories(
                self.data, categories
            )
        elif custom_features:
            self.selected_data, self.selected_features = self.feature_selector.select_custom(
                self.data, custom_features
            )
        else:
            raise ValueError("Pilih: category, categories, atau custom_features")
        
        return self.selected_data, self.selected_features
    
    def optimize_k(self, k_range=(2, 10)):
        """Call optimization.run_elbow_method()"""
        # Preprocess & cache normalized data untuk consistency
        numeric_data = self.preprocessor.handle_missing_values(self.selected_data)
        self.normalized_data, self.scaler_info = self.preprocessor.normalize_data(numeric_data)
        
        # Call elbow dari optimization module
        result = self.optimization.run_elbow_method(self.normalized_data, k_range)
        optimal_k = result['elbow_point']
        
        self.logger.info(f"Optimal K: {optimal_k}")
        return {'optimal_k': optimal_k, 'details': result}
    
    def refine_wcss_with_pso(self, n_clusters, n_particles=20, n_iterations=50):
        """Call optimization.pso_optimizer.optimize_wcss_for_k()"""
        # Gunakan normalized data yang sudah di-cache dari optimize_k()
        if self.normalized_data is None:
            raise ValueError("Call optimize_k() first to normalize data")
        
        # Fit K-Means dulu untuk dapatkan initial centroids
        config = KMeansConfig(n_clusters=n_clusters, verbose=False)
        kmeans_temp = KMeansClusterer(config)
        kmeans_temp.fit(self.normalized_data)
        initial_centroids = kmeans_temp.centroids
        
        # Call PSO dari optimization module dengan initial centroids
        result = self.optimization.pso_optimizer.optimize_wcss_for_k(
            self.normalized_data, n_clusters, 
            initial_centroids=initial_centroids,
            n_particles=n_particles, 
            n_iterations=n_iterations
        )
        
        return result
    
    def run_clustering(self, n_clusters=3, max_iterations=100, normalize=True, initial_centroids=None):
        """Call k_mean.KMeansClusterer.fit()"""
        config = KMeansConfig(
            n_clusters=n_clusters,
            max_iterations=max_iterations,
            tolerance=1e-4,
            random_state=42,
            verbose=self.verbose
        )
        
        # Gunakan cached normalized data (consistency!)
        if normalize and self.normalized_data is None:
            numeric_data = self.preprocessor.handle_missing_values(self.selected_data)
            fitting_data, _ = self.preprocessor.normalize_data(numeric_data)
        elif normalize:
            fitting_data = self.normalized_data
        else:
            fitting_data = self.preprocessor.handle_missing_values(self.selected_data).values
        
        # Call KMeansClusterer.fit() dengan optional initial_centroids
        self.clusterer = KMeansClusterer(config)
        self.clusterer.fit(fitting_data, initial_centroids=initial_centroids)
        
        # Add cluster column
        self.results = self.data.copy()
        self.results['cluster'] = self.clusterer.labels
        
        return self.results
    
    def get_cluster_statistics(self):
        """Compute cluster statistics"""
        if self.results is None or 'cluster' not in self.results.columns:
            raise ValueError("Run clustering first")
        
        stats = {}
        for cluster in sorted(self.results['cluster'].unique()):
            cluster_data = self.results[self.results['cluster'] == cluster]
            stats[f'cluster_{cluster}'] = {
                'size': len(cluster_data),
                'percentage': f"{100 * len(cluster_data) / len(self.results):.2f}%"
            }
        
        return stats
    
    def save_results(self, output_path):
        """Save to CSV"""
        if self.results is None:
            raise ValueError("No results to save")
        
        self.results.to_csv(output_path, index=False)
        self.logger.info(f"Results saved: {output_path}")


# ============================================================================
# EXAMPLE: Simple usage
# ============================================================================

def run_example():
    """Simple example - load, select, optimize, cluster, save"""
    data_path = Path(__file__).parent.parent / "data" / "Cleaned_agg_pasien_MCU_2.csv"
    #output_path = Path(__file__).parent.parent / "data" / "MCU_Pasien_With_Cluster_Labels.csv"
    
    # Setup pipeline
    pipeline = ClusteringPipeline(verbose=True)
    
    # Load → Select → Optimize → Cluster
    pipeline.load_data(str(data_path))
    pipeline.select_features(categories=['hematologi'])
    opt = pipeline.optimize_k(k_range=(2, 10))
    
    # Optional: Refine with PSO
    pso = pipeline.refine_wcss_with_pso(n_clusters=opt['optimal_k'], n_particles=30, n_iterations=100)
    
    # Cluster & Save with PSO-optimized centroids
    pso_centroids = pso['best_centroids']
    results = pipeline.run_clustering(n_clusters=opt['optimal_k'], initial_centroids=pso_centroids)
    #pipeline.save_results(str(output_path))
    
    # Show stats
    stats = pipeline.get_cluster_statistics()
    print(f"\nCluster Stats: {stats}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_example()
