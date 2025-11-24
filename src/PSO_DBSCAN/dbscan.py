import numpy as np

class DBSCAN:
    def __init__(self, epsilon=1.0, min_pts=5):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.labels = None
    
    def fit(self, data):
        """DBSCAN dengan cKDTree (jauh lebih cepat)"""
        from scipy.spatial import cKDTree
        
        n_samples = len(data)
        self.labels = np.full(n_samples, -1, dtype=int)
        
        # Build KDTree
        tree = cKDTree(data)
        
        cluster_id = 0
        
        for i in range(n_samples):
            if self.labels[i] != -1:
                continue
            
            # Query neighbors dengan cKDTree (O(log n))
            neighbors = tree.query_ball_point(data[i], self.epsilon)
            
            if len(neighbors) < self.min_pts:
                self.labels[i] = 0  # Noise
                continue
            
            cluster_id += 1
            self.labels[i] = cluster_id
            queue = neighbors[:]
            
            while queue:
                j = queue.pop(0)
                
                if self.labels[j] == -1:
                    self.labels[j] = cluster_id
                    new_neighbors = tree.query_ball_point(data[j], self.epsilon)
                    if len(new_neighbors) >= self.min_pts:
                        queue.extend([n for n in new_neighbors if self.labels[n] == -1])
                elif self.labels[j] == 0:
                    self.labels[j] = cluster_id
        
        return self.labels