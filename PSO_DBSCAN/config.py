# config.py

CONFIG = {
    # --------------------------------------------------------
    # 1. EMBEDDING STRATEGY (The new switch you added)
    # --------------------------------------------------------
    'embedding': {
        # Options: 'pca' (Linear, faster) or 'umap' (Non-linear, better separation)
        'method': 'pca', 
        
        'pca': {
            'n_components': 10
        },
        
        'umap': {
            # n_neighbors: 15-50 (Local vs Global structure)
            # min_dist: 0.0-0.1 (How tight the clusters are packed)
            'n_neighbors': 30, 
            'min_dist': 0.0,    
            'n_components': 5, 
            'metric': 'euclidean'
        }
    },

    # --------------------------------------------------------
    # 2. PSO HYPERPARAMETERS
    # --------------------------------------------------------
    'pso': {
        'n_particles': 30,
        'max_iter': 14,
        'w': 0.7,      # Inertia
        'c1': 1.5,     # Cognitive (Self)
        'c2': 1.5,     # Social (Swarm)
        # Search space for DBSCAN (Eps, MinSamples)
        'bounds': {
            'eps': (0.1, 5.0),      # Range pencarian epsilon
            'min_samples': (2, 50)  # Range pencarian min_samples
        }
    },
    
    'aco': {  
        'n_particles': 30,    # Ini akan jadi n_ants
        'max_iter': 50,
        
        # Parameter ACO (Opsional, kalau dihapus dia pakai default class)
        'q': 0.5,             # Intensification (Makin kecil makin random, makin besar makin ikut best)
        'xi': 0.85,           # Convergence speed (0.1 cepat, 1.0 lambat/teliti)
        
        # Parameter PSO lama (w, c1, c2) BISA DIHAPUS, atau dibiarkan (tidak akan dipakai)
        
        'bounds': {
            'eps': (0.1, 5.0),
            'min_samples': (2, 50)
        }
    },

    # --------------------------------------------------------
    # 3. OBJECTIVE FUNCTION WEIGHTS
    # --------------------------------------------------------
    'objective': {
        'w_silhouette': 0.6,
        'w_clusters': 0.2, # Penalty for too few/many clusters
        'w_noise': 0.2,    # Penalty for noise points
        'target_clusters': (2, 8) # Preferred number of clusters
    },
    
    # 'objective': {
    #     # 'alpha': Menggantikan 'w_noise'. 
    #     # Penalti jika banyak noise (-1). Semakin besar, semakin "benci" noise.
    #     'alpha': 0.2,  
        
    #     # 'beta': Menggantikan 'w_clusters'. 
    #     # Penalti jika jumlah cluster di luar range target.
    #     'beta': 0.2,   
        
    #     # 'K_min' & 'K_max': Menggantikan 'target_clusters'
    #     # Range jumlah cluster yang diinginkan (inklusif).
    #     'K_min': 3,    
    #     'K_max': 8     
    # }
}