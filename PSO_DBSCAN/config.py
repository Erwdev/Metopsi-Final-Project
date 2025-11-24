# FILE: config.py

CONFIG = {
    'preprocessing': {
        'random_state': 42
    },

    'embedding': {
        'pca': {
            'n_components': 10
        },
        # UMAP usually gives BETTER clusters for DBSCAN than PCA
        'umap': {
            'n_neighbors': 30,      # Naikkan sedikit agar struktur global lebih terjaga
            'min_dist': 0.0,        # Set 0.0 atau 0.1 agar cluster lebih padat (baik untuk DBSCAN)
            'n_components': 2,      # 2 Dimensi cukup untuk clustering density
            'metric': 'euclidean',
            'random_state': 42
        }
    },

    'pso': {
        'n_particles': 50,         # 50 partikel sudah cukup
        'max_iter': 30,            # Tambah iterasi karena ada damping
        'w': 0.9,                  # Inertia akhir (eksploitasi)
        'c1': 1.5,
        'c2': 1.5,
        'bounds': {
            'eps': (0.1, 5.0),     # Pastikan batas ini masuk akal untuk UMAP (biasanya eps UMAP kecil, 0.1-2.0)
            'min_samples': (3, 50)
        }
    },

    'objective': {
        'alpha': 0.5,  # Kurangi penalti noise sedikit jika terlalu ketat
        'beta': 1.5,   # Perketat jumlah cluster agar tidak jadi 1 cluster raksasa
        'K_min': 2,
        'K_max': 15
    }
}