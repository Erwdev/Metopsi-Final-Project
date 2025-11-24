"""

ACO (Ant Colony Optimization) optimizer for DBSCAN hyperparameters.
This version replaces PSO but maintains the exact same interface.
Strategy: Continuous ACO (ACOR) using Archive-based sampling.
"""

import numpy as np
from typing import Dict, Optional
import random

# relative imports
from .clustering import DBSCAN
from .evaluation import silhouette_score

# -------------------------
# Objective Wrapper (SAMA PERSIS - JANGAN UBAH)
# -------------------------
def _objective_wrapper(X: np.ndarray,
                       labels: np.ndarray,
                       alpha: float,
                       beta: float,
                       K_min: int,
                       K_max: int) -> float:
    """
    Sama persis dengan versi PSO. Tidak ada perubahan logic evaluasi.
    """
    unique = set(labels)
    if len(unique - {-1}) <= 1:
        return 1e6

    mask = labels != -1
    if mask.sum() < 2:
        sil = -1.0
    else:
        sil = silhouette_score(X[mask], labels[mask])

    noise_ratio = float(np.sum(labels == -1) / len(labels))
    num_clusters = len(unique - {-1})

    if K_min <= num_clusters <= K_max:
        cluster_penalty = 0.0
    else:
        if num_clusters < K_min:
            cluster_penalty = float(K_min - num_clusters)
        else:
            cluster_penalty = float(num_clusters - K_max)

    return -float(sil) + float(alpha) * noise_ratio + float(beta) * cluster_penalty


# -------------------------
# Ant Class
# -------------------------
class Ant:
    def __init__(self, bounds: dict):
        # Sama seperti Particle, tapi kita sebut Ant
        self.position = np.array([
            np.random.uniform(*bounds['eps']),
            np.random.uniform(*bounds['min_samples'])
        ], dtype=float)
        self.fitness = np.inf


# -------------------------
# ACO Class (Pengganti PSO)
# -------------------------
class ACO:
    def __init__(self,
                 n_particles: int, # Di config namanya n_particles, kita anggap n_ants
                 max_iter: int,
                 bounds: dict,
                 # Parameter spesifik ACO (bisa diatur di config atau pakai default)
                 q: float = 0.5,       # Intensification factor (seberapa fokus ke best solution)
                 xi: float = 0.85,     # Convergence speed (seberapa cepat deviasi mengecil)
                 logger: Optional[object] = None,
                 objective_kwargs: Optional[dict] = None,
                 **kwargs):            # Kwargs untuk menangkap sisa parameter PSO (w, c1, c2) agar tidak error
        
        # Safety check type bounds
        if not isinstance(bounds, dict):
             raise TypeError(f"Bounds must be a dict. Got {type(bounds)}")

        self.n_ants = int(n_particles) # Mapping n_particles -> n_ants
        self.max_iter = int(max_iter)
        self.bounds = bounds
        
        # ACO Parameters
        self.q = float(q) 
        self.xi = float(xi)

        self.logger = logger
        self.objective_kwargs = objective_kwargs or {'alpha': 0.2, 'beta': 0.2, 'K_min': 3, 'K_max': 8}

        # Archive: Menyimpan solusi terbaik (Jejak Pheromone)
        self.archive = [] 
        self.gbest_position = None
        self.gbest_fitness = np.inf
        self.history = []

    def _select_guide_ant(self):
        """Roulette wheel selection untuk memilih semut pemandu dari archive berdasarkan rank."""
        # Semakin tinggi rank (fitness kecil), semakin besar peluang terpilih
        weights = [1.0 / (i + 1) for i in range(len(self.archive))]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        
        # Pilih index berdasarkan probabilitas
        selected_idx = np.random.choice(len(self.archive), p=probs)
        return self.archive[selected_idx]

    def optimize(self, X: np.ndarray, silent: bool = False) -> Dict:
        """
        Main loop ACO. Interface sama persis dengan PSO.optimize
        """
        dbscan = DBSCAN()

        # 1. Inisialisasi Semut Awal (Random)
        ants = [Ant(self.bounds) for _ in range(self.n_ants)]

        if self.logger and not silent:
            self.logger.info(f"ACO Started: ants={self.n_ants}, iter={self.max_iter}")
            self.logger.start_pbar(total=self.max_iter, desc="ACO Optimization")

        for it in range(self.max_iter):
            # --- A. Evaluasi Semut ---
            for ant in ants:
                eps = float(ant.position[0])
                min_samples = int(max(1, round(ant.position[1])))

                labels = dbscan.fit(X, eps=eps, min_samples=min_samples)

                fitness = _objective_wrapper(
                    X=X, labels=labels, **self.objective_kwargs
                )
                ant.fitness = fitness

                # Update Global Best
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = ant.position.copy()

            # --- B. Update Archive (Pheromone Update) ---
            # Gabungkan semut lama (archive) dengan semut baru, lalu urutkan
            all_solutions = self.archive + ants
            # Sort by fitness (ascending / minimize)
            all_solutions.sort(key=lambda x: x.fitness)
            
            # Keep top k solutions (k = n_ants) -> Ini adalah "Pheromone Table" kita
            self.archive = all_solutions[:self.n_ants]
            
            # Record history
            self.history.append(self.gbest_fitness)
            
            if self.logger and not silent:
                self.logger.update_pbar(1)
                self.logger.info(f"Iter {it+1}/{self.max_iter} - Best Fit: {self.gbest_fitness:.4f}")

            # --- C. Generate Semut Baru (Probabilistic Sampling) ---
            # Menghitung standar deviasi dari archive untuk menentukan seberapa luas pencarian (step size)
            positions = np.array([a.position for a in self.archive])
            sigma = np.std(positions, axis=0) * self.xi  # xi mengontrol kecepatan konvergensi
            
            # Pastikan sigma tidak nol (biar tidak stuck)
            sigma = np.maximum(sigma, 1e-6)

            new_ants = []
            for _ in range(self.n_ants):
                # 1. Pilih "Guide" dari archive (semut yang performanya bagus)
                guide = self._select_guide_ant()
                
                # 2. Buat posisi baru disekitar guide menggunakan Gaussian Distribution
                new_pos = np.random.normal(loc=guide.position, scale=sigma)
                
                # 3. Clamp ke bounds
                new_pos[0] = np.clip(new_pos[0], *self.bounds['eps'])
                new_pos[1] = np.clip(new_pos[1], *self.bounds['min_samples'])
                
                new_ant = Ant(self.bounds)
                new_ant.position = new_pos
                new_ants.append(new_ant)
            
            # Ganti populasi semut untuk iterasi berikutnya
            ants = new_ants

        # --- Selesai ---
        if self.logger and not silent:
            self.logger.close_pbar()
            self.logger.success("ACO finished")

        return {
            'eps': float(self.gbest_position[0]),
            'min_samples': int(round(self.gbest_position[1])),
            'fitness': float(self.gbest_fitness)
        }