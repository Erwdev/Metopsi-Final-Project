"""
optimization.py

PSO optimizer for DBSCAN hyperparameters (eps, min_samples).
This version:
- Uses package-relative imports
- Accepts a logger instance (do not import global logger)
- Calls objective_function with (X, labels, alpha, beta, K_min, K_max)
- Uses PSO to update particle positions and keep pbest/gbest
"""

import numpy as np
from typing import Callable, Dict, Optional

# relative imports so this works when used as package: from src.optimization import PSO
from .clustering import DBSCAN
from .evaluation import silhouette_score  # safe silhouette helper

# -------------------------
# Particle class
# -------------------------
class Particle:
    def __init__(self, bounds: dict):
        # position = [eps (float), min_samples (float -> will be rounded)]
        self.position = np.array([
            np.random.uniform(*bounds['eps']),
            np.random.uniform(*bounds['min_samples'])
        ], dtype=float)

        # small initial velocity
        self.velocity = np.random.uniform(-0.05, 0.05, size=2)
        self.pbest = self.position.copy()
        self.pbest_fitness = np.inf


# -------------------------
# objective function (kept as helper wrapper expected by PSO)
# This is a "callable" that PSO will call after DBSCAN produced labels.
# It delegates to the canonical objective signature below.
# -------------------------
def _objective_wrapper(X: np.ndarray,
                       labels: np.ndarray,
                       alpha: float,
                       beta: float,
                       K_min: int,
                       K_max: int) -> float:
    """
    Keep consistency: this matches the canonical signature
    objective_function(X, labels, alpha, beta, K_min, K_max)
    """
    # handle degenerate cases (all noise or <2 clusters)
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

    # cluster penalty - zero if within [K_min, K_max]
    if K_min <= num_clusters <= K_max:
        cluster_penalty = 0.0
    else:
        if num_clusters < K_min:
            cluster_penalty = float(K_min - num_clusters)
        else:
            cluster_penalty = float(num_clusters - K_max)

    # final objective: minimize
    return -float(sil) + float(alpha) * noise_ratio + float(beta) * cluster_penalty


# -------------------------
# PSO class
# -------------------------
class PSO:
    def __init__(self,
                 n_particles: int,
                 max_iter: int,
                 bounds: dict,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 logger: Optional[object] = None,
                 objective_kwargs: Optional[dict] = None):
        """
        bounds: dict with 'eps' and 'min_samples' tuples
                e.g. {'eps': (0.1, 5.0), 'min_samples': (3, 50)}
        logger: instance of your Logger (optional). If provided, PSO uses
                logger.start_pbar / update_pbar / close_pbar and logger.info/success.
        objective_kwargs: {'alpha':..., 'beta':..., 'K_min':..., 'K_max':...}
        """
        self.n_particles = int(n_particles)
        self.max_iter = int(max_iter)
        self.bounds = bounds
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.logger = logger
        self.objective_kwargs = objective_kwargs or {'alpha': 1.0, 'beta': 1.0, 'K_min': 2, 'K_max': 10}

        self.particles = [Particle(bounds) for _ in range(self.n_particles)]
        self.gbest = None
        self.gbest_fitness = np.inf
        # store history for plotting
        self.history = []

    def optimize(self, X: np.ndarray, silent: bool = False) -> Dict:
        """
        Runs PSO to optimize DBSCAN params on data X (embedded).
        Returns: dict {'eps':..., 'min_samples':..., 'fitness':...}
        """

        dbscan = DBSCAN()

        if self.logger and not silent:
            self.logger.info(f"PSO started: particles={self.n_particles}, iter={self.max_iter}")
            self.logger.start_pbar(total=self.max_iter, desc="PSO Optimization")

        for it in range(self.max_iter):
            best_fitness_iter = np.inf
            best_position_iter = None

            for p in self.particles:
                # decode parameters
                eps = float(p.position[0])
                min_samples = int(max(1, round(p.position[1])))

                # run DBSCAN to get labels
                labels = dbscan.fit(X, eps=eps, min_samples=min_samples)

                # evaluate objective using canonical signature
                fitness = _objective_wrapper(
                    X=X,
                    labels=labels,
                    alpha=self.objective_kwargs.get('alpha', 1.0),
                    beta=self.objective_kwargs.get('beta', 1.0),
                    K_min=self.objective_kwargs.get('K_min', 2),
                    K_max=self.objective_kwargs.get('K_max', 10)
                )

                # update personal best
                if fitness < p.pbest_fitness:
                    p.pbest = p.position.copy()
                    p.pbest_fitness = fitness

                # track iteration best
                if fitness < best_fitness_iter:
                    best_fitness_iter = fitness
                    best_position_iter = p.position.copy()

                # update global best
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest = p.position.copy()

            # update velocities and positions after evaluating all particles
            for p in self.particles:
                r1 = np.random.rand()
                r2 = np.random.rand()

                cognitive = self.c1 * r1 * (p.pbest - p.position)
                social = self.c2 * r2 * (self.gbest - p.position)

                p.velocity = self.w * p.velocity + cognitive + social
                p.position = p.position + p.velocity

                # clamp to bounds
                p.position[0] = np.clip(p.position[0], *self.bounds['eps'])
                p.position[1] = np.clip(p.position[1], *self.bounds['min_samples'])

            # record history and update logger progress
            self.history.append(self.gbest_fitness if self.gbest is not None else np.inf)
            if self.logger and not silent:
                self.logger.update_pbar(1)
                self.logger.info(f"Iter {it+1}/{self.max_iter} - best_fitness={self.gbest_fitness:.4f}")

        # close progress bar and final log
        if self.logger and not silent:
            self.logger.close_pbar()
            self.logger.success("PSO finished")

        # prepare return
        if self.gbest is None:
            # fallback: choose best particle
            best_particle = min(self.particles, key=lambda p: p.pbest_fitness)
            gbest = best_particle.pbest.copy()
            gbest_fitness = best_particle.pbest_fitness
        else:
            gbest = self.gbest.copy()
            gbest_fitness = self.gbest_fitness

        result = {
            'eps': float(gbest[0]),
            'min_samples': int(round(gbest[1])),
            'fitness': float(gbest_fitness)
        }

        return result
