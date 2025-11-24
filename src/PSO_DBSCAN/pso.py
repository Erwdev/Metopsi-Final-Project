import numpy as np
from dbscan import DBSCAN

class PSO:
    def __init__(self, n_particles, n_dims, max_iter, bounds):
        self.n_particles = n_particles
        self.n_dims = n_dims
        self.max_iter = max_iter
        self.bounds = bounds
        
        self.particles = np.random.uniform(bounds[0], bounds[1], (n_particles, n_dims))
        self.velocities = np.zeros((n_particles, n_dims))
        self.best_positions = self.particles.copy()
        self.best_fitness = np.full(n_particles, -np.inf)
        self.global_best = None
        self.global_best_fitness = -np.inf
    
    def optimize(self, objective_func):
        w, c1, c2 = 0.7, 1.5, 1.5
        
        for iteration in range(self.max_iter):
            # Evaluate fitness untuk semua particles sekaligus (vectorized)
            fitness_scores = np.array([objective_func(self.particles[i]) for i in range(self.n_particles)])
            
            # Update best positions
            improved = fitness_scores > self.best_fitness
            self.best_fitness[improved] = fitness_scores[improved]
            self.best_positions[improved] = self.particles[improved].copy()
            
            # Update global best
            best_idx = np.argmax(self.best_fitness)
            if self.best_fitness[best_idx] > self.global_best_fitness:
                self.global_best_fitness = self.best_fitness[best_idx]
                self.global_best = self.best_positions[best_idx].copy()
            
            # Update velocities dan positions (vectorized)
            r1 = np.random.uniform(0, 1, (self.n_particles, self.n_dims))
            r2 = np.random.uniform(0, 1, (self.n_particles, self.n_dims))
            self.velocities = (w * self.velocities + 
                              c1 * r1 * (self.best_positions - self.particles) + 
                              c2 * r2 * (self.global_best - self.particles))
            self.particles += self.velocities
            self.particles = np.clip(self.particles, self.bounds[0], self.bounds[1])
            
            print(f'Iter {iteration + 1}/{self.max_iter} - Best: {self.global_best_fitness:.4f}')
        
        return self.global_best, self.global_best_fitness



