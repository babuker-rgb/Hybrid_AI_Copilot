# ================================================================
# Multi-Objective Optimization Engine
# Nile Valley University · Sudan · v29.28-R32
# ================================================================

import numpy as np
from typing import List, Tuple, Dict, Optional, Generator
from dataclasses import dataclass
import torch
from model import HybridTabletModel

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    solution: np.ndarray
    objectives: np.ndarray
    generation: int
    rank: int

class NSGAIIOptimizer:
    """Vectorized NSGA-II implementation for multi-objective optimization"""
    
    def __init__(self, 
                 model: HybridTabletModel,
                 n_objectives: int = 3,
                 population_size: int = 50,
                 generations: int = 80,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.1):
        
        self.model = model
        self.n_objectives = n_objectives
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        self.population = None
        self.fitness = None
        
    def initialize_population(self, n_vars: int) -> np.ndarray:
        """Initialize random population"""
        return np.random.rand(self.population_size, n_vars)
    
    def evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """Evaluate objectives for population"""
        with torch.no_grad():
            predictions = self.model.predict_batch(population)
        
        # Objectives: minimize negative density, minimize negative tensile, minimize efrf
        density = predictions[:, 0]
        tensile = predictions[:, 1]
        efrf = predictions[:, 2]
        
        fitness = np.column_stack([-density, -tensile, efrf])
        return fitness
    
    def fast_non_dominated_sort(self, fitness: np.ndarray) -> List[List[int]]:
        """Fast non-dominated sorting algorithm"""
        n = len(fitness)
        fronts = []
        domination_counts = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if np.all(fitness[i] <= fitness[j]) and np.any(fitness[i] < fitness[j]):
                        dominated_solutions[i].append(j)
                    elif np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                        domination_counts[i] += 1
            if domination_counts[i] == 0:
                fronts.append([i])
        
        current_front = 0
        while True:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            if not next_front:
                break
            fronts.append(next_front)
            current_front += 1
            
        return fronts
    
    def crowding_distance(self, fitness: np.ndarray, front: List[int]) -> np.ndarray:
        """Calculate crowding distance for a front"""
        n = len(front)
        if n <= 2:
            return np.ones(n) * np.inf
            
        distances = np.zeros(n)
        for m in range(self.n_objectives):
            front_sorted = sorted(front, key=lambda x: fitness[x][m])
            distances[0] = np.inf
            distances[-1] = np.inf
            for i in range(1, n-1):
                distances[i] += (fitness[front_sorted[i+1]][m] - fitness[front_sorted[i-1]][m]) / \
                              (fitness[front_sorted[-1]][m] - fitness[front_sorted[0]][m] + 1e-10)
        return distances
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Simulated binary crossover"""
        if np.random.random() > self.crossover_prob:
            return parent1.copy()
            
        child = parent1.copy()
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                beta = 1.0 + 2.0 * np.random.random()
                child[i] = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i])
        return np.clip(child, 0, 1)
    
    def mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation"""
        if np.random.random() > self.mutation_prob:
            return individual.copy()
            
        mutant = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < 0.1:
                delta = np.random.normal(0, 0.1)
                mutant[i] = np.clip(mutant[i] + delta, 0, 1)
        return mutant
    
    def optimize(self, n_vars: int) -> Generator[Tuple[np.ndarray, np.ndarray, List[np.ndarray], int], None, None]:
        """Main optimization loop"""
        # Initialize
        self.population = self.initialize_population(n_vars)
        self.fitness = self.evaluate_fitness(self.population)
        
        pareto_history = []
        
        for gen in range(self.generations):
            # Create offspring
            offspring = []
            for _ in range(self.population_size):
                # Tournament selection
                idx1 = np.random.choice(self.population_size, 2, replace=False)
                idx2 = np.random.choice(self.population_size, 2, replace=False)
                
                # Select parents
                parent1 = self.population[np.argmin(self.fitness[idx1].sum(axis=1))]
                parent2 = self.population[np.argmin(self.fitness[idx2].sum(axis=1))]
                
                # Crossover and mutation
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                offspring.append(child)
            
            offspring = np.array(offspring)
            offspring_fitness = self.evaluate_fitness(offspring)
            
            # Combine parent and offspring populations
            combined_pop = np.vstack([self.population, offspring])
            combined_fitness = np.vstack([self.fitness, offspring_fitness])
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(combined_fitness)
            
            # Select new population
            new_population = []
            remaining = self.population_size
            
            for front in fronts:
                if len(new_population) + len(front) <= remaining:
                    new_population.extend(front)
                else:
                    distances = self.crowding_distance(combined_fitness, front)
                    front_sorted = sorted(front, key=lambda x: distances[front.index(x)], reverse=True)
                    new_population.extend(front_sorted[:remaining - len(new_population)])
                    break
            
            self.population = combined_pop[new_population]
            self.fitness = combined_fitness[new_population]
            
            # Store Pareto front
            front0 = fronts[0]
            pareto_solutions = combined_pop[front0]
            pareto_history.append(pareto_solutions)
            
            yield self.population, self.fitness, pareto_history, gen
            
    def get_best_solutions(self, n_solutions: int = 5) -> List[OptimizationResult]:
        """Get best solutions from Pareto front"""
        fronts = self.fast_non_dominated_sort(self.fitness)
        front0 = fronts[0]
        
        solutions = []
        for idx in front0[:n_solutions]:
            solutions.append(OptimizationResult(
                solution=self.population[idx],
                objectives=self.fitness[idx],
                generation=self.generations,
                rank=0
            ))
        
        return solutions
