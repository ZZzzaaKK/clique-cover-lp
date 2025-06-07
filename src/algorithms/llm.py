import networkx as nx
import random
from typing import List, Set, Dict

class LLMHeuristic:
    """
    Implementation of Chalupa's heuristic algorithm for clique coloring.
    """

    def __init__(self, G: nx.Graph, population_size: int = 50, iterations: int = 100,
                 p_crossover: float = 0.8, p_mutation: float = 0.2,
                 local_search_steps: int = 5):
        """
        Initialize the Chalupa heuristic algorithm.

        Args:
            G: NetworkX graph to color
            population_size: Size of the genetic algorithm population
            iterations: Number of iterations to run
            p_crossover: Probability of crossover
            p_mutation: Probability of mutation
            local_search_steps: Number of local search steps per iteration
        """
        self.G = G
        self.population_size = population_size
        self.iterations = iterations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.local_search_steps = local_search_steps

        # Find maximal cliques in the graph
        self.maximal_cliques = list(nx.find_cliques(G))
        self.num_cliques = len(self.maximal_cliques)
        print(f"Found {self.num_cliques} maximal cliques in the graph.")

    def generate_initial_solution(self) -> List[int]:
        """Generate a random initial coloring."""
        # Simple approach: assign random colors from 0 to number of cliques
        return [random.randint(0, self.num_cliques // 2) for _ in range(self.num_cliques)]

    def fitness(self, solution: List[int]) -> float:
        """
        Calculate fitness of a solution. Higher is better.

        Fitness is based on:
        1. Number of colors used (fewer is better)
        2. Number of violations (fewer is better)
        """
        num_colors = len(set(solution))

        # Check for violations (same color assigned to overlapping cliques)
        violations = 0
        for i in range(self.num_cliques):
            for j in range(i+1, self.num_cliques):
                if solution[i] == solution[j]:  # Same color
                    # Check if cliques overlap
                    if not set(self.maximal_cliques[i]).isdisjoint(set(self.maximal_cliques[j])):
                        violations += 1

        # Fitness decreases with more colors and more violations
        # Violations are heavily penalized
        return 1.0 / (num_colors + 10 * violations)

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform crossover between two parent solutions."""
        if random.random() > self.p_crossover:
            return parent1.copy()

        # One-point crossover
        crossover_point = random.randint(1, self.num_cliques - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(self, solution: List[int]) -> List[int]:
        """Mutate a solution."""
        mutated = solution.copy()
        for i in range(self.num_cliques):
            if random.random() < self.p_mutation:
                # Change color to a random one from existing colors or a new one
                existing_colors = list(set(mutated))
                if random.random() < 0.7 and existing_colors:  # 70% chance of using existing color
                    mutated[i] = random.choice(existing_colors)
                else:
                    # Use a new color or reuse one
                    mutated[i] = random.randint(0, len(existing_colors))
        return mutated

    def local_search(self, solution: List[int]) -> List[int]:
        """Apply local search to improve a solution."""
        best_solution = solution.copy()
        best_fitness = self.fitness(best_solution)

        for _ in range(self.local_search_steps):
            # Try changing the color of a random clique
            candidate = best_solution.copy()
            idx = random.randint(0, self.num_cliques - 1)

            # Try to use a color that minimizes conflicts
            colors_to_try = list(set(candidate))
            colors_to_try.append(len(colors_to_try))  # Try a new color too

            for color in colors_to_try:
                candidate[idx] = color
                candidate_fitness = self.fitness(candidate)
                if candidate_fitness > best_fitness:
                    best_solution = candidate.copy()
                    best_fitness = candidate_fitness

        return best_solution

    def solve(self) -> List[Set[int]]:
        """
        Find a clique coloring using the heuristic.

        Returns:
            List of cliques (each represented as a set of nodes)
        """
        # Initialize population
        population = [self.generate_initial_solution() for _ in range(self.population_size)]

        for iteration in range(self.iterations):
            # Evaluate fitness
            fitness_values = [self.fitness(solution) for solution in population]

            # Selection (tournament selection)
            new_population = []
            for _ in range(self.population_size):
                idx1, idx2 = random.sample(range(self.population_size), 2)
                if fitness_values[idx1] > fitness_values[idx2]:
                    selected = population[idx1]
                else:
                    selected = population[idx2]
                new_population.append(selected)

            # Crossover
            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size:
                    child1 = self.crossover(new_population[i], new_population[i+1])
                    child2 = self.crossover(new_population[i+1], new_population[i])
                    new_population[i] = child1
                    new_population[i+1] = child2

            # Mutation
            for i in range(self.population_size):
                new_population[i] = self.mutate(new_population[i])

            # Local search
            for i in range(self.population_size):
                new_population[i] = self.local_search(new_population[i])

            population = new_population

            # Print progress
            if (iteration + 1) % 10 == 0:
                best_idx = fitness_values.index(max(fitness_values))
                best_solution = population[best_idx]
                num_colors = len(set(best_solution))
                print(f"Iteration {iteration + 1}: Best solution uses {num_colors} colors.")

        # Get best solution
        fitness_values = [self.fitness(solution) for solution in population]
        best_idx = fitness_values.index(max(fitness_values))
        best_solution = population[best_idx]

        # Group cliques by color
        color_to_cliques = {}
        for i, color in enumerate(best_solution):
            if color not in color_to_cliques:
                color_to_cliques[color] = []
            color_to_cliques[color].append(set(self.maximal_cliques[i]))

        # Convert to list of cliques
        result = []
        for color, cliques in color_to_cliques.items():
            # If there are multiple cliques with the same color, merge them
            if len(cliques) > 1:
                merged = set()
                for clique in cliques:
                    merged.update(clique)
                result.append(merged)
            else:
                result.append(cliques[0])

        return result
