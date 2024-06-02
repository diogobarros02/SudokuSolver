import math
from operator import attrgetter
from random import random, shuffle, randint, choice
import numpy as np
import matplotlib.pyplot as plt
import time

class Individual:
    def __init__(self, representation, fixed_mask):
        """Initialize an individual with its representation and fixed mask.
        Calculate its fitness using the get_fitness method."""
        self.size = len(representation)
        self.representation = representation
        self.fixed_mask = fixed_mask
        self.fitness = self.get_fitness()  # Calculate initial fitness

    def constraint_propagation_with_fallback(self):
        """Fill the Sudoku grid using constraint propagation with fallback to random assignment."""
        size = self.size
        subgrid_size = int(math.sqrt(size))

        def is_safe(num, row, col):
            """Check if a number can be safely placed in a cell without violating Sudoku rules."""
            for x in range(size):
                if self.representation[row][x] == num or self.representation[x][col] == num:
                    return False

            start_row = row - row % subgrid_size
            start_col = col - col % subgrid_size

            for i in range(subgrid_size):
                for j in range(subgrid_size):
                    if self.representation[start_row + i][start_col + j] == num:
                        return False
            return True

        def fill_grid():
            """Fill the grid by assigning valid numbers to empty cells or fallback to random numbers."""
            for row in range(size):
                for col in range(size):
                    if self.representation[row][col] == 0:
                        possible_values = [num for num in range(1, size + 1) if is_safe(num, row, col)]
                        if possible_values:
                            self.representation[row][col] = choice(possible_values)
                        else:
                            self.representation[row][col] = randint(1, size)
            return True

        fill_grid()
        return self.representation

    def get_fitness(self):
        """Evaluate the fitness of the individual by counting rule violations."""
        fitness = 0
        size = self.size
        subgrid_size = int(math.sqrt(size))  # Calculate subgrid size

        for i in range(size):
            row = self.representation[i, :]
            col = self.representation[:, i]
            fitness += (size - len(set(num for num in row if num != 0)))  # Count row violations
            fitness += (size - len(set(num for num in col if num != 0)))  # Count column violations

        for i in range(0, size, subgrid_size):
            for j in range(0, size, subgrid_size):
                block = self.representation[i:i + subgrid_size, j:j + subgrid_size].flatten()
                fitness += (size - len(set(num for num in block if num != 0)))  # Count subgrid violations

        return fitness

    def index(self, value):
        """Find the index of a value in the representation."""
        return self.representation.index(value)

    def __len__(self):
        """Return the length of the representation."""
        return len(self.representation)

    def __getitem__(self, position):
        """Get an item from the representation at a specific position."""
        return self.representation[position]

    def __setitem__(self, position, value):
        """Set a value in the representation at a specific position."""
        self.representation[position] = value

    def __repr__(self):
        """Return a string representation of the individual."""
        return f" Fitness: {self.fitness}"

class Population:
    def __init__(self, pop_size, initial_representation):
        """Initialize the population with the given size and initial representation.
        Create individuals with the initial representation and fixed mask."""
        self.pop_size = pop_size
        self.size = initial_representation.shape[0]
        self.fixed_mask = initial_representation != 0
        self.individuals = [
            self.create_individual(initial_representation.copy())
            for _ in range(pop_size)
        ]

    def create_individual(self, initial_representation):
        """Create an individual and process its representation using constraint propagation."""
        temp_individual = Individual(initial_representation, self.fixed_mask)
        processed_representation = temp_individual.constraint_propagation_with_fallback()
        return Individual(processed_representation, self.fixed_mask)

    def random_fill(self, representation, fixed_mask):
        """Randomly fill the unfixed cells in the representation."""
        unfilled_indices = np.argwhere(fixed_mask == 0)
        for i, j in unfilled_indices:
            representation[i, j] = randint(1, self.size)  # Use dynamic grid size
        return representation

    def hamming_distance(self, ind1, ind2):
        """Calculate the Hamming distance between two individuals."""
        return np.sum(ind1.representation != ind2.representation)

    def normalize_distance(self, distance, max_distance):
        """Normalize the distance by the maximum distance."""
        return distance / max_distance

    def sharing_function(self, normalized_distance):
        """Apply the sharing function to the normalized distance."""
        return 1 - normalized_distance

    def get_shared_fitness(self):
        """Calculate and update the shared fitness for each individual in the population."""
        for ind1 in self.individuals:
            distance2ind = [self.hamming_distance(ind1, ind2) for ind2 in self.individuals if ind2 != ind1]

            max_distance = max(distance2ind) if distance2ind else 1  # Avoid division by zero
            norm_distances = [self.normalize_distance(d, max_distance) for d in distance2ind]
            shared_distances = [self.sharing_function(d) for d in norm_distances]

            S_x = sum(shared_distances)
            ind1.fitness = ind1.fitness / S_x if S_x != 0 else ind1.fitness

    def evolve(self, fitness_function, gens, xo_prob, mut_prob, xo, mutate, select, elitism, elitism_prop = 0.1):
        """Evolve the population over a number of generations using the specified genetic operators."""
        fitness = []
        start_time = time.time()  # Start the timer

        for gen in range(1, gens+1):
            new_population = []
            #print("gen", gen)
            if fitness_function == 'basic_fitness_with_sharing':
                self.get_shared_fitness()

            # Elitism: Carry the best individual to the next generation
            if elitism:
                num_elites = max(1, int(elitism_prop * len(self.individuals)))
                elites = sorted(self.individuals, key=attrgetter('fitness'))[:num_elites]
                new_population.extend(elites)

            while len(new_population) < len(self.individuals):
                parent1, parent2 = select(self.individuals)
                parent1_repr, parent2_repr = parent1.representation, parent2.representation

                if random() < xo_prob:
                    offspring1_repr, offspring2_repr = xo(parent1_repr, parent2_repr, parent1.fixed_mask)
                else:
                    offspring1_repr, offspring2_repr = parent1_repr, parent2_repr

                if random() < mut_prob:
                    offspring1_repr = mutate(offspring1_repr, parent1.fixed_mask)
                if random() < mut_prob:
                    offspring2_repr = mutate(offspring2_repr, parent2.fixed_mask)

                new_population.append(Individual(offspring1_repr, parent1.fixed_mask))
                new_population.append(Individual(offspring2_repr, parent2.fixed_mask))

            self.individuals = new_population[:len(self.individuals)]
            best_individual = min(self.individuals, key=attrgetter('fitness'))
            fitness.append(best_individual.get_fitness())

            if best_individual.fitness == 0:
                
                #print("Solution found:")
                #print(best_individual.representation)

                # plt.figure(figsize=(6, 4))
                # plt.plot(range(len(fitness)), fitness, marker='.', linestyle='-')
                # plt.xlabel('Generation')
                # plt.ylabel('Fitness')
                # plt.title('Fitness over Generations')
                # plt.ylim(0, max(fitness) + 2)
                # plt.show()
                
                run_time = (time.time() - start_time) / 60  # Convert to minutes
                return {
                    "best_fitness": best_individual.fitness,
                    "best_individual": best_individual.representation,
                    "run_time": run_time,
                    "convergence_gen": gen
                }
         # plt.figure(figsize=(6, 4))
        # plt.plot(range(len(fitness)), fitness, marker='.', linestyle='-')
        # plt.xlabel('Generation')
        # plt.ylabel('Fitness')
        # plt.title('Fitness over Generations')
        # plt.ylim(0, max(fitness) + 2)
        # plt.show()
        
        run_time = (time.time() - start_time) / 60  # Convert to minutes

        return {
            "best_fitness": best_individual.fitness,
            "best_individual": best_individual.representation,
            "run_time": run_time,
            "convergence_gen": gen
        }

    def __len__(self):
        """Return the number of individuals in the population."""
        return len(self.individuals)

    def __getitem__(self, position):
        """Get an individual at a specific position in the population."""
        return self.individuals[position]