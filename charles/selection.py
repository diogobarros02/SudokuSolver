from random import sample, randint, random, choice
import numpy as np
def tournament_sel(population):
    population_size = len(population)
    tournament_size = randint(population_size // 2, population_size)

    # Ensure there are enough individuals for the tournament
    if population_size < tournament_size:
        tournament_size = population_size

    # Randomly select individuals for the tournament
    tournament_individuals = sample(population, tournament_size)

    # Select the best individual from the tournament
    best_individual = min(tournament_individuals, key=lambda ind: ind.fitness)

    # Randomly select a second individual from the tournament (excluding the first)
    second_individual = choice([ind for ind in tournament_individuals if ind != best_individual])

    return best_individual, second_individual
def roulette_wheel_sel(population):
    total_fitness = sum(1/individual.fitness for individual in population)
    roulette_wheel = []
    accumulated_fitness = 0

    # Create a roulette wheel with slots proportional to fitness
    for individual in population:
        probability = (1/individual.fitness) / total_fitness
        accumulated_fitness += probability
        roulette_wheel.append((individual, accumulated_fitness))

    # Spin the wheel to select the first individual
    spin = random()
    for individual, probability in roulette_wheel:
        if spin <= probability:
            selected_individual1 = individual
            break

    # Spin the wheel again to select the second individual
    spin = random()
    for individual, probability in roulette_wheel:
        if spin <= probability:
            selected_individual2 = individual
            break

    return selected_individual1, selected_individual2
def rank_sel(population):
    # Sort the population based on fitness scores
    sorted_population = sorted(population, key=lambda ind: ind.fitness)

    # Assign selection probabilities based on rank order
    selection_probs = np.array([(len(population) - i) / sum(range(1, len(population) + 1)) for i in range(len(population))])

    # Select two individuals based on selection probabilities
    parent1 = np.random.choice(range(len(sorted_population)), p=selection_probs)
    parent2 = np.random.choice(range(len(sorted_population)), p=selection_probs)

    return sorted_population[parent1], sorted_population[parent2]
def random_sel(population):
    parents = sample(population, 2)  # Select two distinct individuals randomly
    return parents

