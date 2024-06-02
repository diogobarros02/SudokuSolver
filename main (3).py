from charles.charles import Population
from charles.mutation import random_mutation, single_mutation, valid_mutation, subgrid_mutation, trial_mutation
from charles.selection import random_sel, roulette_wheel_sel, tournament_sel, rank_sel
from charles.xo import single_point_xo, cycle_xo, swap_xo, subgrid_xo
import numpy as np
import time
from tabulate import tabulate

# Sample Sudoku problems and their respective solutions
ex_easy = np.array([
    [5, 3, 0, 0, 7, 0, 0, 1, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 5, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 5, 0, 8, 0, 0, 7, 9]
])
ex_medium = np.array([
    [5, 0, 7, 2, 0, 0, 0, 9, 0],
    [0, 0, 6, 0, 3, 0, 7, 0, 1],
    [4, 0, 0, 0, 0, 0, 0, 6, 0],
    [1, 0, 0, 4, 9, 0, 0, 0, 7],
    [0, 0, 0, 5, 0, 8, 0, 0, 0],
    [8, 0, 0, 0, 2, 7, 0, 0, 5],
    [0, 7, 0, 0, 0, 0, 0, 0, 9],
    [2, 0, 9, 0, 8, 0, 6, 0, 0],
    [0, 4, 0, 0, 0, 9, 3, 0, 8]
])
ex_hard = np.array([
    [8, 0, 0, 0, 0, 0, 0, 0, 6],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 7, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 3, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0]
])
ex_extreme = np.array([
    [0, 0, 0, 4, 0, 0, 0, 7, 1],
    [0, 8, 0, 0, 3, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 1, 0, 4, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 8, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 2, 0, 9, 0, 0],
    [7, 0, 4, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 8]
])
sol_easy = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9]
])
sol_medium = np.array([
    [5, 1, 7, 2, 6, 4, 8, 9, 3],
    [9, 2, 6, 8, 3, 5, 7, 4, 1],
    [4, 8, 3, 9, 7, 1, 5, 6, 2],
    [1, 3, 5, 4, 9, 6, 2, 8, 7],
    [7, 9, 2, 5, 1, 8, 4, 3, 6],
    [8, 6, 4, 3, 2, 7, 9, 1, 5],
    [3, 7, 8, 6, 4, 2, 1, 5, 9],
    [2, 5, 9, 1, 8, 3, 6, 7, 4],
    [6, 4, 1, 7, 5, 9, 3, 2, 8]
])
sol_hard = np.array([
    [8, 1, 4, 7, 3, 2, 9, 5, 6],
    [9, 2, 3, 6, 5, 8, 1, 4, 7],
    [6, 7, 5, 4, 4, 1, 6, 8, 3],
    [4, 5, 9, 3, 8, 7, 5, 2, 1],
    [1, 3, 6, 2, 4, 5, 7, 9, 8],
    [2, 8, 7, 1, 6, 9, 5, 3, 4],
    [5, 4, 1, 9, 7, 3, 8, 6, 2],
    [7, 6, 8, 5, 2, 4, 3, 1, 9],
    [3, 9, 2, 8, 1, 6, 4, 7, 5]
])
sol_extreme = np.array([
    [9, 3, 5, 4, 6, 8, 2, 7, 1],
    [2, 8, 1, 5, 3, 7, 4, 6, 9],
    [6, 4, 7, 9, 1, 2, 5, 8, 3],
    [5, 7, 3, 1, 8, 4, 6, 9, 2],
    [4, 1, 2, 6, 9, 3, 8, 5, 7],
    [8, 9, 6, 2, 7, 5, 1, 3, 4],
    [3, 6, 8, 7, 2, 1, 9, 4, 5],
    [7, 2, 4, 8, 5, 9, 3, 1, 6],
    [1, 5, 9, 3, 4, 6, 7, 2, 8]
])
ex4_easy = np.array([
    [1, 0, 0, 0],
    [3, 2, 4, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0]
])
ex4_extreme = np.array([
    [1, 0, 0, 0],
    [4, 0, 3, 1],
    [0, 0, 0, 3],
    [0, 0, 0, 0]
])
ex16_medium = np.array([
    [0, 6, 0, 0, 0, 0, 0, 8, 11, 0, 0, 15, 14, 0, 0, 16],
    [15, 11, 0, 0, 0, 16, 14, 0, 0, 0, 12, 0, 0, 6, 0, 0],
    [13, 0, 9, 12, 0, 0, 0, 0, 3, 16, 14, 0, 15, 11, 10, 0],
    [2, 0, 16, 0, 11, 0, 15, 10, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 15, 11, 10, 0, 0, 16, 2, 13, 8, 9, 12, 0, 0, 0, 0],
    [12, 13, 0, 0, 4, 1, 5, 6, 2, 3, 0, 0, 0, 0, 11, 10],
    [5, 0, 6, 1, 12, 0, 9, 0, 15, 11, 10, 7, 16, 0, 0, 3],
    [0, 2, 0, 0, 0, 10, 0, 11, 6, 0, 5, 0, 0, 13, 0, 9],
    [10, 7, 15, 11, 16, 0, 0, 0, 12, 13, 0, 0, 0, 0, 0, 6],
    [9, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 16, 10, 0, 0, 11],
    [1, 0, 4, 6, 9, 13, 0, 0, 7, 0, 11, 0, 3, 16, 0, 0],
    [16, 14, 0, 0, 7, 0, 10, 15, 4, 6, 1, 0, 0, 0, 13, 8],
    [11, 10, 0, 15, 0, 0, 0, 16, 9, 12, 13, 0, 0, 1, 5, 4],
    [0, 0, 12, 0, 1, 4, 6, 0, 16, 0, 0, 0, 11, 10, 0, 0],
    [0, 0, 5, 0, 8, 12, 13, 0, 10, 0, 0, 11, 2, 0, 0, 14],
    [3, 16, 0, 0, 10, 0, 0, 7, 0, 0, 6, 0, 0, 0, 12, 0]
])
sol16_medium = np.array([
    [7, 6, 14, 3, 10, 12, 4, 8, 11, 13, 1, 15, 14, 9, 2, 16],
    [15, 11, 3, 8, 6, 16, 14, 7, 1, 10, 12, 2, 9, 6, 13, 5],
    [13, 8, 9, 12, 2, 7, 11, 5, 3, 16, 14, 1, 15, 11, 10, 4],
    [2, 9, 16, 5, 11, 14, 15, 10, 1, 4, 3, 6, 8, 12, 7, 13],
    [6, 15, 11, 10, 3, 2, 16, 2, 13, 8, 9, 12, 5, 7, 4, 1],
    [12, 13, 7, 9, 4, 1, 5, 6, 2, 3, 8, 14, 10, 15, 11, 10],
    [5, 4, 6, 1, 12, 8, 9, 14, 15, 11, 10, 7, 16, 2, 12, 3],
    [4, 2, 8, 7, 3, 10, 2, 11, 6, 1, 5, 14, 12, 13, 16, 9],
    [10, 7, 15, 11, 16, 3, 8, 13, 12, 13, 2, 9, 1, 5, 14, 6],
    [9, 5, 1, 14, 15, 9, 1, 12, 8, 2, 7, 16, 10, 4, 3, 11],
    [1, 3, 4, 6, 9, 13, 2, 14, 7, 5, 11, 3, 3, 16, 9, 15],
    [16, 14, 13, 2, 7, 5, 10, 15, 4, 6, 1, 12, 6, 3, 13, 8],
    [11, 10, 2, 15, 5, 3, 7, 16, 9, 12, 13, 8, 4, 1, 5, 4],
    [8, 1, 12, 16, 1, 4, 6, 3, 16, 15, 2, 5, 11, 10, 14, 7],
    [14, 12, 5, 4, 8, 12, 13, 9, 10, 7, 6, 11, 2, 8, 1, 14],
    [3, 16, 10, 13, 10, 11, 3, 7, 14, 9, 6, 4, 13, 11, 12, 2]
])


pop_size = 5000
gens = 100
num_iterations = 5

# Configurations
configs = [
    {"ID": "GA1", "Selection": tournament_sel, "Crossover": single_point_xo, "Mutation": single_mutation,
     "Elitism": True, "FitnessFunction": 'basic_fitness'},
    {"ID": "GA2", "Selection": roulette_wheel_sel, "Crossover": cycle_xo, "Mutation": valid_mutation, "Elitism": True,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA3", "Selection": random_sel, "Crossover": subgrid_xo, "Mutation": trial_mutation, "Elitism": True,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA4", "Selection": tournament_sel, "Crossover": cycle_xo, "Mutation": subgrid_mutation, "Elitism": True,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA5", "Selection": roulette_wheel_sel, "Crossover": swap_xo, "Mutation": single_mutation, "Elitism": True,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA6", "Selection": random_sel, "Crossover": swap_xo, "Mutation": random_mutation, "Elitism": True,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA7", "Selection": random_sel, "Crossover": cycle_xo, "Mutation": subgrid_mutation, "Elitism": True,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA8", "Selection": random_sel, "Crossover": single_point_xo, "Mutation": valid_mutation, "Elitism": True,
     "FitnessFunction": 'basic_fitness'},
]

# Run and collect results
results = {}
for config in configs:
    cumulative_runtime = 0
    cumulative_fitness = 0
    best_runtime = float('inf')
    best_fitness = float('inf')
    #print(config["ID"])

    for i in range(num_iterations):
        #print(i)
        # Create a new population for each configuration
        population = Population(pop_size, ex_medium)

        time0 = time.time()
        result = population.evolve(fitness_function=config["FitnessFunction"],
                                   gens=gens, xo_prob=0.85, mut_prob=0.25,
                                   select=config["Selection"],
                                   xo=config["Crossover"],
                                   mutate=config["Mutation"],
                                   elitism=config["Elitism"],
                                   elitism_prop=0.05)
        time1 = time.time()

        runtime = (time1 - time0) / 60  # Convert to minutes
        fitness = result["best_fitness"]

        cumulative_runtime += runtime
        cumulative_fitness += fitness
        if runtime < best_runtime:
            best_runtime = runtime
        if fitness < best_fitness:
            best_fitness = fitness

    avg_runtime = cumulative_runtime / num_iterations
    avg_fitness = cumulative_fitness / num_iterations

    results[config["ID"]] = {
        "avg_runtime": avg_runtime,
        "avg_fitness": avg_fitness,
        "best_runtime": best_runtime,
        "best_fitness": best_fitness
    }

# Create a list of rows for the table
results_table = []
for id, result in results.items():
    results_table.append([
        id,
        f"{result['avg_runtime']:.2f}",
        result['avg_fitness'],
        f"{result['best_runtime']:.2f}",
        result['best_fitness']
    ])

# Define the column headers
headers = ["ID", "Average Runtime (min)", "Average Fitness", "Best Runtime (min)", "Best Fitness"]

# Print the results table using tabulate
print("Results for each GA configuration:")
print(tabulate(results_table, headers=headers, tablefmt="grid"))


# Running the same GA's without Elitism

configs = [
    {"ID": "GA1", "Selection": tournament_sel, "Crossover": single_point_xo, "Mutation": single_mutation,
     "Elitism": False, "FitnessFunction": 'basic_fitness'},
    {"ID": "GA2", "Selection": roulette_wheel_sel, "Crossover": cycle_xo, "Mutation": valid_mutation, "Elitism": False,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA3", "Selection": random_sel, "Crossover": subgrid_xo, "Mutation": trial_mutation, "Elitism": False,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA4", "Selection": tournament_sel, "Crossover": cycle_xo, "Mutation": subgrid_mutation, "Elitism": False,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA5", "Selection": roulette_wheel_sel, "Crossover": swap_xo, "Mutation": single_mutation, "Elitism": False,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA6", "Selection": random_sel, "Crossover": swap_xo, "Mutation": random_mutation, "Elitism": False,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA7", "Selection": random_sel, "Crossover": cycle_xo, "Mutation": subgrid_mutation, "Elitism": False,
     "FitnessFunction": 'basic_fitness'},
    {"ID": "GA8", "Selection": random_sel, "Crossover": single_point_xo, "Mutation": valid_mutation, "Elitism": False,
     "FitnessFunction": 'basic_fitness'},
]


results = {}
for config in configs:
    cumulative_runtime = 0
    cumulative_fitness = 0
    best_runtime = float('inf')
    best_fitness = float('inf')
    #print(config["ID"])

    for i in range(num_iterations):
        #print(i)
        population = Population(pop_size, ex_medium)

        time0 = time.time()
        result = population.evolve(fitness_function=config["FitnessFunction"],
                                   gens=gens, xo_prob=0.85, mut_prob=0.25,
                                   select=config["Selection"],
                                   xo=config["Crossover"],
                                   mutate=config["Mutation"],
                                   elitism=config["Elitism"],
                                   elitism_prop=0.05)
        time1 = time.time()

        runtime = (time1 - time0) / 60
        fitness = result["best_fitness"]

        cumulative_runtime += runtime
        cumulative_fitness += fitness
        if runtime < best_runtime:
            best_runtime = runtime
        if fitness < best_fitness:
            best_fitness = fitness

    avg_runtime = cumulative_runtime / num_iterations
    avg_fitness = cumulative_fitness / num_iterations

    results[config["ID"]] = {
        "avg_runtime": avg_runtime,
        "avg_fitness": avg_fitness,
        "best_runtime": best_runtime,
        "best_fitness": best_fitness
    }


results_table = []
for id, result in results.items():
    results_table.append([
        id,
        f"{result['avg_runtime']:.2f}",
        result['avg_fitness'],
        f"{result['best_runtime']:.2f}",
        result['best_fitness']
    ])

print("Results for each GA configuration (without elitism):")
print(tabulate(results_table, headers=headers, tablefmt="grid"))


# Grid Search with the best GA operator (GA8 with Elitism)

pop_size = 12500
gens =  125
elitism = True
num_iterations = 5

# Test different values
xo_prob_values = [0.70, 0.80, 0.85, 0.90]
mut_prob_values = [0.10, 0.15, 0.20, 0.25]
elitism_proportion = [0.01, 0.05, 0.10]
results_table = []

# Create a table to store results
for xo_prob in xo_prob_values:
    #print(xo_prob)
    for mut_prob in mut_prob_values:
        #print(mut_prob)
        for elitism_prop in elitism_proportion:
            fitness_sum = 0
            convergence_gen_sum = 0
            run_time_sum = 0
            for i in range(num_iterations):
                # print(f"I'm at: {xo_prob, mut_prob, elitism_prop, i}")
                population = Population(pop_size, ex_medium)

                result = population.evolve('basic_fitness', gens, xo_prob, mut_prob, single_point_xo, valid_mutation,
                                           random_sel, True, elitism_prop)


                fitness_sum += result["best_fitness"]
                convergence_gen_sum += result["convergence_gen"]
                run_time_sum += result["run_time"]

            avg_fitness = fitness_sum / num_iterations
            avg_convergence_gen = convergence_gen_sum / num_iterations
            avg_run_time = run_time_sum / num_iterations
            results_table.append((xo_prob, mut_prob, elitism_prop, avg_fitness, avg_convergence_gen, avg_run_time))

headers2 = ["xo_prob", "mut_prob", "elitism_prop", "Average Best Fitness", "Average Convergence Gen", "Average Run Time"]
print("Grid Search Results:")
print(tabulate(results_table, headers=headers2, tablefmt="grid"))


# Testing the final algorithm which each sudoku difficulty

pop_size = 50000
gens =  150
elitism = True
num_iterations = 1 # 10
dif = {'ex_easy': ex_easy, 'ex_medium': ex_medium, 'ex_hard': ex_hard , 'ex_extreme': ex_extreme}

configs = [
    {"ID": 'ex_easy', "Difficulty": ex_easy, "Selection": random_sel, "Crossover": single_point_xo, "Mutation": valid_mutation, "Elitism": True, "FitnessFunction": 'basic_fitness'},
    {"ID": 'ex_medium', "Difficulty": ex_medium, "Selection": random_sel, "Crossover": single_point_xo, "Mutation": valid_mutation, "Elitism": True, "FitnessFunction": 'basic_fitness'},
    {"ID": 'ex_hard', "Difficulty": ex_hard, "Selection": random_sel, "Crossover": single_point_xo, "Mutation": valid_mutation, "Elitism": True, "FitnessFunction": 'basic_fitness'},
    {"ID": 'ex_extreme', "Difficulty": ex_extreme, "Selection": random_sel, "Crossover": single_point_xo, "Mutation": valid_mutation, "Elitism": True, "FitnessFunction": 'basic_fitness'},
]

results = {}
for config in configs:
    cumulative_runtime = 0
    cumulative_fitness = 0
    best_runtime = float('inf')
    best_fitness = float('inf')
    solved = 0

    for i in range(num_iterations):
        #print(f"Running iteration {i+1} for configuration {config['ID']}")

        # Create a new population for each configuration
        population = Population(pop_size, config["Difficulty"])

        time0 = time.time()
        result = population.evolve(
            fitness_function=config["FitnessFunction"],
            gens=gens, xo_prob=0.80, mut_prob=0.20,
            select=config["Selection"],
            xo=config["Crossover"],
            mutate=config["Mutation"],
            elitism=config["Elitism"],
            elitism_prop=0.10
        )
        time1 = time.time()

        runtime = (time1 - time0) / 60
        fitness = result["best_fitness"]

        cumulative_runtime += runtime
        cumulative_fitness += fitness
        if fitness == 0:
            solved += 1
        if runtime < best_runtime:
            best_runtime = runtime
        if fitness < best_fitness:
            best_fitness = fitness


    avg_runtime = cumulative_runtime / num_iterations
    avg_fitness = cumulative_fitness / num_iterations
    avg_solved = solved / num_iterations

    results[config["ID"]] = {
        "avg_solved": avg_solved,
        "avg_runtime": avg_runtime,
        "avg_fitness": avg_fitness,
        "best_runtime": best_runtime,
        "best_fitness": best_fitness
    }

results_table = []
for id, result in results.items():
    results_table.append([
        id,
        f"{result['avg_solved'] * 100:.2f}%",
        f"{result['avg_runtime']:.2f}",
        result['avg_fitness'],
        f"{result['best_runtime']:.2f}",
        result['best_fitness'],
    ])

headers3 = ["Difficulty", "Average Solved", "Average Runtime (min)", "Average Fitness", "Best Runtime (min)", "Best Fitness"]
print("Final GA Results:")
print(tabulate(results_table, headers=headers3, tablefmt="grid"))
