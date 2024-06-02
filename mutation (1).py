from random import randint, sample, choice
import numpy as np

def single_mutation(individual, fixed_mask):
    # Create a list of indices (i, j) that are not fixed (mutable)
    mutable_indices = [(i, j) for i in range(individual.shape[0]) for j in range(individual.shape[1]) if not fixed_mask[i, j]]
    # Check if there are any mutable indices
    if mutable_indices:
        # Randomly select one of the mutable indices
        index = choice(mutable_indices)
        # Assign a random value between 1 and the size of the grid to the selected index
        individual[index] = randint(1, individual.shape[0])
    
    # Return the mutated individual
    return individual

def random_mutation(individual, fixed_mask):
    # Create a list of indices (i, j) that are not fixed (mutable)
    mutation_indices = [(i, j) for i in range(individual.shape[0]) for j in range(individual.shape[1]) if not fixed_mask[i, j]]
    
    # Iterate over each mutable index
    for indx in mutation_indices:
        # Assign a random value between 1 and the size of the grid to the current index
        individual[indx] = randint(1, individual.shape[0])  # Use dynamic grid size
    
    # Return the mutated individual
    return individual

def subgrid_mutation(individual, fixed_mask):
    size = individual.shape[0]  # Get the size of the Sudoku grid
    subgrid_size = int(np.sqrt(size))  # Calculate the size of each subgrid

    # Iterate over each subgrid in the Sudoku grid
    for i in range(0, size, subgrid_size):
        for j in range(0, size, subgrid_size):
            subgrid = individual[i:i + subgrid_size, j:j + subgrid_size]  # Extract the current subgrid
            valid_values = list(set(range(1, size + 1)) - set(subgrid.flatten()) - set(
                fixed_mask[i:i + subgrid_size, j:j + subgrid_size].flatten()))  # Determine valid values for the subgrid

            # Calculate empty indices within the subgrid based on fixed_mask
            empty_indices = [(x, y) for x in range(subgrid_size) for y in range(subgrid_size) if
                             fixed_mask[i + x, j + y] == 0]

            # If there are valid values and empty indices, randomly assign one to a random index within the subgrid
            if valid_values and empty_indices:
                random_value = choice(valid_values)  # Select a random valid value
                random_index = choice(empty_indices)  # Select a random empty index within the subgrid
                individual[i + random_index[0], j + random_index[1]] = random_value  # Assign the random value to the selected index

    return individual  # Return the mutated individual

def valid_mutation(individual, fixed_mask):
    size = individual.shape[0]  # Get the size of the Sudoku grid
    subgrid_size = int(np.sqrt(size))  # Calculate the size of each subgrid

    # Create a list of mutable indices (cells that are not fixed)
    mutable_indices = [(i, j) for i in range(size) for j in range(size) if not fixed_mask[i, j]]
    
    if mutable_indices:  # If there are mutable cells
        for i, j in mutable_indices:
            # Calculate the set of possible values for the current cell
            possible_values = list(set(range(1, size + 1)) -
                                   set(individual[i, :]) -  # Values already in the row
                                   set(individual[:, j]) -  # Values already in the column
                                   set(individual[
                                       i // subgrid_size * subgrid_size:(i // subgrid_size + 1) * subgrid_size,
                                       j // subgrid_size * subgrid_size:(j // subgrid_size + 1) * subgrid_size
                                       ].flatten()))  # Values already in the subgrid

            if possible_values:  # If there are valid values available
                individual[i, j] = choice(possible_values)  # Assign a random valid value to the cell

    return individual  # Return the mutated individual


def trial_mutation(individual, fixed_mask):
    size = individual.shape[0]  # Get the size of the Sudoku grid
    subgrid_size = int(np.sqrt(size))  # Calculate the size of each subgrid
    available_numbers = list(range(1, size + 1))  # List of possible numbers for the grid

    # Create a list of mutable indices (cells that are not fixed)
    mutable_indices = [(i, j) for i in range(size) for j in range(size) if not fixed_mask[i, j]]

    if mutable_indices:  # If there are mutable cells
        for i, j in mutable_indices:
            # Find fixed numbers in the row, column, and subgrid
            row_values = set(individual[i, k] for k in range(size) if fixed_mask[i, k])
            col_values = set(individual[k, j] for k in range(size) if fixed_mask[k, j])
            subgrid_values = set(
                individual[m, n]
                for m in range((i // subgrid_size) * subgrid_size, (i // subgrid_size + 1) * subgrid_size)
                for n in range((j // subgrid_size) * subgrid_size, (j // subgrid_size + 1) * subgrid_size)
                if fixed_mask[m, n]
            )

            # Combine fixed values from row, column, and subgrid
            combined_values = row_values | col_values | subgrid_values

            # Exclude these combined values from the available numbers to get possible values
            possible_values = list(set(available_numbers) - combined_values)
            
            # Assign a random possible value to the cell
            individual[i, j] = choice(possible_values)

    return individual  # Return the mutated individual