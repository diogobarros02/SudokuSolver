from random import randint
import numpy as np

def find_valid_digit(grid, row, col):
    """Find a valid digit for the Sudoku cell at (row, col)."""
    size = grid.shape[0]
    digits = set(range(1, size + 1))  # Initialize a set of all possible digits

    # Remove digits already present in the row, column, and block
    digits -= set(grid[row, :])
    digits -= set(grid[:, col])

    # Dynamically calculate the block size based on the grid size
    block_size = int(np.sqrt(size))

    # Calculate the starting row and column for the block
    start_row, start_col = row // block_size * block_size, col // block_size * block_size
    for r in range(start_row, start_row + block_size):
        for c in range(start_col, start_col + block_size):
            digits.discard(grid[r, c])  # Remove digits already present in the block

    return digits  # Return the remaining valid digits

def find_valid_digit_from_mask(mask, size):
    """Find indices where the mask is not set."""
    return np.where(mask.flatten() == 0)[0]  # Return indices of non-fixed cells

################# CROSSOVER FUNCTIONS #################

def single_point_xo(parent1_repr, parent2_repr, fixed_mask):
    size = parent1_repr.shape[0]

    # Flatten the representation for easier manipulation
    parent1_flat = parent1_repr.flatten()
    parent2_flat = parent2_repr.flatten()

    # Get indices where fixed_mask is 0
    unfixed_indices = find_valid_digit_from_mask(fixed_mask, size)

    # Check if there is at least one unfixed position to perform the crossover
    if len(unfixed_indices) < 1:
        raise ValueError("No unfixed positions to perform crossover")

    # Select one random index from the unfixed positions
    idx = np.random.choice(unfixed_indices)

    # Create copies of the parent representations for the offspring
    offspring1_flat = np.copy(parent1_flat)
    offspring2_flat = np.copy(parent2_flat)

    # Swap the values at the chosen index between the two parents
    offspring1_flat[idx], offspring2_flat[idx] = parent2_flat[idx], parent1_flat[idx]

    # Reshape offspring back to 2D representation
    offspring1 = offspring1_flat.reshape(size, size)
    offspring2 = offspring2_flat.reshape(size, size)

    return offspring1, offspring2  # Return the offspring

def swap_xo(parent1_repr, parent2_repr, fixed_mask):
    size = parent1_repr.shape[0]
    point = randint(1, size - 1)  # Random crossover point

    # Create offspring representations by swapping the parts of parents
    o1_repr = np.copy(parent1_repr)
    o2_repr = np.copy(parent2_repr)

    # Perform crossover
    for i in range(size):
        if i >= point:  # Beyond the crossover point
            o1_repr[i, :], o2_repr[i, :] = parent2_repr[i, :], parent1_repr[i, :]
        else:  # Before the crossover point
            o1_repr[i, :], o2_repr[i, :] = parent1_repr[i, :], parent2_repr[i, :]

    return o1_repr, o2_repr  # Return the offspring

def cycle_xo(parent1, parent2, fixed_mask):
    size = parent1.shape[0]
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)

    # Create a mask to keep track of visited positions
    visited = np.zeros((size, size), dtype=bool)

    # Perform cycle crossover
    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                # Start a new cycle
                cycle = [(i, j)]
                while True:
                    # Find the corresponding position in the other parent
                    idx = np.where(parent1 == parent2[cycle[-1]])  # Find the index of the corresponding value
                    if len(idx[0]) == 0 or np.any(visited[idx]):  # If no corresponding position is found or it's already visited
                        break
                    next_pos = (idx[0][0], idx[1][0])
                    if next_pos == cycle[0]:  # If the cycle is completed
                        break
                    cycle.append(next_pos)
                    visited[next_pos] = True

                # Alternate between parents to form offspring
                for k, (r, c) in enumerate(cycle):
                    if k % 2 == 0:
                        offspring1[r, c] = parent1[r, c]
                        offspring2[r, c] = parent2[r, c]
                    else:
                        offspring1[r, c] = parent2[r, c]
                        offspring2[r, c] = parent1[r, c]

    # Fill in empty cells with random values
    unfilled_indices = np.where(offspring1 == 0)
    for r, c in zip(unfilled_indices[0], unfilled_indices[1]):
        offspring1[r, c] = np.random.choice(range(1, size + 1))

    unfilled_indices = np.where(offspring2 == 0)
    for r, c in zip(unfilled_indices[0], unfilled_indices[1]):
        offspring2[r, c] = np.random.choice(range(1, size + 1))

    # Copy fixed positions from parents
    offspring1[fixed_mask] = parent1[fixed_mask]
    offspring2[fixed_mask] = parent2[fixed_mask]

    return offspring1, offspring2  # Return the offspring

def subgrid_xo(parent1, parent2, fixed_mask):
    # Dynamically calculate the subgrid size based on the grid size
    size = parent1.shape[0]
    subgrid_size = int(np.sqrt(size))

    offspring1 = np.copy(parent1)
    offspring2 = np.copy(parent2)

    # Randomly determine the number of subgrids to crossover
    num_subgrids_to_change = np.random.randint(1, subgrid_size**2)

    # Randomly select subgrids to change
    subgrid_indices = np.random.choice(range(size), size=num_subgrids_to_change, replace=False)

    # Perform crossover operation within the selected subgrids
    for index in subgrid_indices:
        # Select the corresponding subgrid from both parents
        row_start, col_start = (index // subgrid_size) * subgrid_size, (index % subgrid_size) * subgrid_size
        subgrid1 = parent1[row_start:row_start + subgrid_size, col_start:col_start + subgrid_size]
        subgrid2 = parent2[row_start:row_start + subgrid_size, col_start:col_start + subgrid_size]

        offspring1[row_start:row_start + subgrid_size, col_start:col_start + subgrid_size] = subgrid2
        offspring2[row_start:row_start + subgrid_size, col_start:col_start + subgrid_size] = subgrid1

    return offspring1, offspring2  # Return the offspring