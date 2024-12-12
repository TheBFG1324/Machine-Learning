"""
========================================================
EECS 658 - Assignment 7
Author:    Cameron Denton
Date:      November 25th, 2024
========================================================

Brief Description:
------------------
This program implements the Value Iteration Algorithm on a 5x5 grid with two terminal 
states (0,0) and (4,4). The algorithm updates the values of each non-terminal state 
until the values converge to within a threshold (0.001).

At selected iterations, it prints the grid and the derived policy. After convergence, 
it plots the mean absolute error (Δ) of the value updates versus the iteration count.

Inputs:
-------
- None

Outputs:
--------
- Prints grids and policies at iterations 1, 2, and 10, and upon convergence.
- A plot of the error (mean difference in values) vs. iteration count.

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from David Johnson's lectures.
- Some code and comments suggested by ChatGPT, excluding function summaries.

========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 5
TERMINAL_STATES = [(0, 0), (4, 4)]
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_EFFECTS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}
DISCOUNT_FACTOR = 1.0
REWARD = -1

# Initialize the grid to zeros
GRID = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def is_valid_state(state):
    """
    Checks if a given state (row, col) is within the grid boundaries.
    """
    return 0 <= state[0] < GRID_SIZE and 0 <= state[1] < GRID_SIZE


def is_terminal_state(state):
    """
    Checks if a state is terminal.
    """
    return state in TERMINAL_STATES


def get_blank_grid():
    """
    Returns a blank 5x5 grid filled with zeros.
    """
    return [[0] * GRID_SIZE for _ in range(GRID_SIZE)]


def get_next_state(state, action):
    """
    Given a state and an action, returns the resulting next state.
    If the action would go out of bounds, returns the original state.
    """
    effect = ACTION_EFFECTS[action]
    next_state = (state[0] + effect[0], state[1] + effect[1])
    if not is_valid_state(next_state):
        return state
    return next_state


def calculate_grid_difference(grid1, grid2):
    """
    Calculates the mean absolute difference between two grids.
    """
    return np.abs(np.array(grid1) - np.array(grid2)).mean()


def calculate_new_value(position, grid):
    """
    For Value Iteration:
    Calculates the new value of a given position by considering all possible actions,
    taking the max over their resulting values (plus the reward).
    """
    values = []
    for action in ACTIONS:
        next_state = get_next_state(position, action)
        x, y = next_state
        values.append(grid[x][y])
    return REWARD + DISCOUNT_FACTOR * max(values)


def get_policy(grid):
    """
    Derives the policy from the current grid values.
    For each state, chooses the action that leads to the highest valued next state.
    Terminal states are marked 'T'.
    """
    policy = [['' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            position = (i, j)
            if is_terminal_state(position):
                policy[i][j] = 'T'
                continue

            best_action = None
            best_value = float('-inf')
            for action in ACTIONS:
                next_state = get_next_state(position, action)
                x, y = next_state
                if grid[x][y] > best_value:
                    best_value = grid[x][y]
                    best_action = action
            policy[i][j] = best_action
    return policy


def value_iteration():
    """
    Performs Value Iteration until convergence (threshold = 0.001).
    Prints grid and policy at specific iterations and upon convergence.
    Returns final grid and list of mean errors per iteration.
    """
    iteration = 0
    min_change = 1e-4
    cur_change = float('inf')

    print("Initial Grid:")
    print(np.array(GRID))

    changes = []

    while cur_change > min_change:
        new_grid = get_blank_grid()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                position = (i, j)
                if not is_terminal_state(position):
                    new_grid[i][j] = calculate_new_value(position, GRID)

        change = calculate_grid_difference(new_grid, GRID)
        changes.append(change)
        cur_change = change
        GRID[:] = new_grid[:]
        iteration += 1

        if iteration in [1, 2, 10] or cur_change <= min_change:
            print(f"Iteration {iteration}:")
            print(np.array(GRID))
            policy = get_policy(GRID)
            print("Policy:")
            for row in policy:
                print(row)

    print(f"Converged at iteration {iteration}")
    print("Final Grid:")
    print(np.array(GRID))

    return GRID, changes


def plot_error(changes):
    """
    Plots the mean error (Δ) over iterations to show convergence.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(changes, label="Error (Δ)")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Error (Δ)")
    plt.title("Value Iteration Convergence")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    final_grid, changes = value_iteration()
    plot_error(changes)


if __name__ == "__main__":
    main()
