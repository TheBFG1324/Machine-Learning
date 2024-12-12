"""
========================================================
EECS 658 - Assignment 8
Author:    Cameron Denton
Date:      December 3rd, 2024
========================================================

Brief Description:
------------------
This program implements the Monte Carlo First-Visit algorithm to estimate the 
value function V(s) for each state in a 5x5 Gridworld. The agent follows an 
equiprobable random policy. The algorithm runs episodes until the value function 
converges, determined by a specified threshold on the maximum change in V(s) 
across states.

During the run, the program:
- Prints the counts N(s), sums S(s), and value estimates V(s) at certain epochs.
- Prints the details of each episode (states, rewards, discount factor, returns).
- Plots the error value (max difference between consecutive V(s) estimates) 
  versus the number of episodes.

Inputs:
-------
- None (the Gridworld environment and parameters are defined within the code).

Outputs:
--------
- Prints N(s), S(s), V(s) at the beginning, at selected epochs, and upon convergence.
- Prints the episode steps (state, reward, discount factor, and computed return G(s)).
- Plots the error vs. episodes after convergence.

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from lectures.
- Some code and comments suggested by ChatGPT, excluding function summaries.

========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Gridworld parameters
grid_height = 5
grid_width = 5
states = [(i, j) for i in range(grid_height) for j in range(grid_width)]
terminal_states = [(0, 0), (4, 4)]
actions = ['up', 'down', 'left', 'right']

# Initialize Value Function and statistics
V = {(i, j): 0.0 for i in range(grid_height) for j in range(grid_width)}
N = {(i, j): 0 for i in range(grid_height) for j in range(grid_width)}  # Counts of first visits
S = {(i, j): 0.0 for i in range(grid_height) for j in range(grid_width)} # Sums of returns for states

def convert_to_grid(dictionary, grid_height=5, grid_width=5):
    """
    Converts a dictionary with state-value pairs into a 2D NumPy array for printing.
    """
    grid = np.zeros((grid_height, grid_width))
    for (i, j), value in dictionary.items():
        grid[i, j] = value
    return grid

def print_epoch_data(epoch, N, S, V, grid_height=5, grid_width=5):
    """
    Prints the counts N(s), sums S(s), and value estimates V(s) in a grid format.
    Called at certain epochs for debugging/inspection.
    """
    print(f"\nEpoch {epoch}:")
    print("N(s):")
    print(convert_to_grid(N, grid_height, grid_width))
    print("\nS(s):")
    print(convert_to_grid(S, grid_height, grid_width))
    print("\nV(s):")
    print(convert_to_grid(V, grid_height, grid_width))

def policy(state):
    """
    Equiprobable random policy: uniformly select an action from the set of actions.
    """
    return np.random.choice(actions)

def reward_function(state, action, next_state):
    """
    Defines the reward structure:
    - 0 for reaching a terminal state.
    - -1 for any other move (including if the move doesn't change the state due to walls).
    """
    if next_state in terminal_states:
        return 0
    else:
        return -1

def step(state, action):
    """
    Executes an action from a given state and returns the next state and reward.
    """
    i, j = state
    if state in terminal_states:
        return state, 0  # Terminal states have no further transitions

    # Compute next state
    if action == 'up':
        i_new = max(i - 1, 0)
        next_state = (i_new, j)
    elif action == 'down':
        i_new = min(i + 1, grid_height - 1)
        next_state = (i_new, j)
    elif action == 'left':
        j_new = max(j - 1, 0)
        next_state = (i, j_new)
    elif action == 'right':
        j_new = min(j + 1, grid_width - 1)
        next_state = (i, j_new)
    else:
        next_state = state

    # Determine reward
    if state == next_state:
        # No movement, hitting a wall
        reward = -1
    else:
        reward = reward_function(state, action, next_state)
    return next_state, reward

def monte_carlo_first_visit(threshold=0.001, gamma=0.9):
    """
    Implements the Monte Carlo First Visit algorithm:
    - Generate episodes following equiprobable random policy.
    - Update V(s) using the first-visit returns for each state.
    - Continue until max change in V(s) < threshold.
    """
    V_prev = None
    error_list = []
    error_value = float('inf')
    episode_count = 0

    while error_value > threshold:
        # Initialize a random non-terminal start state
        state = (np.random.randint(grid_height), np.random.randint(grid_width))
        while state in terminal_states:
            state = (np.random.randint(grid_height), np.random.randint(grid_width))

        episode_states = []
        episode_rewards = []

        # Generate one episode
        while True:
            action = policy(state)
            next_state, reward = step(state, action)
            episode_states.append(state)
            episode_rewards.append(reward)
            if next_state in terminal_states:
                episode_states.append(next_state)
                break
            state = next_state

        # Compute returns (G) and update V(s) using first-visit approach
        G = 0
        visited_states = set()
        for t in reversed(range(len(episode_states) - 1)):
            state_t = episode_states[t]
            reward_t = episode_rewards[t]
            G = gamma * G + reward_t
            if state_t not in visited_states:
                visited_states.add(state_t)
                N[state_t] += 1
                S[state_t] += G
                V[state_t] = S[state_t] / N[state_t]

        # Check convergence error
        if V_prev is not None:
            error_value = max(abs(V[s] - V_prev[s]) for s in V)
            error_list.append(error_value)
        else:
            error_value = float('inf')
            error_list.append(error_value)

        V_prev = V.copy()

        # Print intermediate results
        if episode_count in [0, 1, 9] or error_value <= threshold:
            print_epoch_data(episode_count + 1, N, S, V)
            print("k\tState\tr\tγ\tG(s)")
            G_print = 0
            for k in reversed(range(len(episode_states) - 1)):
                state_k = episode_states[k]
                reward_k = episode_rewards[k]
                G_print = gamma * G_print + reward_k
                print(f"{k}\t{state_k}\t{reward_k}\t{gamma}\t{G_print}")

        episode_count += 1

    # Plot error value vs episodes
    plt.plot(range(episode_count), error_list)
    plt.xlabel('Episode')
    plt.ylabel('Max |V_k(s) - V_{k-1}(s)|')
    plt.title('Error Value vs Episode')
    plt.show()

# Run the Monte Carlo First Visit algorithm
monte_carlo_first_visit(threshold=0.001, gamma=0.9)
