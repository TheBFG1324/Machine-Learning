"""
========================================================
EECS 658 - Assignment 8
Author:    Cameron Denton
Date:      December 3rd, 2024
========================================================

Brief Description:
------------------
This program implements the Monte Carlo Every-Visit algorithm to estimate the value 
function V(s) in a 5x5 Gridworld. The agent follows an equiprobable random policy, 
collecting returns for each state and using these returns to update V(s).

The algorithm runs episodes until convergence, determined by a threshold for the 
maximum change in V(s) between iterations. At certain epochs, it prints:
- The counts N(s), sums S(s), and value function V(s) in a grid format.
- Episode details, including states, rewards, discount factor, and computed returns.

After convergence, it plots the error value (max difference in V(s)) vs. episodes.

Inputs:
-------
- None (Gridworld environment is defined within the program).

Outputs:
--------
- Prints N(s), S(s), and V(s) at the beginning and certain epochs.
- Prints episode details (states, rewards, returns).
- Plots the error value over episodes.

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from lectures.
- Some code and comments suggested by ChatGPT, but not function summaries.

========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Gridworld dimensions
grid_height = 5
grid_width = 5

# Define states and terminal states
states = [(i, j) for i in range(grid_height) for j in range(grid_width)]
terminal_states = [(0, 0), (4, 4)]

# Actions
actions = ['up', 'down', 'left', 'right']

# Initialize V(s), returns, counts (N(s)), sums (S(s))
V = {(i, j): 0.0 for i in range(grid_height) for j in range(grid_width)}
returns = {(i, j): [] for i in range(grid_height) for j in range(grid_width)}
N = {(i, j): 0 for i in range(grid_height) for j in range(grid_width)}
S = {(i, j): 0.0 for i in range(grid_height) for j in range(grid_width)}

def policy(state):
    """
    Equiprobable random policy: returns a random action from the available actions.
    """
    return np.random.choice(actions)

def convert_to_grid(dictionary, grid_height=5, grid_width=5):
    """
    Converts a state-value dictionary into a 2D NumPy array for representation.
    """
    grid = np.zeros((grid_height, grid_width))
    for (i, j), value in dictionary.items():
        grid[i, j] = value
    return grid

def print_epoch_data(epoch, N, S, V, grid_height=5, grid_width=5):
    """
    Prints N(s), S(s), and V(s) in grid format for a given epoch.
    """
    print(f"\nEpoch {epoch}:")
    print("N(s):")
    print(convert_to_grid(N, grid_height, grid_width))
    print("\nS(s):")
    print(convert_to_grid(S, grid_height, grid_width))
    print("\nV(s):")
    print(convert_to_grid(V, grid_height, grid_width))

def reward_function(state, action, next_state):
    """
    Defines the reward function for the environment.
    """
    if next_state in terminal_states:
        return 0
    elif state == next_state:
        return -1
    else:
        return -1

def step(state, action):
    """
    State transition based on the given state and action.
    """
    i, j = state
    if state in terminal_states:
        return state, 0

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

    if state == next_state and state not in terminal_states:
        reward = -1
    else:
        reward = reward_function(state, action, next_state)
    return next_state, reward

def monte_carlo_every_visit(threshold=0.001, gamma=0.9):
    """
    Runs the Monte Carlo Every-Visit algorithm until convergence.
    Prints epoch data and plots the error value vs episodes.
    """
    V_list = []
    error_list = []
    error_value = float('inf')
    episode = 0
    V_prev = None

    while error_value > threshold:
        # Random initial state (non-terminal)
        state = (np.random.randint(grid_height), np.random.randint(grid_width))
        while state in terminal_states:
            state = (np.random.randint(grid_height), np.random.randint(grid_width))

        episode_states = []
        episode_rewards = []

        # Generate an episode
        while True:
            action = policy(state)
            next_state, reward = step(state, action)
            episode_states.append(state)
            episode_rewards.append(reward)
            if next_state in terminal_states:
                episode_states.append(next_state)
                break
            state = next_state

        # Calculate returns and update V(s)
        G = 0
        for t in reversed(range(len(episode_states) - 1)):
            state_t = episode_states[t]
            reward_t = episode_rewards[t]
            G = gamma * G + reward_t
            N[state_t] += 1
            S[state_t] += G
            V[state_t] = S[state_t] / N[state_t]

        if V_prev is not None:
            error_value = max([abs(V[s] - V_prev[s]) for s in V])
            error_list.append(error_value)
        else:
            error_value = float('inf')
            error_list.append(error_value)
        V_prev = V.copy()
        V_list.append(V.copy())

        # Print at specific epochs
        if episode in [0, 1, 9] or error_value <= threshold:
            print_epoch_data(episode + 1, N, S, V)
            print("k\tState\tr\tγ\tG(s)")
            G_print = 0
            for k in reversed(range(len(episode_states) - 1)):
                state_k = episode_states[k]
                reward_k = episode_rewards[k]
                G_print = gamma * G_print + reward_k
                print(f"{k}\t{state_k}\t{reward_k}\t{gamma}\t{G_print}")

        episode += 1

    # Plot error vs episodes
    plt.plot(range(episode), error_list)
    plt.xlabel('Episode')
    plt.ylabel('Max |V_k(s) - V_{k-1}(s)|')
    plt.title('Error Value vs Episode')
    plt.show()

# Run the Monte Carlo Every-Visit algorithm
monte_carlo_every_visit(threshold=0.001, gamma=0.9)
