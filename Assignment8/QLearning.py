"""
========================================================
EECS 658 - Assignment 8
Author:    Cameron Denton
Date:      December 3rd, 2024
========================================================

Brief Description:
------------------
This program implements the Q-Learning algorithm in a 5x5 Gridworld environment. 
The goal is for the agent to learn the optimal policy to reach a terminal state 
from a given start state. 

The algorithm updates Q-values for state-action pairs as the agent interacts with 
the environment. It runs until the Q-values converge to within a specified threshold. 
During the run, the program:
- Prints the Q-table at certain episodes and upon convergence.
- Plots the error value (max difference in Q-values) vs. episodes.
- Prints the rewards matrix (R).
- Prints the optimal path from a specified start state to a terminal state using 
  the learned Q-values.

Inputs:
-------
- None (the Gridworld environment and parameters are defined within the program).

Outputs:
--------
- Q-tables at initial and certain intervals.
- A plot of the error value vs. episodes.
- The rewards matrix (R).
- The optimal path from a given start state to the terminal state.

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
state_indices = {(i, j): idx for idx, (i, j) in enumerate(states)}
terminal_states = [(0, 0), (4, 4)]
terminal_indices = [state_indices[s] for s in terminal_states]

actions = ['up', 'down', 'left', 'right']
action_indices = {a: idx for idx, a in enumerate(actions)}

num_states = len(states)
num_actions = len(actions)

# Initialize Q-table and Rewards matrix
Q = np.zeros((num_states, num_actions))
R = np.full((num_states, num_actions), -1)  # Initialize with -1

def step(state, action):
    """
    Execute one action from a given state and return the next state and reward.
    """
    i, j = state
    if state in terminal_states:
        return state, 0
    
    if action == 'up':
        i_new = max(i - 1, 0)
        s_prime = (i_new, j)
    elif action == 'down':
        i_new = min(i + 1, grid_height - 1)
        s_prime = (i_new, j)
    elif action == 'left':
        j_new = max(j - 1, 0)
        s_prime = (i, j_new)
    elif action == 'right':
        j_new = min(j + 1, grid_width - 1)
        s_prime = (i, j_new)
    else:
        s_prime = state

    if s_prime == state and state not in terminal_states:
        r = -1  # Penalty for hitting a wall
    else:
        r = 0   # Default reward

    return s_prime, r

# Update the rewards for reaching terminal states
for s in states:
    s_idx = state_indices[s]
    for a in actions:
        a_idx = action_indices[a]
        next_state, _ = step(s, a)
        if next_state in terminal_states:
            R[s_idx, a_idx] = 100
        else:
            R[s_idx, a_idx] = 0

# Q-Learning parameters
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.1    # Exploration rate
threshold = 0.001

def get_next_state_index(s_idx, a_idx):
    """
    Given a state index and action index, return the next state index.
    """
    s = states[s_idx]
    a = actions[a_idx]
    next_state, _ = step(s, a)
    return state_indices[next_state]

def q_learning():
    """
    Runs Q-Learning until convergence.
    Prints Q-tables at certain episodes and upon convergence.
    Plots error vs. episodes and prints the rewards matrix and optimal path.
    """
    global Q
    Q_prev = Q.copy()
    error_list = []
    error_value = float('inf')
    episode = 0

    while error_value > threshold:
        # Random initial non-terminal state
        s_idx = np.random.randint(0, num_states)
        while s_idx in terminal_indices:
            s_idx = np.random.randint(0, num_states)

        # Run one episode until terminal
        while s_idx not in terminal_indices:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                a_idx = np.random.randint(0, num_actions)
            else:
                a_idx = np.argmax(Q[s_idx, :])

            next_s_idx = get_next_state_index(s_idx, a_idx)
            reward = R[s_idx, a_idx]

            td_target = reward + gamma * np.max(Q[next_s_idx, :])
            td_error = td_target - Q[s_idx, a_idx]
            Q[s_idx, a_idx] += alpha * td_error

            s_idx = next_s_idx

        # Check for convergence
        error_value = np.max(np.abs(Q - Q_prev))
        error_list.append(error_value)
        Q_prev = Q.copy()

        # Print Q-tables at certain iterations
        if episode in [0, 1, 9] or error_value <= threshold:
            print(f"\nEpisode {episode + 1}:")
            print("Q-Table:")
            print(Q)

        episode += 1

    # Plot error vs episodes
    plt.plot(range(episode), error_list)
    plt.xlabel('Episode')
    plt.ylabel('Max |Q_t(s,a) - Q_{t-1}(s,a)|')
    plt.title('Error Value vs Episode')
    plt.show()

    print("\nRewards Matrix (R):")
    print(R)

    # Determine optimal path from a given start state to terminal
    start_state = (2, 1)
    target_state = (0, 0)
    if start_state in states and target_state in terminal_states:
        s_idx = state_indices[start_state]
        optimal_path = [start_state]
        while s_idx not in terminal_indices:
            a_idx = np.argmax(Q[s_idx, :])
            action = actions[a_idx]
            next_state, _ = step(states[s_idx], action)
            optimal_path.append(next_state)
            s_idx = state_indices[next_state]
            if len(optimal_path) > 50:
                print("Path length exceeded, stopping.")
                break

        print(f"\nOptimal path from {start_state} to {target_state}:")
        print(optimal_path)
    else:
        print(f"Invalid start or target state: {start_state}, {target_state}")

# Execute Q-Learning
q_learning()
