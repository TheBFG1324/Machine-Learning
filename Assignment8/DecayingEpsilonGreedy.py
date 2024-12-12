"""
========================================================
EECS 658 - Assignment 8
Author:    Cameron Denton
Date:      December 3rd, 2024
========================================================

Brief Description:
------------------
This program implements the Decaying Epsilon-Greedy reinforcement learning algorithm 
in a 5x5 Gridworld environment. The agent starts with a high exploration rate (epsilon)
and decreases it over time to transition from exploration to exploitation. It updates 
the Q-values using a Q-learning update rule.

The program tracks the convergence of Q-values by measuring the maximum change in the Q-table 
between consecutive episodes. It also records cumulative average rewards over episodes.

Finally, it displays:
- The Q-table at certain episodes and upon convergence.
- A plot of the error (max difference in Q-values) versus episodes.
- A plot of the cumulative average reward over episodes.
- The Rewards matrix (R).
- The optimal path from a specified start state to a terminal state using the learned Q-values.

Inputs:
-------
- None (the Gridworld environment is defined in the code).

Outputs:
--------
- Prints the Q-table at episodes 1, 2, 10, and upon convergence.
- Plots the error (max Q-value difference) vs. episodes.
- Plots the cumulative average reward vs. episodes.
- Prints the rewards matrix (R).
- Prints the optimal path from a specified start state to a terminal state.

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from lectures.
- Some code and comments provided by ChatGPT, but excluding function summaries.

========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Gridworld dimensions
grid_height = 5
grid_width = 5

# Define states and their indices
states = [(i, j) for i in range(grid_height) for j in range(grid_width)]
state_indices = {state: idx for idx, state in enumerate(states)}
num_states = len(states)

# Define terminal states
terminal_states = [(0, 0), (4, 4)]
terminal_indices = [state_indices[s] for s in terminal_states]

# Actions and their indices
actions = ['up', 'down', 'left', 'right']
action_indices = {action: idx for idx, action in enumerate(actions)}
num_actions = len(actions)

# Initialize Q-table and R matrix
Q = np.zeros((num_states, num_actions))
R = np.full((num_states, num_actions), -1)

def step(state, action):
    """
    Executes an action in the Gridworld from the given state and returns the next state and reward.
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

    # Determine reward
    if state == next_state and state not in terminal_states:
        reward = -1  # Hitting a wall
    else:
        reward = 0

    if next_state in terminal_states:
        reward = 100  # Reaching terminal

    return next_state, reward

# Fill R matrix based on the environment dynamics
for state in states:
    s_idx = state_indices[state]
    for action in actions:
        a_idx = action_indices[action]
        _, reward = step(state, action)
        R[s_idx, a_idx] = reward

# Decaying Epsilon-Greedy parameters
alpha = 0.1     # Learning rate
gamma = 0.9     # Discount factor
epsilon_start = 1.0
epsilon_end = 0.1
episodes = 1000
decay_rate = (epsilon_end / epsilon_start) ** (1 / episodes)
epsilon = epsilon_start
threshold = 0.001

def get_next_state_index(s_idx, a_idx):
    """
    Given a state index and action index, returns the index of the next state.
    """
    state = states[s_idx]
    action = actions[a_idx]
    next_state, _ = step(state, action)
    return state_indices[next_state]

def decaying_epsilon_greedy():
    """
    Runs the Decaying Epsilon-Greedy algorithm until convergence.
    Prints Q-tables at certain episodes, plots error and cumulative rewards, and shows the optimal path.
    """
    global Q, epsilon
    Q_prev = np.zeros_like(Q)
    error_list = []
    cumulative_rewards = []
    total_reward = 0
    error_value = float('inf')
    episode = 0

    while error_value > threshold:
        s_idx = np.random.randint(0, num_states)
        while s_idx in terminal_indices:
            s_idx = np.random.randint(0, num_states)

        episode_reward = 0

        # Run one episode
        while s_idx not in terminal_indices:
            # Epsilon-greedy action
            if np.random.uniform(0, 1) < epsilon:
                a_idx = np.random.randint(0, num_actions)
            else:
                a_idx = np.argmax(Q[s_idx, :])

            next_s_idx = get_next_state_index(s_idx, a_idx)
            reward = R[s_idx, a_idx]
            episode_reward += reward

            # Q-learning update
            td_target = reward + gamma * np.max(Q[next_s_idx, :])
            td_error = td_target - Q[s_idx, a_idx]
            Q[s_idx, a_idx] += alpha * td_error

            s_idx = next_s_idx

        epsilon *= decay_rate
        total_reward += episode_reward
        cumulative_rewards.append(total_reward / (episode + 1))

        # Check convergence error
        error_value = np.max(np.abs(Q - Q_prev))
        error_list.append(error_value)
        Q_prev = Q.copy()

        # Print Q-table at specified episodes and upon convergence
        if episode in [0, 1, 9] or error_value <= threshold:
            print(f"\nEpisode {episode + 1}:")
            print("Q-Table:")
            print(Q)

        episode += 1

    # Plot the error and cumulative average reward
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(error_list) + 1), error_list)
    plt.xlabel('Episode')
    plt.ylabel('Max |Q_t - Q_{t-1}|')
    plt.title('Error vs Episode')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_rewards) + 1), cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Average Reward')
    plt.title('Cumulative Average Reward')

    plt.tight_layout()
    plt.show()

    print("\nRewards Matrix (R):")
    print(R)

    # Determine optimal path from a given start state
    start_state = (2, 1)
    if start_state in states:
        s_idx = state_indices[start_state]
        optimal_path = [start_state]
        while s_idx not in terminal_indices:
            a_idx = np.argmax(Q[s_idx, :])
            action = actions[a_idx]
            next_state, _ = step(states[s_idx], action)
            optimal_path.append(next_state)
            s_idx = state_indices[next_state]
            if len(optimal_path) > 50:  # Safety check
                print("Path length exceeded limit, stopping.")
                break

        print(f"\nOptimal path from {start_state} to a terminal state:")
        print(optimal_path)
    else:
        print(f"Start state {start_state} is invalid.")

# Run the Decaying Epsilon-Greedy algorithm
decaying_epsilon_greedy()
