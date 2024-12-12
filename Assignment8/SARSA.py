"""
========================================================
EECS 658 - Assignment 8
Author:    Cameron Denton
Date:      December 3rd, 2024
========================================================

Brief Description:
------------------
This program implements the SARSA algorithm in a 5x5 Gridworld environment. The agent learns 
an action-value function Q(s,a) through interaction with the environment. It uses an 
epsilon-greedy policy for action selection and updates Q-values at each step. The learning 
continues until the Q-values converge below a certain threshold.

During execution, the program:
- Prints the Q-table at certain episodes and upon convergence.
- Plots the error value (max difference in Q-values) vs. episodes.
- Plots the cumulative average reward vs. episodes.
- Prints the rewards matrix (R).
- Prints the optimal path from a specified start state to a target terminal state, based on 
  the learned Q-values.

Inputs:
-------
- None (the Gridworld environment is defined in the code).

Outputs:
--------
- Q-tables at initial and certain intervals.
- A plot of the error value vs. episodes.
- A plot of cumulative average reward vs. episodes.
- The rewards matrix (R).
- The optimal path from a given start state to the target state.

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from class lectures.
- Some code and comments suggested by ChatGPT, excluding function summaries.

========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Grid dimensions
grid_height = 5
grid_width = 5

# Define states and terminal states
states = [(i, j) for i in range(grid_height) for j in range(grid_width)]
state_indices = {state: idx for idx, state in enumerate(states)}
num_states = len(states)

terminal_states = [(0, 0), (4, 4)]
terminal_indices = [state_indices[s] for s in terminal_states]

# Actions
actions = ['up', 'down', 'left', 'right']
action_indices = {action: idx for idx, action in enumerate(actions)}
num_actions = len(actions)

# Initialize Q-table and Rewards matrix
Q = np.zeros((num_states, num_actions))
R = np.full((num_states, num_actions), -1)

def step(state, action):
    """
    Execute one action from a given state and return the next state and reward.
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
        reward = 0
    if next_state in terminal_states:
        reward = 100

    return next_state, reward

# Update Rewards matrix
for state in states:
    s_idx = state_indices[state]
    for action in actions:
        a_idx = action_indices[action]
        next_state, reward = step(state, action)
        R[s_idx, a_idx] = reward

# SARSA parameters
alpha = 0.1     # Learning rate
gamma = 0.9     # Discount factor
epsilon = 0.1   # Exploration rate
threshold = 0.001

def get_next_state_index(s_idx, a_idx):
    """
    Given a state index and action index, returns the next state index.
    """
    state = states[s_idx]
    action = actions[a_idx]
    next_state, _ = step(state, action)
    return state_indices[next_state]

def sarsa():
    """
    Runs the SARSA algorithm until convergence and prints/plots required information.
    """
    global Q
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

        # Epsilon-greedy initial action selection
        if np.random.uniform(0, 1) < epsilon:
            a_idx = np.random.randint(0, num_actions)
        else:
            a_idx = np.argmax(Q[s_idx, :])

        episode_reward = 0

        # Run one episode
        while s_idx not in terminal_indices:
            next_s_idx = get_next_state_index(s_idx, a_idx)
            reward = R[s_idx, a_idx]
            episode_reward += reward

            # Epsilon-greedy next action selection
            if np.random.uniform(0, 1) < epsilon:
                next_a_idx = np.random.randint(0, num_actions)
            else:
                next_a_idx = np.argmax(Q[next_s_idx, :])

            # SARSA update
            td_target = reward + gamma * Q[next_s_idx, next_a_idx]
            td_error = td_target - Q[s_idx, a_idx]
            Q[s_idx, a_idx] += alpha * td_error

            s_idx = next_s_idx
            a_idx = next_a_idx

        total_reward += episode_reward
        cumulative_rewards.append(total_reward / (episode + 1))

        # Check convergence
        error_value = np.max(np.abs(Q - Q_prev))
        error_list.append(error_value)
        Q_prev = Q.copy()

        # Print Q-table at specified episodes
        if episode in [0, 1, 9] or error_value <= threshold:
            print(f"\nEpisode {episode + 1}:")
            print("Q-Table:")
            print(Q)

        episode += 1

    # Plot error and cumulative average rewards
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(error_list) + 1), error_list)
    plt.xlabel('Episode')
    plt.ylabel('Max |Q_t(s,a) - Q_{t-1}(s,a)|')
    plt.title('Error Value vs Episode')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_rewards) + 1), cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Average Reward')
    plt.title('Cumulative Average Reward')

    plt.tight_layout()
    plt.show()

    print("\nRewards Matrix (R):")
    print(R)

    # Determine optimal path from (2, 1) to (0, 0)
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
                print("Path length exceeded limit, stopping.")
                break

        print(f"\nOptimal path from {start_state} to {target_state}:")
        print(optimal_path)
    else:
        print(f"Start state {start_state} or target state {target_state} is invalid.")

# Run SARSA
sarsa()
