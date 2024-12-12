"""
========================================================
EECS 658 - Assignment 8
Author:    Cameron Denton
Date:      December 3rd, 2024
========================================================

Brief Description:
------------------
This program implements the Monte Carlo On-Policy Every-Visit algorithm in a 5x5 Gridworld.
The agent follows an epsilon-greedy policy to select actions, collects returns for each 
state-action pair, and updates both the value function V(s) and the policy π(s).

The algorithm runs until convergence of the value function, determined by a threshold 
on the maximum change in V(s). During execution, the program:

- Prints N(s), S(s), V(s), and π(s) at select epochs and upon convergence.
- Prints episode details including states, actions, rewards, discount factor, and returns G(s).
- Plots the error value (maximum difference between V(s) in consecutive iterations) vs. episodes.

Inputs:
-------
- None (the Gridworld environment and parameters are defined in the code).

Outputs:
--------
- N(s), S(s), V(s), and π(s) at initial and selected epochs.
- Episode details (state, action, reward, returns).
- Plot of error vs. episodes after convergence.

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

# Gridworld Parameters
grid_height = 5
grid_width = 5
states = [(i, j) for i in range(grid_height) for j in range(grid_width)]
terminal_states = [(0, 0), (4, 4)]
actions = ['up', 'down', 'left', 'right']

# Initialize value function and statistics
V = {(i, j): 0.0 for i in range(grid_height) for j in range(grid_width)}
N = {(i, j): 0 for i in range(grid_height) for j in range(grid_width)}
S = {(i, j): 0.0 for i in range(grid_height) for j in range(grid_width)}

# Initialize policy arbitrarily (excluding terminal states)
policy_table = {
    (i, j): np.random.choice(actions)
    for i in range(grid_height) for j in range(grid_width)
    if (i, j) not in terminal_states
}

# Epsilon for epsilon-greedy policy
epsilon = 0.1

def convert_to_grid(dictionary, grid_height=5, grid_width=5):
    """
    Converts a dictionary into a 2D grid representation.
    """
    grid = np.zeros((grid_height, grid_width), dtype=object)
    for (i, j), value in dictionary.items():
        grid[i, j] = value
    return grid

def print_epoch_data(epoch, N, S, V, policy_table, grid_height=5, grid_width=5):
    """
    Prints N(s), S(s), V(s), and π(s) in grid format for a given epoch.
    """
    print(f"\nEpoch {epoch}:")
    print("N(s):")
    print(convert_to_grid(N, grid_height, grid_width))
    print("\nS(s):")
    print(convert_to_grid(S, grid_height, grid_width))
    print("\nV(s):")
    print(convert_to_grid(V, grid_height, grid_width))
    print("\nPolicy (π):")
    policy_grid = np.full((grid_height, grid_width), ' ')
    for i in range(grid_height):
        for j in range(grid_width):
            if (i, j) in terminal_states:
                policy_grid[i, j] = 'T'
            else:
                policy_grid[i, j] = policy_table[(i, j)]
    print(policy_grid)

def reward_function(state, action, next_state):
    """
    Reward structure:
    - 0 for reaching a terminal state.
    - -1 otherwise.
    """
    if next_state in terminal_states:
        return 0
    elif state == next_state:
        return -1
    else:
        return -1

def step(state, action):
    """
    Executes an action from a given state and returns next_state and reward.
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

def monte_carlo_on_policy(threshold=0.001, gamma=0.9, max_steps=100):
    """
    Monte Carlo On-Policy Every Visit:
    - Follows epsilon-greedy policy.
    - Updates V(s) and π(s) from returns and improves policy.
    - Runs until max change in V(s) < threshold.
    """
    global policy_table
    V_prev = None
    error_list = []
    error_value = float('inf')
    episode = 0

    while error_value > threshold:
        # Random start state (non-terminal)
        state = (np.random.randint(grid_height), np.random.randint(grid_width))
        while state in terminal_states:
            state = (np.random.randint(grid_height), np.random.randint(grid_width))

        episode_states = []
        episode_actions = []
        episode_rewards = []
        step_count = 0

        # Generate episode with epsilon-greedy w.r.t. π(s)
        while True:
            # Epsilon-greedy selection
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(actions)
            else:
                action = policy_table[state]

            next_state, reward = step(state, action)
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            step_count += 1
            if next_state in terminal_states or step_count >= max_steps:
                episode_states.append(next_state)
                if step_count >= max_steps:
                    print(f"Episode {episode + 1}: Max steps reached. Ending episode.")
                break
            state = next_state

        # Update V(s), N(s), S(s), and π(s)
        G = 0
        visited_state_action_pairs = set()
        for t in reversed(range(len(episode_states) - 1)):
            state_t = episode_states[t]
            action_t = episode_actions[t]
            reward_t = episode_rewards[t]
            G = gamma * G + reward_t
            state_action_pair = (state_t, action_t)
            if state_action_pair not in visited_state_action_pairs:
                visited_state_action_pairs.add(state_action_pair)
                N[state_t] += 1
                S[state_t] += G
                V[state_t] = S[state_t] / N[state_t]

                # Policy improvement
                q_values = []
                for a in actions:
                    ns, r = step(state_t, a)
                    q = r + gamma * V.get(ns, 0)
                    q_values.append(q)
                best_action = actions[np.argmax(q_values)]
                policy_table[state_t] = best_action

        # Check convergence
        if V_prev is not None:
            error_value = max(abs(V[s] - V_prev[s]) for s in V)
            error_list.append(error_value)
        else:
            error_value = float('inf')
            error_list.append(error_value)
        V_prev = V.copy()

        # Print at certain epochs
        if episode in [0, 1, 9] or error_value <= threshold:
            print_epoch_data(episode + 1, N, S, V, policy_table)
            print("k\tState\tAction\tr\tγ\tG(s)")
            G_print = 0
            for k in reversed(range(len(episode_states) - 1)):
                state_k = episode_states[k]
                action_k = episode_actions[k]
                reward_k = episode_rewards[k]
                G_print = gamma * G_print + reward_k
                print(f"{k}\t{state_k}\t{action_k}\t{reward_k}\t{gamma}\t{G_print}")

        episode += 1

    # Plot error vs episodes
    plt.plot(range(episode), error_list)
    plt.xlabel('Episode')
    plt.ylabel('Max |V_k(s) - V_{k-1}(s)|')
    plt.title('Error Value vs Episode')
    plt.show()

# Run the Monte Carlo On-Policy Every Visit algorithm
monte_carlo_on_policy(threshold=0.001, gamma=0.9, max_steps=100)
