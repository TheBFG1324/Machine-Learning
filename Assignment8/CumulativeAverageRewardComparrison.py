"""
========================================================
EECS 658 - Assignment 8
Author:    Cameron Denton
Date:      December 3rd, 2024
========================================================

Brief Description:
------------------
This program compares three reinforcement learning algorithms—Q-Learning, SARSA, 
and Q-Learning with a Decaying Epsilon-Greedy policy—in a Gridworld environment. 
For each algorithm, it runs multiple episodes, collects cumulative average rewards, 
and plots them for comparison. The Gridworld is a 5x5 environment with two terminal 
states and zero reward transitions (except hitting walls and reaching terminal states).

Inputs:
-------
- None (the Gridworld environment and parameters are defined in the program).

Outputs:
--------
- A plot of cumulative average rewards over episodes for each of the three algorithms:
  * Q-Learning
  * SARSA
  * Decaying Epsilon-Greedy Q-Learning

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from class lectures.
- Some code and commentary suggested by ChatGPT, excluding function summaries.

========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Gridworld Parameters
# --------------------------------------------------------
episodes = 1000           # Number of episodes for training
grid_height = 5           # Gridworld height
grid_width = 5            # Gridworld width
states = [(i, j) for i in range(grid_height) for j in range(grid_width)]
state_indices = {state: idx for idx, state in enumerate(states)}
num_states = len(states)
actions = ['up', 'down', 'left', 'right']
action_indices = {action: idx for idx, action in enumerate(actions)}
num_actions = len(actions)
terminal_states = [(0, 0), (4, 4)]
terminal_indices = [state_indices[s] for s in terminal_states]

# --------------------------------------------------------
# Environment Dynamics
# --------------------------------------------------------
def step(state, action):
    """
    Execute one step in the environment given a state and action.
    Moves the agent on the grid and returns the next state and reward.
    Terminal states yield no transitions.
    Walls and edges yield a penalty.
    Reaching a terminal state yields a high reward.
    """
    i, j = state
    if state in terminal_states:
        return state, 0  # Terminal state reached

    # Compute next state based on action
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

    # Assign rewards
    if state == next_state and state not in terminal_states:
        reward = -1  # Penalty for hitting a wall
    else:
        reward = 0

    if next_state in terminal_states:
        reward = 100  # Reward for reaching terminal

    return next_state, reward

# Precompute rewards R for each state-action pair
R = np.zeros((num_states, num_actions))
for s in states:
    s_idx = state_indices[s]
    for a in actions:
        a_idx = action_indices[a]
        _, reward = step(s, a)
        R[s_idx, a_idx] = reward

def get_next_state_index(s_idx, a_idx):
    """
    Given state index and action index, return the next state's index.
    """
    state = states[s_idx]
    action = actions[a_idx]
    next_state, _ = step(state, action)
    return state_indices[next_state]

# --------------------------------------------------------
# Q-Learning
# --------------------------------------------------------
def q_learning_rewards():
    """
    Run Q-Learning to obtain cumulative average rewards per episode.
    Uses epsilon-greedy exploration with fixed epsilon.
    """
    Q = np.zeros((num_states, num_actions))
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    cumulative_rewards = []
    total_reward = 0

    for episode in range(episodes):
        s_idx = np.random.randint(num_states)
        while s_idx in terminal_indices:
            s_idx = np.random.randint(num_states)

        episode_reward = 0

        while s_idx not in terminal_indices:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                a_idx = np.random.randint(num_actions)
            else:
                a_idx = np.argmax(Q[s_idx, :])

            next_s_idx = get_next_state_index(s_idx, a_idx)
            reward = R[s_idx, a_idx]
            episode_reward += reward

            # Q-Learning update
            td_target = reward + gamma * np.max(Q[next_s_idx, :])
            td_error = td_target - Q[s_idx, a_idx]
            Q[s_idx, a_idx] += alpha * td_error

            s_idx = next_s_idx

        total_reward += episode_reward
        cumulative_rewards.append(total_reward / (episode + 1))

    return cumulative_rewards

# --------------------------------------------------------
# SARSA
# --------------------------------------------------------
def sarsa_rewards():
    """
    Run SARSA to obtain cumulative average rewards per episode.
    Uses epsilon-greedy exploration with fixed epsilon.
    """
    Q = np.zeros((num_states, num_actions))
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    cumulative_rewards = []
    total_reward = 0

    for episode in range(episodes):
        s_idx = np.random.randint(num_states)
        while s_idx in terminal_indices:
            s_idx = np.random.randint(num_states)

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            a_idx = np.random.randint(num_actions)
        else:
            a_idx = np.argmax(Q[s_idx, :])

        episode_reward = 0

        while s_idx not in terminal_indices:
            next_s_idx = get_next_state_index(s_idx, a_idx)
            reward = R[s_idx, a_idx]
            episode_reward += reward

            # Epsilon-greedy action selection for the next step
            if np.random.rand() < epsilon:
                next_a_idx = np.random.randint(num_actions)
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

    return cumulative_rewards

# --------------------------------------------------------
# Decaying Epsilon Q-Learning
# --------------------------------------------------------
def decaying_epsilon_rewards():
    """
    Run Q-Learning with a decaying epsilon.
    Epsilon decreases from 1.0 to 0.1 over the given number of episodes.
    """
    Q = np.zeros((num_states, num_actions))
    alpha = 0.1
    gamma = 0.9
    epsilon_start = 1.0
    epsilon_end = 0.1
    decay_rate = (epsilon_end / epsilon_start) ** (1 / episodes)
    epsilon = epsilon_start
    cumulative_rewards = []
    total_reward = 0

    for episode in range(episodes):
        s_idx = np.random.randint(num_states)
        while s_idx in terminal_indices:
            s_idx = np.random.randint(num_states)

        episode_reward = 0

        while s_idx not in terminal_indices:
            # Epsilon-greedy with decaying epsilon
            if np.random.rand() < epsilon:
                a_idx = np.random.randint(num_actions)
            else:
                a_idx = np.argmax(Q[s_idx, :])

            next_s_idx = get_next_state_index(s_idx, a_idx)
            reward = R[s_idx, a_idx]
            episode_reward += reward

            # Q-Learning update
            td_target = reward + gamma * np.max(Q[next_s_idx, :])
            td_error = td_target - Q[s_idx, a_idx]
            Q[s_idx, a_idx] += alpha * td_error

            s_idx = next_s_idx

        epsilon *= decay_rate
        total_reward += episode_reward
        cumulative_rewards.append(total_reward / (episode + 1))

    return cumulative_rewards

# --------------------------------------------------------
# Main Execution: Run all algorithms and plot results
# --------------------------------------------------------
rewards_q_learning = q_learning_rewards()
rewards_sarsa = sarsa_rewards()
rewards_decaying_epsilon = decaying_epsilon_rewards()

plt.figure(figsize=(10, 6))
plt.plot(range(1, episodes + 1), rewards_q_learning, label='Q-Learning')
plt.plot(range(1, episodes + 1), rewards_sarsa, label='SARSA')
plt.plot(range(1, episodes + 1), rewards_decaying_epsilon, label='Decaying Epsilon-Greedy')
plt.xlabel('Episode')
plt.ylabel('Cumulative Average Reward')
plt.title('Cumulative Average Reward Comparison')
plt.legend()
plt.grid(True)
plt.show()
