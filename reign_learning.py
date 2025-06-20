import numpy as np
import random
import time

# Define environment parameters
position = 5  # Positions from 0 to 4 (goal is at position 4)
actions = 2   # 0 = Move Left, 1 = Move Right

# Initialize Q-table with slight preference for moving right
Q = np.zeros((position, actions))
Q[:, 1] = 0.01  # Slight bias toward action 1 (right)

# Q-learning parameters
episodes = 100        # Increase episodes for better learning
learning_rate = 0.8   # How quickly the agent learns
gamma = 0.9           # Importance of future rewards
epsilon = 0.3         # Chance of exploring instead of exploiting

# Training phase
for episode in range(episodes):
    # Start from a random position (not always 0)
    state = random.randint(0, position - 2)

    # Epsilon-greedy action selection
    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, actions - 1)
    else:
        action = np.argmax(Q[state])

    # Move based on action
    if action == 0:  # Move Left
        next_state = max(0, state - 1)
    else:            # Move Right
        next_state = min(position - 1, state + 1)

    # Define rewards
    reward = 10 if next_state == position - 1 else -1

    # Q-learning update rule
    Q[state, action] += learning_rate * (
        reward + gamma * np.max(Q[next_state]) - Q[state, action]
    )

    # Move to next state
    state = next_state

# Final Q-table after training
print("\n🏁 Final Q-table after training:")
print(Q)

# Test the trained agent
print("\nTesting the trained agent's path to goal:")
state = 0
steps = 0
max_steps = 20

while state < position - 1 and steps < max_steps:
    # Allow some exploration during testing to escape corner
    if random.uniform(0, 1) < 0.3 and steps < 5:
        action = random.randint(0, actions - 1)
    else:
        action = np.argmax(Q[state])

    if action == 0:
        next_state = max(0, state - 1)
    else:
        next_state = min(position - 1, state + 1)

    print(f"Step {steps}: Position {state} → Action {action} → Next Position {next_state}")
    state = next_state
    steps += 1
    time.sleep(0.2)  # Just for better visual pacing

# Outcome
if state == position - 1:
    print(f"Reached goal in {steps} steps.")
else:
    print("Agent failed to reach goal within max steps.")
