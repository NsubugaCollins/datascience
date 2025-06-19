import numpy as np
import random

# Define constants
positions = 2  # positions: 0 and 1 (simplified for sequence)
sequence_length = 3  # length of required action sequence
actions = 2  # 0: Left, 1: Right

# Q-table size: states = positions * sequence steps
Q = np.zeros((positions * sequence_length, actions))

# Parameters
episodes = 500
learning_rate = 0.8
gamma = 0.9
epsilon = 0.2

# The correct action sequence to cross the road: Right (1), Left (0), Right (1)
correct_sequence = [1, 0, 1]

def get_state(pos, step):
    """Encode state as single int from position and step in sequence"""
    return step * positions + pos

def next_position(pos, action):
    """Calculate next position based on current pos and action"""
    if action == 0:  # Left
        return max(0, pos - 1)
    else:  # Right
        return min(positions - 1, pos + 1)

for episode in range(episodes):
    pos = 0  # start position
    step = 0  # start of action sequence
    total_reward = 0

    done = False
    while not done:
        state = get_state(pos, step)

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, actions - 1)
        else:
            action = np.argmax(Q[state])

        # Check if action matches required action in sequence
        if action == correct_sequence[step]:
            reward = 10  # correct action in sequence
            step += 1
            if step == sequence_length:  # sequence complete
                done = True
        else:
            reward = -10  # wrong action, penalize

        # Update position
        pos = next_position(pos, action)
        total_reward += reward

        # Next state after action
        next_state = get_state(pos, step if step < sequence_length else sequence_length - 1)

        # Q-learning update
        Q[state, action] = Q[state, action] + learning_rate * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}, Total reward: {total_reward}")

# Test the trained agent
print("\nTesting trained agent's action sequence:")
pos = 0
step = 0
actions_taken = []

while step < sequence_length:
    state = get_state(pos, step)
    action = np.argmax(Q[state])
    actions_taken.append('Left' if action == 0 else 'Right')
    pos = next_position(pos, action)

    if action == correct_sequence[step]:
        step += 1
    else:
        print("Agent failed to follow correct sequence.")
        break
else:
    print(f"Agent successfully followed the sequence: {actions_taken}")
