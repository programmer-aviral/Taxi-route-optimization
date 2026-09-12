import os

from environment import TaxiEnvironment
from q_learning import QLearningAgent


# ==================================================
# SETTINGS
# ==================================================

TEST_EPISODES = 100
MAX_STEPS = 200

MODEL_PATH = os.path.join(
    "models",
    "q_table.pkl"
)


# ==================================================
# CHECK MODEL
# ==================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# ==================================================
# CREATE ENVIRONMENT
# ==================================================

env = TaxiEnvironment()


# ==================================================
# CREATE AGENT
# ==================================================

agent = QLearningAgent(
    actions=4,
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=0.0,
    epsilon_decay=0.999,
    min_epsilon=0.05
)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

agent.load(MODEL_PATH)


print()
print("==============================================")
print("FRESH-ROUTE MODEL EVALUATION")
print("==============================================")
print()

print(
    "Loaded Q-table states:",
    len(agent.q_table)
)

print()


# ==================================================
# METRICS
# ==================================================

successful_episodes = 0
total_reward = 0
total_steps = 0


# ==================================================
# TEST
# ==================================================

for episode in range(TEST_EPISODES):

    # Create a completely new random route
    state = env.reset()

    episode_reward = 0
    episode_steps = 0

    for step in range(MAX_STEPS):

        # Choose the best learned action
        action = agent.choose_action(state)

        # Perform action
        next_state, reward, done = env.step(action)

        # Update current state
        state = next_state

        # Track metrics
        episode_reward += reward
        episode_steps += 1

        # Destination reached
        if done:

            successful_episodes += 1
            break


    total_reward += episode_reward
    total_steps += episode_steps


# ==================================================
# CALCULATE METRICS
# ==================================================

success_rate = (
    successful_episodes
    / TEST_EPISODES
    * 100
)

average_reward = (
    total_reward
    / TEST_EPISODES
)

average_steps = (
    total_steps
    / TEST_EPISODES
)


# ==================================================
# RESULTS
# ==================================================

print(
    "Test episodes:",
    TEST_EPISODES
)

print(
    "Successful episodes:",
    successful_episodes
)

print(
    f"Success rate: {success_rate:.2f}%"
)

print(
    f"Average reward: {average_reward:.2f}"
)

print(
    f"Average steps: {average_steps:.2f}"
)

print()
print("==============================================")
print("EVALUATION COMPLETE")
print("==============================================")