import os
import pickle
import random

from environment import TaxiEnvironment
from q_learning import QLearningAgent


# ==============================================
# SETTINGS
# ==============================================

TEST_EPISODES = 100
MAX_STEPS = 500

MODEL_PATH = os.path.join(
    "models",
    "q_table.pkl"
)


# ==============================================
# RANDOM AGENT TEST
# ==============================================

def test_random_agent():

    env = TaxiEnvironment()

    total_rewards = 0
    total_steps = 0
    successes = 0

    for _ in range(TEST_EPISODES):

        state = env.reset()

        episode_reward = 0
        episode_steps = 0

        for _ in range(MAX_STEPS):

            action = random.randint(0, 3)

            next_state, reward, done = env.step(
                action
            )

            state = next_state

            episode_reward += reward
            episode_steps += 1

            if done:
                successes += 1
                break

        total_rewards += episode_reward
        total_steps += episode_steps

    return (
        successes / TEST_EPISODES * 100,
        total_rewards / TEST_EPISODES,
        total_steps / TEST_EPISODES
    )


# ==============================================
# Q-LEARNING AGENT TEST
# ==============================================

def test_q_learning_agent():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    env = TaxiEnvironment()

    agent = QLearningAgent(
        actions=4,
        epsilon=0.0
    )

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        agent.q_table = pickle.load(file)

    total_rewards = 0
    total_steps = 0
    successes = 0

    for _ in range(TEST_EPISODES):

        state = env.reset()

        episode_reward = 0
        episode_steps = 0

        for _ in range(MAX_STEPS):

            action = agent.choose_action(
                state
            )

            next_state, reward, done = env.step(
                action
            )

            state = next_state

            episode_reward += reward
            episode_steps += 1

            if done:
                successes += 1
                break

        total_rewards += episode_reward
        total_steps += episode_steps

    return (
        successes / TEST_EPISODES * 100,
        total_rewards / TEST_EPISODES,
        total_steps / TEST_EPISODES
    )


# ==============================================
# RUN COMPARISON
# ==============================================

print()
print("==============================================")
print("RANDOM vs Q-LEARNING")
print("==============================================")
print()

random_success, random_reward, random_steps = (
    test_random_agent()
)

q_success, q_reward, q_steps = (
    test_q_learning_agent()
)


# ==============================================
# RESULTS
# ==============================================

print("Random Agent")
print("----------------------------------------------")
print(
    f"Success Rate : {random_success:.2f}%"
)
print(
    f"Average Reward: {random_reward:.2f}"
)
print(
    f"Average Steps : {random_steps:.2f}"
)

print()

print("Q-Learning Agent")
print("----------------------------------------------")
print(
    f"Success Rate : {q_success:.2f}%"
)
print(
    f"Average Reward: {q_reward:.2f}"
)
print(
    f"Average Steps : {q_steps:.2f}"
)

print()

print("==============================================")
print("COMPARISON COMPLETE")
print("==============================================")