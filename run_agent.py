import os
import pickle
import time

import pygame

from environment import TaxiEnvironment
from q_learning import QLearningAgent


# ==================================================
# SETTINGS
# ==================================================

MODEL_PATH = os.path.join(
    "models",
    "q_table.pkl"
)

WIDTH = 1000
HEIGHT = 700

FPS = 5


# ==================================================
# CHECK MODEL
# ==================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )


# ==================================================
# LOAD ENVIRONMENT
# ==================================================

env = TaxiEnvironment(
    WIDTH,
    HEIGHT
)


# ==================================================
# LOAD AGENT
# ==================================================

agent = QLearningAgent(
    actions=4,
    epsilon=0.0
)


with open(
    MODEL_PATH,
    "rb"
) as file:

    agent.q_table = pickle.load(file)


# ==================================================
# START PYGAME
# ==================================================

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Trained Taxi Agent"
)

clock = pygame.time.Clock()


# ==================================================
# START EPISODE
# ==================================================

state = env.reset()

running = True
done = False

step_count = 0
total_reward = 0


# ==================================================
# MAIN LOOP
# ==================================================

while running:

    # ----------------------------------------------
    # PYGAME EVENTS
    # ----------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False


    # ----------------------------------------------
    # AGENT MAKES DECISION
    # ----------------------------------------------

    if not done:

        action = agent.choose_action(state)

        next_state, reward, done = env.step(
            action
        )

        state = next_state

        total_reward += reward

        step_count += 1

        print(
            f"Step: {step_count} | "
            f"Action: {action} | "
            f"Reward: {reward} | "
            f"State: {state}"
        )


    # ----------------------------------------------
    # DRAW ENVIRONMENT
    # ----------------------------------------------

    env.render(screen)

    pygame.display.flip()

    clock.tick(FPS)


    # ----------------------------------------------
    # DESTINATION REACHED
    # ----------------------------------------------

    if done:

        print()
        print("================================")
        print("DESTINATION REACHED!")
        print("================================")
        print(
            "Total steps:",
            step_count
        )
        print(
            "Total reward:",
            total_reward
        )

        time.sleep(2)

        running = False


# ==================================================
# CLOSE PYGAME
# ==================================================

pygame.quit()