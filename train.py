import csv
import os

from environment import TaxiEnvironment
from q_learning import QLearningAgent


# ==========================================
# TRAINING SETTINGS
# ==========================================

EPISODES = 5000
MAX_STEPS = 200

MODEL_PATH = "models/q_table.pkl"
METRICS_PATH = "results/training_metrics.csv"


# ==========================================
# CREATE REQUIRED FOLDERS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================
# CREATE ENVIRONMENT
# ==========================================

env = TaxiEnvironment()


# ==========================================
# CREATE NEW Q-LEARNING AGENT
# ==========================================
#
# We create a NEW agent so training starts
# from an empty Q-table.
#
# This is important because our new environment
# uses a different state representation.
#

agent = QLearningAgent(
    actions=4,
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=1.0,
    epsilon_decay=0.999,
    min_epsilon=0.05
)


# ==========================================
# OPEN CSV FILE FOR TRAINING RESULTS
# ==========================================

with open(METRICS_PATH, "w", newline="") as file:

    writer = csv.writer(file)

    # CSV header
    writer.writerow([
        "episode",
        "total_reward",
        "steps",
        "success",
        "epsilon",
        "q_states"
    ])


    # ==========================================
    # MAIN TRAINING LOOP
    # ==========================================

    for episode in range(1, EPISODES + 1):

        # Reset environment
        state = env.reset()

        total_reward = 0
        success = False
        steps_taken = 0


        # ==========================================
        # ONE EPISODE
        # ==========================================

        for step in range(1, MAX_STEPS + 1):

            # --------------------------------------
            # 1. Choose action
            # --------------------------------------
            action = agent.choose_action(state)


            # --------------------------------------
            # 2. Perform action
            # --------------------------------------
            next_state, reward, done = env.step(action)


            # --------------------------------------
            # 3. Learn from the result
            # --------------------------------------
            agent.update(
                state,
                action,
                reward,
                next_state,
                done
            )


            # --------------------------------------
            # 4. Move to next state
            # --------------------------------------
            state = next_state


            # --------------------------------------
            # 5. Track reward
            # --------------------------------------
            total_reward += reward
            steps_taken = step


            # --------------------------------------
            # 6. Check whether destination reached
            # --------------------------------------
            if done:

                success = True
                break


        # ==========================================
        # REDUCE EXPLORATION
        # ==========================================

        agent.decay_epsilon()


        # ==========================================
        # SAVE EPISODE RESULTS
        # ==========================================

        writer.writerow([
            episode,
            total_reward,
            steps_taken,
            int(success),
            agent.epsilon,
            len(agent.q_table)
        ])


        # ==========================================
        # PRINT PROGRESS
        # ==========================================

        if episode % 500 == 0:

            print(
                f"Episode {episode} | "
                f"Reward: {total_reward:.2f} | "
                f"Steps: {steps_taken} | "
                f"Success: {success} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Q-states: {len(agent.q_table)}"
            )


# ==========================================
# SAVE Q-TABLE
# ==========================================
#
# IMPORTANT:
# Your QLearningAgent must contain a save()
# method for this line to work.
#

agent.save(MODEL_PATH)


# ==========================================
# TRAINING FINISHED
# ==========================================

print()
print("==========================================")
print("Training completed!")
print("==========================================")
print(f"Q-table saved to: {MODEL_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")