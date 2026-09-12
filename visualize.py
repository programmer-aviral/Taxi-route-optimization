import csv

import matplotlib.pyplot as plt


# ==============================================
# LOAD CSV
# ==============================================

episodes = []
rewards = []
steps = []
success = []
epsilon = []


with open(
    "results/training_metrics.csv",
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        episodes.append(
            int(row["episode"])
        )

        rewards.append(
            float(row["reward"])
        )

        steps.append(
            int(row["steps"])
        )

        success.append(
            int(row["success"])
        )

        epsilon.append(
            float(row["epsilon"])
        )


# ==============================================
# REWARD GRAPH
# ==============================================

plt.figure(figsize=(10, 5))

plt.plot(
    episodes,
    rewards,
    alpha=0.3,
    label="Episode reward"
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Taxi RL Reward During Training")

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/reward_curve.png"
)

plt.show()


# ==============================================
# STEPS GRAPH
# ==============================================

plt.figure(figsize=(10, 5))

plt.plot(
    episodes,
    steps,
    alpha=0.5
)

plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Taxi Steps per Episode")

plt.tight_layout()

plt.savefig(
    "results/steps_curve.png"
)

plt.show()


# ==============================================
# SUCCESS GRAPH
# ==============================================

window = 50

success_rate = []

for i in range(len(success)):

    start = max(
        0,
        i - window + 1
    )

    current_window = success[start:i + 1]

    rate = (
        sum(current_window)
        / len(current_window)
        * 100
    )

    success_rate.append(rate)


plt.figure(figsize=(10, 5))

plt.plot(
    episodes,
    success_rate
)

plt.xlabel("Episode")
plt.ylabel("Success Rate (%)")
plt.title("Taxi Success Rate During Training")

plt.tight_layout()

plt.savefig(
    "results/success_curve.png"
)

plt.show()


# ==============================================
# EPSILON GRAPH
# ==============================================

plt.figure(figsize=(10, 5))

plt.plot(
    episodes,
    epsilon
)

plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Exploration Rate During Training")

plt.tight_layout()

plt.savefig(
    "results/epsilon_curve.png"
)

plt.show()


print("Graphs saved in results/")