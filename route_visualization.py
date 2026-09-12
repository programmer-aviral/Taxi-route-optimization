import csv
import os

import matplotlib.pyplot as plt


# =========================================================
# FILE PATH
# =========================================================

CSV_PATH = os.path.join(
    "results",
    "route_analysis.csv"
)

RESULTS_FOLDER = "results"


# =========================================================
# CHECK FILE
# =========================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Route analysis file not found: {CSV_PATH}\n"
        f"Run this first:\n"
        f"python route_analysis.py"
    )


# =========================================================
# LOAD CSV DATA
# =========================================================

routes = []

optimal_steps = []
q_learning_steps = []
efficiencies = []
invalid_moves = []
rewards = []


with open(
    CSV_PATH,
    "r",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        routes.append(
            int(row["route"])
        )

        optimal_steps.append(
            int(row["optimal_steps"])
        )

        q_learning_steps.append(
            int(row["q_learning_steps"])
        )

        efficiencies.append(
            float(row["efficiency_percent"])
        )

        invalid_moves.append(
            int(row["invalid_moves"])
        )

        rewards.append(
            float(row["reward"])
        )


# =========================================================
# CREATE RESULTS FOLDER
# =========================================================

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)


# =========================================================
# GRAPH 1
# OPTIMAL VS Q-LEARNING STEPS
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    routes,
    optimal_steps,
    label="Optimal Steps"
)

plt.plot(
    routes,
    q_learning_steps,
    label="Q-Learning Steps"
)

plt.xlabel("Route")
plt.ylabel("Steps")

plt.title(
    "Optimal Route vs Q-Learning Route"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "optimal_vs_qlearning.png"
    )
)

plt.show()


# =========================================================
# GRAPH 2
# ROUTE EFFICIENCY
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    routes,
    efficiencies
)

plt.axhline(
    y=100,
    linestyle="--",
    label="Perfect Efficiency"
)

plt.xlabel("Route")
plt.ylabel("Efficiency (%)")

plt.title(
    "Q-Learning Route Efficiency"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "route_efficiency.png"
    )
)

plt.show()


# =========================================================
# GRAPH 3
# INVALID MOVES
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    routes,
    invalid_moves
)

plt.xlabel("Route")
plt.ylabel("Invalid Moves")

plt.title(
    "Invalid Moves per Route"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "invalid_moves.png"
    )
)

plt.show()


# =========================================================
# GRAPH 4
# REWARD PER ROUTE
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    routes,
    rewards
)

plt.xlabel("Route")
plt.ylabel("Reward")

plt.title(
    "Q-Learning Reward per Route"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "route_rewards.png"
    )
)

plt.show()


# =========================================================
# DONE
# =========================================================

print()
print("=" * 55)
print("ROUTE VISUALIZATION COMPLETE")
print("=" * 55)
print()

print(
    "Saved graphs:"
)

print(
    "results/optimal_vs_qlearning.png"
)

print(
    "results/route_efficiency.png"
)

print(
    "results/invalid_moves.png"
)

print(
    "results/route_rewards.png"
)

print()