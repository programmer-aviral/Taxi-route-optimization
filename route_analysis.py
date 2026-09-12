import csv
import os

from environment import TaxiEnvironment
from q_learning import QLearningAgent


# =========================================================
# SETTINGS
# =========================================================

TEST_EPISODES = 100
MAX_STEPS = 200

MODEL_PATH = os.path.join(
    "models",
    "q_table.pkl"
)

RESULTS_PATH = os.path.join(
    "results",
    "route_analysis.csv"
)


# =========================================================
# CHECK MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        f"Train the agent first using: python train.py"
    )


# =========================================================
# CREATE ENVIRONMENT
# =========================================================

env = TaxiEnvironment()


# =========================================================
# CREATE AGENT
# =========================================================

agent = QLearningAgent(
    actions=4,
    learning_rate=0.1,
    discount_factor=0.9,

    # Evaluation should not explore randomly.
    epsilon=0.0,

    epsilon_decay=0.999,
    min_epsilon=0.05
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

agent.load(MODEL_PATH)


# =========================================================
# PRINT HEADER
# =========================================================

print()
print("=" * 55)
print("Q-LEARNING ROUTE EFFICIENCY ANALYSIS")
print("=" * 55)
print()

print(
    f"Loaded Q-table states: {len(agent.q_table)}"
)

print(
    f"Test routes: {TEST_EPISODES}"
)

print()


# =========================================================
# CREATE RESULTS FOLDER
# =========================================================

os.makedirs(
    "results",
    exist_ok=True
)


# =========================================================
# METRICS
# =========================================================

successful_routes = 0

total_q_steps = 0
total_optimal_steps = 0

total_extra_steps = 0

total_invalid_moves = 0

total_reward = 0

efficiencies = []


# =========================================================
# CSV RESULT FILE
# =========================================================

with open(
    RESULTS_PATH,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "route",
        "start_x",
        "start_y",
        "destination_x",
        "destination_y",
        "optimal_steps",
        "q_learning_steps",
        "extra_steps",
        "efficiency_percent",
        "invalid_moves",
        "reward",
        "success"
    ])


    # =====================================================
    # TEST EACH ROUTE
    # =====================================================

    for episode in range(1, TEST_EPISODES + 1):

        # -------------------------------------------------
        # Generate a completely new route
        # -------------------------------------------------

        state = env.reset()


        start_x = state[0]
        start_y = state[1]

        destination_x = state[2]
        destination_y = state[3]


        # -------------------------------------------------
        # Calculate shortest possible road distance
        # -------------------------------------------------

        optimal_steps = env.get_distance(
            env.taxi_grid,
            env.destination_grid
        )


        # This should always be finite because reset()
        # only selects connected road cells.

        if optimal_steps == float("inf"):

            print(
                f"Route {episode}: skipped "
                f"(no connected path)"
            )

            continue


        # -------------------------------------------------
        # Run Q-learning agent
        # -------------------------------------------------

        route_reward = 0

        q_steps = 0

        invalid_moves = 0

        success = False


        for step in range(
            1,
            MAX_STEPS + 1
        ):

            # ---------------------------------------------
            # Choose learned action
            # ---------------------------------------------

            action = agent.choose_action(
                state
            )


            # ---------------------------------------------
            # Execute action
            # ---------------------------------------------

            next_state, reward, done = env.step(
                action
            )


            # ---------------------------------------------
            # Track reward
            # ---------------------------------------------

            route_reward += reward

            q_steps = step


            # ---------------------------------------------
            # Detect invalid movement
            # ---------------------------------------------

            if reward == -5:

                invalid_moves += 1


            # ---------------------------------------------
            # Move to next state
            # ---------------------------------------------

            state = next_state


            # ---------------------------------------------
            # Destination reached
            # ---------------------------------------------

            if done:

                success = True
                break


        # -------------------------------------------------
        # Update global metrics
        # -------------------------------------------------

        total_q_steps += q_steps

        total_optimal_steps += optimal_steps

        total_invalid_moves += invalid_moves

        total_reward += route_reward


        # -------------------------------------------------
        # Success
        # -------------------------------------------------

        if success:

            successful_routes += 1


            # ---------------------------------------------
            # Extra steps
            # ---------------------------------------------

            extra_steps = q_steps - optimal_steps


            # Extra steps can never be negative.
            # Safety protection in case of future
            # environment changes.

            extra_steps = max(
                0,
                extra_steps
            )


            total_extra_steps += extra_steps


            # ---------------------------------------------
            # Efficiency
            # ---------------------------------------------
            #
            # Example:
            #
            # Optimal = 10
            # Q-learning = 12
            #
            # Efficiency =
            # 10 / 12 * 100
            #
            # = 83.33%
            #

            if q_steps > 0:

                efficiency = (
                    optimal_steps
                    / q_steps
                    * 100
                )

            else:

                efficiency = 0


            efficiencies.append(
                efficiency
            )


        else:

            extra_steps = ""

            efficiency = 0


        # -------------------------------------------------
        # Save route result
        # -------------------------------------------------

        writer.writerow([
            episode,
            start_x,
            start_y,
            destination_x,
            destination_y,
            optimal_steps,
            q_steps,
            extra_steps,
            f"{efficiency:.2f}",
            invalid_moves,
            route_reward,
            int(success)
        ])


        # -------------------------------------------------
        # Print every route
        # -------------------------------------------------

        print(
            f"Route {episode:3d} | "
            f"Start ({start_x:2d},{start_y:2d}) | "
            f"Goal ({destination_x:2d},{destination_y:2d}) | "
            f"Optimal: {optimal_steps:3d} | "
            f"Q-learning: {q_steps:3d} | "
            f"Invalid: {invalid_moves:2d} | "
            f"Reward: {route_reward:4d} | "
            f"Success: {success}"
        )


# =========================================================
# FINAL METRICS
# =========================================================

success_rate = (
    successful_routes
    / TEST_EPISODES
    * 100
)


average_q_steps = (
    total_q_steps
    / TEST_EPISODES
)


average_optimal_steps = (
    total_optimal_steps
    / TEST_EPISODES
)


average_extra_steps = (
    total_extra_steps
    / successful_routes
    if successful_routes > 0
    else 0
)


average_invalid_moves = (
    total_invalid_moves
    / TEST_EPISODES
)


average_reward = (
    total_reward
    / TEST_EPISODES
)


average_efficiency = (
    sum(efficiencies)
    / len(efficiencies)
    if efficiencies
    else 0
)


# =========================================================
# PRINT FINAL REPORT
# =========================================================

print()
print("=" * 55)
print("FINAL ROUTE EFFICIENCY REPORT")
print("=" * 55)
print()

print(
    f"Test routes              : {TEST_EPISODES}"
)

print(
    f"Successful routes        : {successful_routes}"
)

print(
    f"Success rate             : {success_rate:.2f}%"
)

print()

print(
    f"Average optimal steps    : "
    f"{average_optimal_steps:.2f}"
)

print(
    f"Average Q-learning steps : "
    f"{average_q_steps:.2f}"
)

print(
    f"Average extra steps      : "
    f"{average_extra_steps:.2f}"
)

print()

print(
    f"Average efficiency       : "
    f"{average_efficiency:.2f}%"
)

print(
    f"Average invalid moves    : "
    f"{average_invalid_moves:.2f}"
)

print(
    f"Average reward           : "
    f"{average_reward:.2f}"
)

print()

print(
    f"Detailed results saved to:"
)

print(
    f"{RESULTS_PATH}"
)

print()

print("=" * 55)
print("ANALYSIS COMPLETE")
print("=" * 55)