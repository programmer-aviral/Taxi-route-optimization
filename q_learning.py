import os
import pickle
import random


class QLearningAgent:
    """
    Q-Learning agent.

    The agent stores Q-values for each state and action.

    State:
        A tuple representing the current environment state.

    Actions:
        0 = UP
        1 = DOWN
        2 = LEFT
        3 = RIGHT
    """

    def __init__(
        self,
        actions=4,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.999,
        min_epsilon=0.05
    ):
        # Number of possible actions
        self.actions = actions

        # Learning rate (alpha)
        self.learning_rate = learning_rate

        # Discount factor (gamma)
        self.discount_factor = discount_factor

        # Exploration probability
        self.epsilon = epsilon

        # How quickly exploration decreases
        self.epsilon_decay = epsilon_decay

        # Minimum exploration
        self.min_epsilon = min_epsilon

        # Q-table
        #
        # Example:
        # {
        #     (10, 9, 9, 10, -1, 1): [0.2, -1.5, 0.8, 4.2]
        # }
        #
        # Four numbers correspond to:
        # UP, DOWN, LEFT, RIGHT
        self.q_table = {}


    # =========================================================
    # GET Q-VALUES
    # =========================================================

    def get_q_values(self, state):
        """
        Return the Q-values for a state.

        If the state has never been seen before,
        create four Q-values initialized to 0.
        """

        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.actions

        return self.q_table[state]


    # =========================================================
    # CHOOSE ACTION
    # =========================================================

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy strategy.

        With probability epsilon:
            Explore by choosing a random action.

        Otherwise:
            Exploit by choosing the action with the highest Q-value.
        """

        q_values = self.get_q_values(state)

        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.actions)

        # Exploitation
        max_q = max(q_values)

        # Find all actions having the maximum Q-value.
        best_actions = [
            action
            for action, value in enumerate(q_values)
            if value == max_q
        ]

        # Randomly choose between tied best actions
        return random.choice(best_actions)


    # =========================================================
    # UPDATE Q-VALUE
    # =========================================================

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-value using the Q-learning formula.

        Q(s,a) = Q(s,a) +
                 alpha * (
                     reward + gamma * max(Q(s',a'))
                     - Q(s,a)
                 )
        """

        current_q_values = self.get_q_values(state)
        current_q = current_q_values[action]

        # If episode finished, there is no future reward.
        if done:
            target = reward

        else:
            next_q_values = self.get_q_values(next_state)
            best_next_q = max(next_q_values)

            target = (
                reward
                + self.discount_factor * best_next_q
            )

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            target - current_q
        )

        current_q_values[action] = new_q


    # =========================================================
    # EPSILON DECAY
    # =========================================================

    def decay_epsilon(self):
        """
        Gradually reduce exploration.
        """

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )


    # =========================================================
    # SAVE Q-TABLE
    # =========================================================

    def save(self, filepath):
        """
        Save the complete agent state to a pickle file.

        This saves:
            - Q-table
            - epsilon
            - learning parameters
            - number of actions

        A temporary file is used first so a partially-written
        model is less likely to replace the existing model.
        """

        # Make sure the parent folder exists
        parent_folder = os.path.dirname(filepath)

        if parent_folder:
            os.makedirs(parent_folder, exist_ok=True)

        temporary_path = filepath + ".tmp"

        data = {
            "q_table": self.q_table,
            "actions": self.actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
        }

        try:
            with open(temporary_path, "wb") as file:
                pickle.dump(
                    data,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL
                )

            # Replace old model with completed model
            os.replace(temporary_path, filepath)

        except Exception:
            # Remove temporary file if saving failed
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

            raise

        print(
            f"Model saved successfully: {filepath}"
        )
        print(
            f"Q-table states saved: {len(self.q_table)}"
        )


    # =========================================================
    # LOAD Q-TABLE
    # =========================================================

    def load(self, filepath):
        """
        Load a previously saved agent.
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model file not found: {filepath}"
            )

        with open(filepath, "rb") as file:
            data = pickle.load(file)

        self.q_table = data["q_table"]

        # Restore parameters when available
        self.actions = data.get(
            "actions",
            self.actions
        )

        self.learning_rate = data.get(
            "learning_rate",
            self.learning_rate
        )

        self.discount_factor = data.get(
            "discount_factor",
            self.discount_factor
        )

        self.epsilon = data.get(
            "epsilon",
            self.epsilon
        )

        self.epsilon_decay = data.get(
            "epsilon_decay",
            self.epsilon_decay
        )

        self.min_epsilon = data.get(
            "min_epsilon",
            self.min_epsilon
        )

        print(
            f"Model loaded successfully: {filepath}"
        )
        print(
            f"Q-table states loaded: {len(self.q_table)}"
        )