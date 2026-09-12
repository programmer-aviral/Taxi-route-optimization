# Taxi Route Optimization Using Reinforcement Learning

A 2D city simulation in which a Q-learning agent learns to navigate a road network and reach randomly generated destinations, while handling moving traffic, invalid road boundaries, and shaped rewards. Built entirely from scratch in Python and Pygame — no external RL library was used.

| Category          | Details                                          |
|-------------------|--------------------------------------------------|
| Project           | Taxi Route Optimization                          |
| Domain            | Reinforcement Learning                           |
| Algorithm         | Q-Learning (tabular, implemented from scratch)   |
| Environment       | 2D grid city (20 × 14 grid, 50-pixel cells)      |
| Framework         | Python + Pygame + Matplotlib                     |
| Shortest-path     | BFS (used for evaluation only, not agent policy) |
| Model persistence | Pickle (`models/q_table.pkl`)                    |
| Evaluation        | Route efficiency, success rate, extra steps      |

---

## Project Summary

This project trains a taxi agent to navigate a 2D city road network from a random start location to a random destination using **Q-Learning**. The agent receives shaped rewards for making progress, penalties for moving farther away or attempting invalid moves, and additional traffic penalties when other vehicles are nearby. After 5,000 training episodes, the trained Q-table is saved and reloaded for evaluation and Pygame visualization.

The project is educational and research-oriented. It is not a production navigation system. Its purpose is to demonstrate how an agent can discover route strategies through interaction with an environment rather than through explicit programming.

---

## Quick Start

```bash
git clone https://github.com/programmer-aviral/Taxi-route-optimization.git
cd Taxi-route-optimization

python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt

python train.py          # Train agent (5000 episodes)
python evaluate.py       # Evaluate on 100 fresh routes
python route_analysis.py # Route-vs-BFS efficiency report
python visualize.py      # Generate training charts
python main.py           # Manual Pygame demo (keyboard control)
python run_agent.py      # Watch the trained agent navigate live
```

---

## Visual Story

```
Problem
  ↓
RL Formulation
  ↓
Environment Design
  ↓
State Representation
  ↓
Action Space
  ↓
Reward Design
  ↓
Q-Learning Training
  ↓
Model Saved (q_table.pkl)
  ↓
Evaluation on Fresh Routes
  ↓
BFS Route Comparison
  ↓
Metrics + Charts
  ↓
Pygame Visualization
```

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Objectives](#2-project-objectives)
3. [System Architecture](#3-system-architecture)
4. [Reinforcement Learning Formulation](#4-reinforcement-learning-formulation)
5. [Environment Design](#5-environment-design)
6. [State Representation](#6-state-representation)
7. [Action Space](#7-action-space)
8. [Reward Design](#8-reward-design)
9. [Q-Learning Algorithm](#9-q-learning-algorithm)
10. [Exploration vs Exploitation](#10-exploration-vs-exploitation)
11. [Traffic System](#11-traffic-system)
12. [Passenger System](#12-passenger-system)
13. [Training Pipeline](#13-training-pipeline)
14. [Training Metrics CSV](#14-training-metrics-csv)
15. [Route Analysis CSV](#15-route-analysis-csv)
16. [Training Charts](#16-training-charts)
17. [Route Efficiency Analysis](#17-route-efficiency-analysis)
18. [Evaluation Methodology](#18-evaluation-methodology)
19. [Model Persistence](#19-model-persistence)
20. [Project File Structure](#20-project-file-structure)
21. [Installation](#21-installation)
22. [How to Run](#22-how-to-run)
23. [Demo Controls](#23-demo-controls)
24. [Challenges and Engineering Decisions](#24-challenges-and-engineering-decisions)
25. [Key Learnings](#25-key-learnings)
26. [Limitations](#26-limitations)
27. [Future Improvements](#27-future-improvements)
28. [Interview Explanation](#28-interview-explanation)
29. [Interview Q&A](#29-interview-qa)
30. [Technical Glossary](#30-technical-glossary)
31. [Results Disclaimer](#31-results-disclaimer)
32. [Author](#32-author)

---

## 1. Problem Statement

**Can an agent learn to navigate a road network to a randomly generated destination by receiving rewards and penalties — without being given an explicit route?**

In traditional routing, a programmer writes an algorithm (such as Dijkstra's or A*) that computes the shortest path and instructs a vehicle to follow it. That approach requires hard-coding the cost model and the environment structure.

Reinforcement Learning takes a different approach: the agent discovers routing behaviour on its own by trying actions and observing the consequences. After enough interaction, the agent learns which moves are valuable in which situations — not because it was told the route, but because the reward signal guided it there.

This is a meaningful RL problem because:

- The state space is large enough to require generalization over many grid positions.
- The destination changes on every episode, so the agent cannot memorise a single path.
- Traffic introduces non-stationarity, making the problem more realistic.
- Success must be measured not just by arrival, but by route efficiency.

---

## 2. Project Objectives

All objectives below are implemented in the repository:

- Build a navigable 2D city environment with roads, buildings, and collision detection.
- Represent taxi navigation as a Markov Decision Process (MDP).
- Implement Q-learning from scratch, without using any external RL library.
- Train the agent on 5,000 episodes with random start and destination positions.
- Introduce 10 moving traffic vehicles that dynamically affect the RL state.
- Measure route efficiency by comparing Q-learning steps against BFS optimal steps.
- Log per-episode and per-route metrics to structured CSV files.
- Generate training charts (reward, steps, success rate, epsilon decay).
- Save and reload trained Q-tables using Python's pickle module.
- Provide a live Pygame visualization of both manual and agent-controlled driving.

---

## 3. System Architecture

```
┌──────────────────────────────────────┐
│         Pygame Simulation Layer      │
│  (main.py / run_agent.py)            │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│          TaxiEnvironment             │
│  (environment.py)                    │
│                                      │
│  ┌────────────┐  ┌────────────────┐  │
│  │   World    │  │ TrafficManager │  │
│  │ (world.py) │  │ (traffic.py)   │  │
│  └────────────┘  └────────────────┘  │
│  ┌────────────┐  ┌────────────────┐  │
│  │   Taxi     │  │ BFS Distance   │  │
│  │ (taxi.py)  │  │ Maps           │  │
│  └────────────┘  └────────────────┘  │
└───────────────┬──────────────────────┘
                │
         State (tuple)
                │
                ▼
┌──────────────────────────────────────┐
│          QLearningAgent              │
│  (q_learning.py)                     │
│                                      │
│  ┌─────────────────────────────────┐ │
│  │  Q-Table (dict)                 │ │
│  │  {state: [Q_UP, Q_DOWN,         │ │
│  │           Q_LEFT, Q_RIGHT]}     │ │
│  └─────────────────────────────────┘ │
│                                      │
│  epsilon-greedy action selection     │
│  Bellman update                      │
│  epsilon decay                       │
└───────────────┬──────────────────────┘
                │
         Action (0/1/2/3)
                │
                ▼
┌──────────────────────────────────────┐
│  Environment Step → Reward → State'  │
└──────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  Logging: training_metrics.csv       │
│           route_analysis.csv         │
│  Model:   models/q_table.pkl         │
│  Charts:  results/*.png              │
└──────────────────────────────────────┘
```

**Passenger system** (`passenger.py`) is implemented as a visual simulation layer accessible in `main.py`. It is not part of the RL training objective in the current version.

---

## 4. Reinforcement Learning Formulation

| Component          | Implementation in this project                      |
|--------------------|-----------------------------------------------------|
| Agent              | The taxi                                            |
| Environment        | 2D city grid with roads, buildings, and traffic     |
| State              | (taxi_x, taxi_y, destination_x, destination_y, traffic_level) |
| Actions            | UP / DOWN / LEFT / RIGHT                            |
| Reward             | Shaped: progress bonus + distance penalty + traffic penalty |
| Episode            | Starts at a random road cell, ends at destination   |
| Policy             | Epsilon-greedy over the Q-table                     |
| Learning algorithm | Q-Learning (tabular, off-policy)                    |
| Model              | Q-Table stored as a Python dictionary               |

**Agent:** The taxi is the entity that makes decisions. At each step it reads its current state, consults its Q-table, and picks an action.

**Environment:** The city is a 20 × 14 grid of 50-pixel cells. Some cells are road cells; others contain buildings. The environment enforces road boundaries and handles collisions.

**Episode:** Each episode resets the taxi and destination to randomly selected connected road cells. An episode ends when the taxi reaches the destination or the step limit (200) is exceeded.

**Policy:** During training the agent uses epsilon-greedy: it sometimes explores randomly and sometimes exploits its Q-values. During evaluation epsilon is set to 0.0, so the agent always picks its best known action.

---

## 5. Environment Design

### Grid Dimensions

```
Grid width:   20 cells
Grid height:  14 cells
Cell size:    50 pixels per cell
Window width: 20 × 50 = 1000 pixels
Window height:14 × 50 = 700 pixels
```

### Road Layout

The environment defines road areas using pixel ranges. A cell is a valid road cell if its centre pixel falls inside a horizontal or vertical road band.

```python
HORIZONTAL_ROADS = [
    (200, 300),   # y-pixel band 200–299
    (450, 550),   # y-pixel band 450–549
]

VERTICAL_ROADS = [
    (250, 350),   # x-pixel band 250–349
    (550, 650),   # x-pixel band 550–649
    (850, 950),   # x-pixel band 850–949
]
```

A cell is considered driveable only if it is both a road cell **and** does not collide with a building (`world.is_valid_position`). The list of all valid cells is precomputed at startup.

### BFS Distance Maps

At startup, the environment runs Breadth-First Search (BFS) from every valid road cell to compute the shortest road-following distance to every other reachable cell. These distance maps are stored in `self.distance_maps` and are used **only for evaluation purposes** — to measure how many steps a perfect navigation would require.

> **Important:** BFS does not control or influence the agent's actions at any point. The RL agent decides where to move using its Q-table.

### Conceptual City Layout

```
  0    5   10   15   19
  ·    ·    ·    ·    ·
0 ·····················
  ··   |    |    |   ··
  ··   |    |    |   ··
4 ····─┼────┼────┼─···· ← horizontal road band
  ··   |    |    |   ··
  ··   |    |    |   ··
8 ····─┼────┼────┼─···· ← horizontal road band
  ··   |    |    |   ··
13·····················

     ↑         ↑    ↑
     vertical road bands
```

---

## 6. State Representation

The RL state is a Python tuple with five integer values:

```
(
    taxi_x,          # taxi's grid column (0–19)
    taxi_y,          # taxi's grid row (0–13)
    destination_x,   # destination's grid column (0–19)
    destination_y,   # destination's grid row (0–13)
    traffic_level    # 0 = none, 1 = light, 2 = heavy
)
```

**Example state:**

```
(12, 6, 15, 10, 1)
```

This means: the taxi is at grid column 12, row 6; the destination is at column 15, row 10; and there is light traffic (one vehicle within radius 1 of the taxi).

Each unique tuple is an entry in the Q-table. If the agent visits a state it has never seen before, the Q-table initialises that entry with four zeros — one Q-value per action.

The traffic level is included in the state so the agent can distinguish between the same physical position with and without nearby traffic, allowing it to learn traffic-aware routing behaviour.

---

## 7. Action Space

| Action Code | Direction | Grid Delta |
|-------------|-----------|------------|
| 0           | UP        | (0, -1)    |
| 1           | DOWN      | (0, +1)    |
| 2           | LEFT      | (-1, 0)    |
| 3           | RIGHT     | (+1, 0)    |

The environment converts an action number into a (dx, dy) delta and applies it to the taxi's current grid position. If the resulting cell is not a valid road cell, the move is rejected and a penalty of –10 is applied. The taxi stays in its current position.

---

## 8. Reward Design

### Why Reward Shaping?

A pure sparse reward (+100 only at destination, 0 everywhere else) would require the agent to stumble upon the destination by chance before it can learn anything. On a 20 × 14 grid with random start/goal pairs, this can take a very long time. Reward shaping provides intermediate feedback — rewarding progress and penalising regression — so the agent learns directional behaviour much faster.

### Base Reward Table

| Situation                         | Reward |
|-----------------------------------|--------|
| Destination reached               | +100   |
| Move closer to destination (BFS)  | +2     |
| Same BFS distance after move      | -1     |
| Move farther from destination      | -4     |
| Invalid move (non-road cell)       | -10    |

Distance comparisons use the precomputed BFS shortest-road-distance, not Euclidean distance, so the agent is rewarded for road-following progress rather than straight-line proximity.

### Traffic Penalty Table

| Traffic Level        | Nearby Cars | Penalty |
|----------------------|-------------|---------|
| 0 — no traffic       | 0 cars      | 0       |
| 1 — light traffic    | 1 car       | -1      |
| 2 — heavy traffic    | 2+ cars     | -3      |

Traffic penalties are added to the base reward on every step. This teaches the agent that moving through congested areas has a cost.

---

## 9. Q-Learning Algorithm

### Conceptual Background

Q-Learning is a model-free, off-policy reinforcement learning algorithm. The agent maintains a table of Q-values, where each entry Q(s, a) represents the estimated total future reward of taking action a in state s and then behaving optimally thereafter.

### The Bellman Update

After every step the agent updates its Q-table using:

```
Q(s, a) ← Q(s, a) + α × [ r + γ × max Q(s', a') − Q(s, a) ]
                                         a'
```

Where:

| Symbol        | Meaning                          | Value in this project |
|---------------|----------------------------------|-----------------------|
| Q(s, a)       | Current estimate for (state, action) | stored in q_table |
| α (alpha)     | Learning rate                    | 0.1                   |
| r             | Reward received this step        | from environment.step |
| γ (gamma)     | Discount factor                  | 0.9                   |
| s'            | Next state                       | returned by step()    |
| max Q(s', a') | Best Q-value in next state       | computed from q_table |

### Implementation in `q_learning.py`

The `QLearningAgent` class stores the Q-table as a Python dictionary where each key is a state tuple and each value is a list of four floats (one per action):

```python
self.q_table = {
    (12, 6, 15, 10, 1): [0.2, -1.5, 0.8, 4.2],
    ...
}
```

If a state is visited for the first time, `get_q_values()` initialises it with `[0.0, 0.0, 0.0, 0.0]`.

The update is implemented in `update()`:

```python
target = reward + self.discount_factor * max(next_q_values)
new_q = current_q + self.learning_rate * (target - current_q)
```

If the episode is done (destination reached), the future term is omitted and the target is simply the final reward.

---

## 10. Exploration vs Exploitation

### Epsilon-Greedy Policy

| Condition          | Behaviour                      |
|--------------------|-------------------------------|
| random() < epsilon | Choose a random action (explore) |
| random() >= epsilon| Choose best Q-value action (exploit) |

When multiple actions share the same maximum Q-value, the agent picks randomly among tied actions to prevent systematic bias.

### Epsilon Decay Schedule (from `train.py`)

```
Initial epsilon:   1.0     (100% exploration)
Epsilon decay:     0.999   (per episode)
Minimum epsilon:   0.05    (5% floor)
```

Over 5,000 episodes:

```
Episode 1:    epsilon ≈ 1.000  → mostly random
Episode 1000: epsilon ≈ 0.368  → balanced
Episode 3000: epsilon ≈ 0.050  → mostly greedy (at floor)
Episode 5000: epsilon = 0.050  → fully exploiting learned policy
```

```
High epsilon               Low epsilon
     ↓                          ↓
Random actions           Best Q-value actions
     ↓                          ↓
Discover environment     Use learned knowledge
     ↓                          ↓
Build Q-table            Refine Q-table
```

The epsilon floor of 0.05 ensures the agent never becomes completely deterministic, which preserves some robustness to unseen states.

---

## 11. Traffic System

### Implementation (`traffic.py`)

The `TrafficManager` creates 10 `TrafficCar` instances at startup. Each car:

- Is placed on a randomly chosen valid road cell.
- Moves in one direction at 2 pixels per frame.
- When it reaches the centre of the next road cell, it uses `get_neighbors()` to choose a new connected road cell and changes direction.
- Never enters a building or non-road cell.

Traffic cars move continuously and independently of the taxi. They do not model real traffic flow (no lane discipline, no collision avoidance with each other), but they create a dynamic environment where the traffic state around the taxi changes from step to step.

### Traffic State Detection

```python
def get_traffic_level(self):
    nearby_count = self.traffic.count_nearby_cars(
        self.taxi_grid,
        radius=1           # Manhattan distance ≤ 1
    )
    if nearby_count == 0: return 0
    if nearby_count == 1: return 1
    return 2
```

The traffic level becomes part of the RL state tuple, meaning the Q-table learns to distinguish identical grid positions under different traffic conditions.

### Manhattan Distance Radius

```
radius = 1  →  cells directly adjacent to taxi
              (up, down, left, right)
```

---

## 12. Passenger System

### What Is Implemented

`passenger.py` provides a full `Passenger` and `PassengerManager` implementation with the following state machine:

```
waiting
  ↓  (taxi arrives at pickup location)
onboard
  ↓  (taxi arrives at passenger destination)
completed
```

Each passenger has:
- A randomly selected road-cell pickup location.
- A randomly selected road-cell destination (connected to pickup by a valid BFS path).
- Pygame drawing: shown as a blue circle with a skin-coloured head while waiting; hidden once onboard.

`main.py` creates a `PassengerManager`, spawns a passenger, detects pickup when the taxi reaches the passenger's location, updates the destination marker to the passenger's destination after pickup, and detects drop-off on arrival.

### Important Scope Clarification

The passenger system is **a visual simulation and manual demo feature**. It is **not integrated into the Q-learning training objective**. During training (`train.py`), the agent's only goal is to reach a randomly generated destination cell. Passenger pickup/drop-off is not part of the reward signal or the state representation in the current version.

---

## 13. Training Pipeline

### RL Training Loop

```
┌──────────────────────────────────────────┐
│  Start: episode = 1                       │
└──────────────────┬───────────────────────┘
                   │
          env.reset()  →  random start + destination
                   │
          state = get_state()
                   │
          ┌────────▼─────────────────────┐
          │  For each step (max 200)     │
          │                              │
          │  1. agent.choose_action(s)   │
          │      epsilon-greedy          │
          │                              │
          │  2. env.step(action)         │
          │      → next_state, reward,   │
          │         done                 │
          │                              │
          │  3. agent.update(s, a, r,    │
          │      s', done)               │
          │      → Bellman update        │
          │                              │
          │  4. state = next_state       │
          │                              │
          │  5. if done: break           │
          └────────────────────────────┬─┘
                                       │
          agent.decay_epsilon()
                                       │
          Log to training_metrics.csv
          (episode, reward, steps,
           success, epsilon, q_states)
                                       │
          if episode % 500 == 0: print
                                       │
          episode += 1
                                       │
          if episode > 5000: break ───→ agent.save("models/q_table.pkl")
```

### Key Training Parameters

| Parameter        | Value               |
|------------------|---------------------|
| Episodes         | 5,000               |
| Max steps/episode| 200                 |
| Learning rate α  | 0.1                 |
| Discount factor γ| 0.9                 |
| Initial epsilon  | 1.0                 |
| Epsilon decay    | 0.999 per episode   |
| Min epsilon      | 0.05                |
| Traffic cars     | 10                  |
| Traffic radius   | 1 (Manhattan)       |
| Model path       | models/q_table.pkl  |
| Metrics path     | results/training_metrics.csv |

---

## 14. Training Metrics CSV

**File:** `results/training_metrics.csv`  
**Rows:** One row per training episode (5,000 rows after a full run).

| Column       | Type    | Description                                                             |
|--------------|---------|-------------------------------------------------------------------------|
| `episode`    | Integer | Episode number (1 to 5000)                                              |
| `total_reward` | Float | Sum of all rewards collected during the episode (including penalties)   |
| `steps`      | Integer | Number of steps taken before termination (destination reached or limit) |
| `success`    | 0 or 1  | 1 if the agent reached the destination, 0 if the step limit was hit     |
| `epsilon`    | Float   | Exploration probability at the end of this episode                      |
| `q_states`   | Integer | Number of unique states in the Q-table at the end of this episode       |

### How to use this CSV

- Plot `episode` vs `total_reward` to see whether average reward improves over time.
- Plot `episode` vs `steps` to see whether the agent reaches destinations faster.
- Plot a rolling average of `success` over episode windows to track the learning rate.
- Plot `epsilon` to verify the decay schedule.
- Plot `q_states` to see how quickly the agent is discovering new states.

---

## 15. Route Analysis CSV

**File:** `results/route_analysis.csv`  
**Produced by:** `route_analysis.py`  
**Rows:** One row per evaluated test route (100 rows per run).

| Column              | Type    | Description                                                                           |
|---------------------|---------|---------------------------------------------------------------------------------------|
| `route`             | Integer | Route index (1 to 100)                                                                |
| `start_x`           | Integer | Taxi start grid column                                                                |
| `start_y`           | Integer | Taxi start grid row                                                                   |
| `destination_x`     | Integer | Destination grid column                                                               |
| `destination_y`     | Integer | Destination grid row                                                                  |
| `optimal_steps`     | Integer | Shortest road-following distance calculated by BFS                                    |
| `q_learning_steps`  | Integer | Steps the Q-learning agent actually took to reach the destination                     |
| `extra_steps`       | Integer | `q_learning_steps − optimal_steps` (blank if agent did not succeed)                  |
| `efficiency_percent`| Float   | `(optimal_steps / q_learning_steps) × 100` — 100% means perfectly optimal route      |
| `invalid_moves`     | Integer | Number of attempted moves that were rejected (non-road cell)                          |
| `reward`            | Float   | Total reward accumulated on this route                                                |
| `success`           | 0 or 1  | 1 if the agent reached the destination, 0 if the step limit (200) was exceeded        |

### How to interpret this CSV

A row like:

```
route=14, start=(5,4), dest=(17,10), optimal=14, q_learning=14, extra=0, efficiency=100.00, invalid=0, success=1
```

means the agent navigated the route perfectly — same number of steps as BFS, no invalid moves.

A row like:

```
route=27, start=(5,4), dest=(17,10), optimal=14, q_learning=22, extra=8, efficiency=63.64, invalid=3, success=1
```

means the agent reached the destination but took 8 extra steps and made 3 invalid move attempts.

A row with `success=0` means the agent ran out of steps without reaching the destination.

---

## 16. Training Charts

After running `python train.py` followed by `python visualize.py`, four charts are saved to `results/`:

### Reward Curve — `results/reward_curve.png`

- **X-axis:** Episode number (1–5000)
- **Y-axis:** Total reward for that episode
- **What to look for:** The trend should move upward over time, from very negative values early (many invalid moves, step-limit exceeded) toward positive values as the agent learns. Individual episode rewards are noisy; a smoothed trendline or rolling average confirms improvement.

### Steps Curve — `results/steps_curve.png`

- **X-axis:** Episode number
- **Y-axis:** Steps taken per episode
- **What to look for:** Steps should decrease over time as the agent finds shorter routes. The step limit is 200, so early failed episodes hit 200 steps; later successful episodes should trend much lower.

### Success Rate Curve — `results/success_curve.png`

- **X-axis:** Episode number
- **Y-axis:** Rolling success rate (%) over a 50-episode window
- **What to look for:** Should trend upward from near 0% toward a higher plateau, indicating the agent increasingly reaches the destination within the step limit.

### Epsilon Curve — `results/epsilon_curve.png`

- **X-axis:** Episode number
- **Y-axis:** Epsilon value (0.0 to 1.0)
- **What to look for:** A smooth exponential decay curve from 1.0 down to 0.05, flattening at the minimum. This confirms the exploration schedule is working as configured.

> **Note:** No GIF or video recording is included in the repository. To observe the agent's behaviour visually, run `python run_agent.py` after training.

---

## 17. Route Efficiency Analysis

### The Key Question

Success rate alone is not a complete measure of routing quality. A taxi agent that reaches every destination but takes twice as many steps as necessary is not a good routing agent. This project measures **route efficiency** by comparing the Q-learning agent's path against the shortest possible road-following path computed by BFS.

### Efficiency Formula

```
Efficiency (%) = (optimal_steps / q_learning_steps) × 100
```

Examples:

| Optimal Steps | Q-Learning Steps | Extra Steps | Efficiency |
|---------------|-----------------|-------------|------------|
| 10            | 10              | 0           | 100.00%    |
| 10            | 12              | 2           | 83.33%     |
| 10            | 20              | 10          | 50.00%     |

### Why Q-Learning Can Take More Steps Than BFS

BFS has global knowledge of the entire map and always finds the shortest path. The Q-learning agent has only its local state (position, destination, traffic level) and the Q-values accumulated from experience. Several factors can cause suboptimal routes:

- States the agent rarely visited during training may have poorly estimated Q-values.
- Traffic penalties may cause the agent to detour around congested areas (which may be desirable but adds steps).
- Ties between similar Q-values can lead to temporarily inefficient movement.

### Evaluation Flow

```
Load models/q_table.pkl
       ↓
For each of 100 test routes:
  env.reset() → random start + destination
       ↓
  BFS distance = env.get_distance(start, destination)
       ↓
  Run agent with epsilon=0.0 (no exploration)
       ↓
  Count steps, invalid moves, reward
       ↓
  efficiency = optimal / q_steps × 100
       ↓
  Write row to route_analysis.csv
       ↓
Print summary statistics
```

---

## 18. Evaluation Methodology

Both `evaluate.py` and `route_analysis.py` evaluate the trained agent on completely fresh routes generated after loading the model. This ensures the reported metrics reflect generalisation performance, not memorisation of training routes.

Key evaluation design decisions:

- **Epsilon = 0.0 during evaluation.** The agent always picks its best known action, so results are deterministic for any given Q-table and random route.
- **100 test routes.** Each route uses `env.reset()`, which randomly selects start and destination from the valid road cell pool.
- **Max 200 steps.** If the agent cannot reach the destination within 200 steps, the episode is recorded as a failure.
- **BFS as ground truth.** The `get_distance()` function uses the precomputed BFS map to determine the optimal number of steps for each test route.
- **Multiple metrics.** Reporting success rate, average steps, average efficiency, and average invalid moves provides a more complete picture than any single number.

---

## 19. Model Persistence

### What Is Saved

The `QLearningAgent.save()` method saves the complete agent state to a pickle file:

```python
data = {
    "q_table":        self.q_table,          # dict: state → [Q_UP, Q_DOWN, Q_LEFT, Q_RIGHT]
    "actions":        self.actions,           # 4
    "learning_rate":  self.learning_rate,     # 0.1
    "discount_factor":self.discount_factor,   # 0.9
    "epsilon":        self.epsilon,           # current epsilon at save time
    "epsilon_decay":  self.epsilon_decay,     # 0.999
    "min_epsilon":    self.min_epsilon,       # 0.05
}
```

### Atomic Write

The save method writes to a temporary file first (`q_table.pkl.tmp`) and then uses `os.replace()` to atomically replace the target file. This prevents a partial write from corrupting the saved model.

### Loading

`QLearningAgent.load()` reads the pickle file and restores all fields. Any field not present in the file (for backward compatibility) falls back to the current agent's value.

### File Location

```
models/q_table.pkl
```

The `models/` directory is created automatically by `train.py` using `os.makedirs("models", exist_ok=True)`.

---

## 20. Project File Structure

```
Taxi-route-optimization/
│
├── main.py                 # Pygame demo: manual keyboard control + passenger system
├── train.py                # Q-learning training loop (5000 episodes)
├── evaluate.py             # Load model, test on 100 fresh routes, print metrics
├── route_analysis.py       # Route-vs-BFS efficiency analysis, writes route_analysis.csv
├── route_visualization.py  # Pygame visualization of route analysis (path drawing)
├── run_agent.py            # Watch the trained agent navigate live in Pygame
├── compare_agents.py       # Comparison utility for multiple agent configurations
├── visualize.py            # Generate training charts from training_metrics.csv
│
├── environment.py          # TaxiEnvironment: MDP, rewards, BFS distance maps
├── q_learning.py           # QLearningAgent: Q-table, epsilon-greedy, Bellman update, save/load
├── taxi.py                 # Taxi sprite: position, drawing, set_position synchronisation
├── world.py                # City world: roads, buildings, destination rendering
├── traffic.py              # TrafficCar + TrafficManager: moving vehicles on road network
├── passenger.py            # Passenger + PassengerManager: pickup/drop-off simulation
│
├── requirements.txt        # Python dependencies (pygame, matplotlib, numpy, etc.)
├── .gitignore
│
├── models/
│   └── q_table.pkl         # Trained Q-table (created by train.py)
│
└── results/
    ├── training_metrics.csv    # Per-episode training log
    ├── route_analysis.csv      # Per-route BFS comparison log
    ├── reward_curve.png        # Episode reward chart
    ├── steps_curve.png         # Episode steps chart
    ├── success_curve.png       # Rolling success rate chart
    └── epsilon_curve.png       # Epsilon decay chart
```

---

## 21. Installation

Python 3.9 or later is recommended based on the pinned dependency versions.

```bash
# Clone the repository
git clone https://github.com/programmer-aviral/Taxi-route-optimization.git
cd Taxi-route-optimization

# Create a virtual environment
python -m venv .venv

# Activate — Windows PowerShell
.venv\Scripts\Activate.ps1

# Activate — macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies (from `requirements.txt`)

| Package         | Version   | Purpose                              |
|-----------------|-----------|--------------------------------------|
| pygame          | 2.6.1     | 2D simulation and rendering          |
| matplotlib      | 3.11.1    | Training charts                      |
| numpy           | 2.5.3     | Numerical support                    |
| pillow          | 12.3.0    | Image handling for matplotlib        |
| contourpy       | 1.3.3     | matplotlib dependency                |
| cycler          | 0.12.1    | matplotlib dependency                |
| fonttools       | 4.65.0    | matplotlib dependency                |
| kiwisolver      | 1.5.1     | matplotlib dependency                |
| packaging       | 26.3      | matplotlib dependency                |
| pyparsing       | 3.3.2     | matplotlib dependency                |
| python-dateutil | 2.9.0     | matplotlib dependency                |
| six             | 1.17.0    | matplotlib dependency                |

---

## 22. How to Run

All commands should be run from the project root directory with the virtual environment active.

```bash
# Step 1 — Train the agent
python train.py
# Runs 5000 episodes. Saves model to models/q_table.pkl
# Logs per-episode metrics to results/training_metrics.csv
# Progress is printed every 500 episodes

# Step 2 — Evaluate the trained agent
python evaluate.py
# Loads models/q_table.pkl
# Tests on 100 fresh random routes with epsilon=0.0
# Prints: success rate, average reward, average steps

# Step 3 — Route efficiency analysis
python route_analysis.py
# Loads models/q_table.pkl
# Compares Q-learning routes against BFS optimal routes on 100 test routes
# Writes detailed per-route data to results/route_analysis.csv
# Prints summary statistics

# Step 4 — Generate training charts
python visualize.py
# Reads results/training_metrics.csv
# Saves: reward_curve.png, steps_curve.png,
#        success_curve.png, epsilon_curve.png

# Step 5 — Manual demo (keyboard controlled)
python main.py
# Opens Pygame window
# Drive the taxi manually; passenger spawns at a random location

# Step 6 — Watch the trained agent
python run_agent.py
# Opens Pygame window
# Loads trained Q-table
# Agent navigates autonomously at 5 FPS so the route is visible
```

---

## 23. Demo Controls

The following controls are available in `main.py` (manual demo mode):

| Key              | Action    |
|------------------|-----------|
| Arrow Key UP     | Move Up   |
| Arrow Key DOWN   | Move Down |
| Arrow Key LEFT   | Move Left |
| Arrow Key RIGHT  | Move Right|
| W                | Move Up   |
| S                | Move Down |
| A                | Move Left |
| D                | Move Right|
| Window Close     | Exit      |

The status bar shows:
- Current passenger state (waiting / onboard / completed)
- Event messages (passenger picked up, trip completed)
- Controls reminder

`run_agent.py` does not use keyboard input; the trained agent controls the taxi autonomously.

---

## 24. Challenges and Engineering Decisions

### Taxi Position Synchronisation

**Problem:** Early in development, the taxi's visual `pygame.Rect` position (used for rendering and collision detection) could become desynchronised from the logical grid position tracked by `self.taxi_grid`.

**Diagnosis:** The `Taxi` object had a separate `rect` attribute that was only updated when `pygame.sprite` draw calls fired, while `taxi_grid` was updated directly in the environment logic.

**Fix:** A `Taxi.set_position(x, y)` method was introduced that updates both `self.x`, `self.y`, and `self.rect` atomically. The environment calls `set_position` after every valid move.

**Lesson:** In simulation environments, logical and visual state must be kept in strict sync. Separating update responsibilities without a single source of truth leads to subtle rendering bugs that only appear under certain conditions.

### Grid-to-Pixel Coordinate Mapping

**Problem:** Road cells are defined in pixel ranges (e.g., horizontal band 200–300px), while the Q-learning state uses grid indices (e.g., row 4). Confusing the two coordinate systems caused incorrect road-validity checks.

**Diagnosis:** `is_road_cell()` receives a grid cell `(grid_x, grid_y)` but must check pixel-based road band definitions. The conversion must happen inside the method.

**Fix:** `is_road_cell()` converts grid coordinates to the cell's centre pixel (`grid_x * CELL_SIZE + CELL_SIZE // 2`) before checking against road bands.

**Lesson:** When two coordinate systems coexist in a simulation, create dedicated conversion functions (`grid_to_pixel`, `pixel_to_grid`) and always use them rather than performing arithmetic inline.

### Atomic Model Saving

**Problem:** If the training process was interrupted mid-write, the pickle file could be partially written and corrupt, making it unloadable on the next run.

**Fix:** `QLearningAgent.save()` writes to a `.tmp` file first and uses `os.replace()` to swap it in atomically. Either the old model or the new model is intact; a partial write never replaces the working model.

### Disconnected Route Detection in Reset

**Problem:** If `env.reset()` selected a start and destination that were not connected by a valid road path, the BFS distance would return infinity, making the episode impossible to solve.

**Fix:** `reset()` loops until it finds a (start, destination) pair where `get_distance(start, destination) != float("inf")`.

---

## 25. Key Learnings

### Machine Learning

- **Reinforcement Learning** is fundamentally different from supervised learning: there is no ground truth label, only a reward signal from the environment. The agent must discover structure through exploration.
- **Q-Learning** is an off-policy algorithm, meaning it can learn from exploratory actions without being constrained to follow the current policy. This makes it sample-efficient for tabular problems.
- **Q-Table structure**: the key insight is that the Q-table is simply a lookup table mapping (state, action) pairs to estimated values. It grows as new states are visited.
- **Reward shaping**: a well-designed intermediate reward signal dramatically reduces the number of episodes needed to learn useful behaviour. Designing the reward is as important as implementing the algorithm.
- **Epsilon decay**: the balance between exploration and exploitation is one of the most important hyperparameters in RL. Too fast a decay leads to premature convergence; too slow a decay wastes training time on random actions.

### Software Engineering

- **Modular architecture**: separating `environment.py`, `q_learning.py`, `taxi.py`, `world.py`, `traffic.py`, and `passenger.py` made it possible to test and modify each component independently.
- **State synchronisation**: in real-time simulation, keeping visual and logical state in sync is a non-trivial engineering challenge.
- **Atomic file writes**: model persistence should never leave the saved model in a partially written state.
- **CSV logging**: logging structured metrics to CSV files during training is inexpensive and enables rich post-hoc analysis without re-running training.
- **Coordinate system discipline**: defining clear boundaries between pixel space and grid space, and using dedicated conversion functions, prevents an entire class of subtle bugs.

### Data Analysis

- **Route comparison**: success rate alone is an insufficient metric for a routing agent. The efficiency ratio (optimal steps / actual steps) is a more informative measure.
- **Rolling averages**: episode-level metrics are noisy. Rolling averages over a window of 50 episodes reveal the true learning trend.
- **Reproducibility**: because start/destination positions are randomly generated, reported metrics are run-dependent. This is an important caveat to communicate clearly.

---

## 26. Limitations

The following limitations reflect the current implementation honestly:

- **Tabular Q-learning does not scale.** The Q-table maps every unique state tuple to a list of Q-values. As the state space grows (larger grid, more features), the table grows too large to be practical. Function approximation (e.g., neural networks) is needed for larger environments.
- **Discrete grid environment.** The city is a fixed 20 × 14 grid. This is adequate for demonstrating RL but is not a realistic map.
- **Simplified traffic model.** Traffic vehicles move randomly between connected road cells. They do not model lane discipline, queuing, stop signals, or real urban congestion patterns.
- **Traffic is not globally optimised.** Traffic behaviour is emergent from random movement, not from any traffic flow model.
- **Passenger system is not in the RL training loop.** The pickup/drop-off sequence is implemented and demonstrated in `main.py`, but it is not part of the Q-learning reward or state. The agent currently learns only to navigate to a single destination.
- **No neural network.** This is a tabular algorithm. DQN, PPO, and other deep RL approaches would be needed for larger and more complex environments.
- **No real geographic data.** The city is a synthetic grid. Real-world routing requires integration with map data.
- **Run-to-run variability.** Random route generation means evaluation metrics vary between runs.

---

## 27. Future Improvements

### Version 2 — Passenger-Aware RL

- Integrate passenger pickup and drop-off into the RL training objective.
- Extend the state to include passenger status (waiting, onboard, completed).
- Redesign the reward to include pickup and drop-off events.
- Train the agent on the full two-stage task: navigate to passenger, then navigate to destination.

### Version 3 — Improved Environment

- Add more varied traffic patterns (directional traffic, congestion zones).
- Increase grid size.
- Add dynamic obstacles or road closures.
- Implement a richer state representation (direction of travel, distance to passenger, elapsed steps).

### Version 4 — Deep Q-Network (DQN)

- Replace the Q-table with a neural network using PyTorch or TensorFlow.
- Implement experience replay and a target network.
- Allow the agent to generalise across larger state spaces without table explosion.
- Support larger and more complex city layouts.

### Version 5 — Realistic Map Data

- Integrate OpenStreetMap data for real road networks.
- Convert geographic coordinates into a grid representation.
- Train and evaluate on real urban environments.

> All items in this roadmap are future plans and are not implemented in the current repository.

---

## 28. Interview Explanation

*How I would describe this project in approximately 60–90 seconds:*

"I built a 2D taxi navigation simulation where a Q-learning agent learns to drive from a random starting point to a random destination on a city road network. The environment is a 20 × 14 grid with predefined road corridors and 10 moving traffic vehicles.

The state the agent sees at each step is a tuple of five values: its own position, the destination's position, and the current traffic level — zero, light, or heavy — based on how many other cars are within one grid cell of the taxi.

I designed a shaped reward: plus 100 for reaching the destination, plus 2 for moving closer by BFS distance, minus 4 for moving farther, minus 10 for invalid moves, and an additional traffic penalty of minus 1 or minus 3 depending on congestion. The shaping was important — without intermediate feedback, the agent would rarely discover the destination and could not learn.

The agent uses standard Q-learning with an epsilon-greedy policy and an epsilon decay from 1.0 down to 0.05 over 5,000 episodes. The Q-table maps state tuples to four Q-values — one per direction — and grows dynamically as new states are encountered. After training I save the table to a pickle file and reload it for evaluation.

For evaluation I measure not just success rate but route efficiency: the ratio of the BFS-optimal steps to the agent's actual steps. This is logged per route to a CSV file and summarised.

The main thing I learned is that reward design is at least as important as algorithm choice in RL. I also built proper engineering around the project — CSV logging, model persistence with atomic writes, coordinate system discipline, and a Pygame visualizer so I could actually watch the agent navigate."

---

## 29. Interview Q&A

**Q1: Why did you choose Q-Learning rather than a planning algorithm?**

Q-Learning demonstrates the core RL loop — agent, environment, reward, state, action, Q-table — clearly and without the complexity of deep networks. It was the appropriate choice for a finite, discrete grid environment where the state space is manageable. A planning algorithm such as A* would find an optimal route given the map, but it would not learn from interaction or handle dynamic traffic as part of the policy.

**Q2: What does BFS do in your project?**

BFS is used only to measure the shortest road-following distance between two cells. It answers the question "how many steps would a perfect navigator need?" This measurement is used for: (a) shaping the reward (closer = +2, farther = -4) and (b) computing route efficiency in evaluation. BFS is never used to select the agent's actions. The Q-learning agent decides where to move using its Q-table.

**Q3: What is the state representation and why did you include traffic level?**

The state is `(taxi_x, taxi_y, destination_x, destination_y, traffic_level)`. Without traffic level, the agent would see the same Q-values at a given position regardless of congestion, and could not learn to respond to traffic. Including traffic level allows the Q-table to associate higher action costs with congested positions.

**Q4: Why use reward shaping instead of a sparse reward?**

A sparse reward of +100 only at the destination gives the agent no signal during the vast majority of steps. On a 20 × 14 grid with random start/goal pairs, the agent would rarely stumble upon the destination by chance, especially early in training when epsilon is near 1. Shaped rewards — rewarding progress and penalising regression — provide a learning signal at every step, dramatically reducing the number of episodes needed.

**Q5: What does epsilon represent and what happens when it reaches the minimum?**

Epsilon is the probability of choosing a random action instead of the best known action. When epsilon reaches the minimum floor of 0.05, the agent explores randomly 5% of the time and exploits its Q-values 95% of the time. The floor prevents the agent from becoming fully deterministic, which would stop it from discovering new states it may have missed during training.

**Q6: How do you define an optimal route in this project?**

An optimal route is the route with the fewest possible road-following steps between the start and destination, as computed by BFS over the valid road cell graph. A route with efficiency = 100% means the agent took exactly as many steps as BFS would prescribe.

**Q7: Why might the Q-learning agent take more steps than BFS?**

Several reasons: the agent's Q-values are estimates, not exact; the agent may have limited experience with certain start/destination combinations; traffic penalties may cause the agent to take longer alternative paths (which may be reasonable if those paths have less congestion); and ties between similar Q-values can introduce momentary inefficiency.

**Q8: How do you evaluate the model without overfitting to training routes?**

Both `evaluate.py` and `route_analysis.py` call `env.reset()` after loading the model. This generates completely fresh random routes that the agent has not seen during training. Evaluation is done with `epsilon=0.0` so the agent is fully exploiting its learned Q-values.

**Q9: What happens if the state space grows too large for a Q-table?**

The dictionary-based Q-table grows with every new state visited. For a large grid, many destinations, or a richer state representation (e.g., full passenger state), the table could become too large to fit in memory and too sparse to generalise. The solution is function approximation — replacing the table with a neural network (DQN) that can generalise across similar states without requiring every state to be visited explicitly.

**Q10: How is the passenger system related to Q-learning?**

Currently it is not. The passenger system is a visual demonstration in `main.py` that shows pickup/drop-off logic and Pygame rendering. It is not part of the training reward or the state representation. Integrating the passenger into RL training (two-phase navigation with pickup and drop-off) is listed as a future improvement.

**Q11: Why use pickle for model persistence?**

Pickle serialises any Python object, making it straightforward to save and load the dictionary-based Q-table along with all agent hyperparameters. The save method uses atomic write (temp file + `os.replace`) to prevent corruption. For larger or production systems, a more structured format (HDF5, JSON, PyTorch checkpoint) would be preferable.

**Q12: What is the significance of the discount factor γ = 0.9?**

γ = 0.9 means the agent values a reward one step in the future at 90% of its face value. After 10 steps, a future reward is worth 0.9^10 ≈ 35% of its immediate value. This encourages the agent to seek rewards reasonably quickly rather than indefinitely deferring them. A lower γ would make the agent more short-sighted; a value of 1.0 would treat all future rewards equally regardless of distance.

**Q13: How would you improve this project?**

Three concrete next steps: first, integrate the passenger system into the RL training objective. Second, implement DQN to handle larger state spaces. Third, add a richer traffic model with lane awareness and congestion zones to make the routing problem more realistic.

---

## 30. Technical Glossary

| Term              | Definition                                                                                                   |
|-------------------|--------------------------------------------------------------------------------------------------------------|
| **Agent**         | The decision-making entity — in this project, the taxi.                                                      |
| **Environment**   | The world the agent interacts with — the 2D city grid with roads, traffic, and destination.                  |
| **State**         | A description of the current situation that the agent uses to make decisions.                                |
| **Action**        | A choice the agent makes at each step: UP, DOWN, LEFT, or RIGHT.                                             |
| **Reward**        | A scalar signal the environment gives the agent after each action, indicating how good or bad the action was. |
| **Episode**       | One complete run from a random start to either the destination or the step limit.                            |
| **Policy**        | The rule that maps states to actions. In this project: epsilon-greedy over the Q-table.                      |
| **Q-Value**       | The estimated total future reward of taking a specific action in a specific state and then acting optimally.  |
| **Q-Table**       | A dictionary mapping state tuples to lists of Q-values, one per action.                                      |
| **Bellman Update**| The equation that updates Q-values based on observed rewards and estimated future rewards.                   |
| **Epsilon**       | The exploration probability — the chance that the agent picks a random action instead of its best known one.  |
| **Exploration**   | Trying new actions to discover information about the environment.                                            |
| **Exploitation**  | Using the best known Q-values to take the action expected to yield the highest reward.                        |
| **BFS**           | Breadth-First Search — a graph traversal algorithm that finds shortest paths in unweighted graphs.           |
| **Route Efficiency** | The ratio of BFS-optimal steps to Q-learning steps, expressed as a percentage. 100% = perfectly optimal. |
| **Reward Shaping**| Adding intermediate rewards to guide learning faster than a sparse terminal reward alone would allow.         |
| **MDP**           | Markov Decision Process — the formal mathematical framework for sequential decision making under uncertainty. |

---

## 31. Results Disclaimer

Evaluation metrics reported by `evaluate.py` and `route_analysis.py` depend on the specific random routes and traffic configurations generated during each run. They also depend on the quality of the training run that produced the Q-table being evaluated. Different training runs, different random seeds, and different system states will produce different numbers.

Reported values should be treated as illustrative of what the project achieves under typical conditions, not as guaranteed performance bounds. The project is an educational RL simulation, not a production navigation system.

---

## 32. Author

**Aviral Gandhi**

GitHub: [https://github.com/programmer-aviral](https://github.com/programmer-aviral)

Repository: [https://github.com/programmer-aviral/Taxi-route-optimization](https://github.com/programmer-aviral/Taxi-route-optimization)
