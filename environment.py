
import random
from collections import deque

import pygame

from taxi import Taxi
from world import World
from traffic import TrafficManager


class TaxiEnvironment:
    """
    Reinforcement Learning environment for taxi route optimization.

    Actions:
        0 = UP
        1 = DOWN
        2 = LEFT
        3 = RIGHT

    State:
        (
            taxi_x,
            taxi_y,
            destination_x,
            destination_y,
            traffic_level
        )

    Traffic levels:
        0 = no nearby traffic
        1 = light traffic
        2 = heavy traffic
    """

    # =========================================================
    # ACTIONS
    # =========================================================

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    # =========================================================
    # GRID / WORLD
    # =========================================================

    CELL_SIZE = 50

    GRID_WIDTH = 20
    GRID_HEIGHT = 14

    WORLD_WIDTH = GRID_WIDTH * CELL_SIZE
    WORLD_HEIGHT = GRID_HEIGHT * CELL_SIZE

    # =========================================================
    # ROAD LAYOUT
    # =========================================================

    HORIZONTAL_ROADS = [
        (200, 300),
        (450, 550),
    ]

    VERTICAL_ROADS = [
        (250, 350),
        (550, 650),
        (850, 950),
    ]

    # =========================================================
    # TRAFFIC SETTINGS
    # =========================================================

    NUMBER_OF_TRAFFIC_CARS = 10
    TRAFFIC_RADIUS = 1

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self):
        pygame.init()

        self.world = World()
        self.taxi = Taxi()

        self.valid_cells = []
        self.distance_maps = {}

        self.taxi_grid = None
        self.destination_grid = None

        # Build road network
        self._build_road_map()

        # Build shortest-path maps
        self._build_distance_maps()

        # Create traffic manager
        self.traffic = TrafficManager(
            self,
            number_of_cars=self.NUMBER_OF_TRAFFIC_CARS
        )

    # =========================================================
    # GRID TO PIXEL
    # =========================================================

    def grid_to_pixel(self, grid_x, grid_y):
        """
        Convert grid coordinates into pixel coordinates.

        Example:
            grid (5, 4)
            ->
            pixel (250, 200)
        """

        return (
            grid_x * self.CELL_SIZE,
            grid_y * self.CELL_SIZE
        )

    # =========================================================
    # PIXEL TO GRID
    # =========================================================

    def pixel_to_grid(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates into grid coordinates.
        """

        return (
            int(pixel_x // self.CELL_SIZE),
            int(pixel_y // self.CELL_SIZE)
        )

    # =========================================================
    # ROAD CELL CHECK
    # =========================================================

    def is_road_cell(self, grid_x, grid_y):
        """
        Check whether a grid cell belongs to a road.
        """

        # Outside grid
        if not (
            0 <= grid_x < self.GRID_WIDTH
            and 0 <= grid_y < self.GRID_HEIGHT
        ):
            return False

        # Convert grid cell to pixel area
        x = grid_x * self.CELL_SIZE
        y = grid_y * self.CELL_SIZE

        # Use center of the cell
        center_x = x + self.CELL_SIZE // 2
        center_y = y + self.CELL_SIZE // 2

        # Horizontal roads
        for y1, y2 in self.HORIZONTAL_ROADS:
            if y1 <= center_y < y2:
                return True

        # Vertical roads
        for x1, x2 in self.VERTICAL_ROADS:
            if x1 <= center_x < x2:
                return True

        return False

    # =========================================================
    # BUILD ROAD MAP
    # =========================================================

    def _build_road_map(self):
        """
        Create the list of valid driving cells.

        A cell is valid when:
            1. It is on a road.
            2. The taxi does not collide with a building there.
        """

        self.valid_cells = []

        for grid_y in range(self.GRID_HEIGHT):

            for grid_x in range(self.GRID_WIDTH):

                # Must be road
                if not self.is_road_cell(
                    grid_x,
                    grid_y
                ):
                    continue

                # Convert to pixels
                pixel_x, pixel_y = self.grid_to_pixel(
                    grid_x,
                    grid_y
                )

                # Check building collision
                if self.world.is_valid_position(
                    pixel_x,
                    pixel_y,
                    self.taxi.width,
                    self.taxi.height
                ):
                    self.valid_cells.append(
                        (grid_x, grid_y)
                    )

    # =========================================================
    # VALID CELL
    # =========================================================

    def is_valid_cell(self, grid_x, grid_y):
        """
        Check whether a grid cell is driveable.
        """

        return (
            grid_x,
            grid_y
        ) in self.valid_cells

    # =========================================================
    # GET NEIGHBORS
    # =========================================================

    def get_neighbors(self, cell):
        """
        Return all directly connected valid road cells.
        """

        x, y = cell

        possible_neighbors = [
            (x, y - 1),  # UP
            (x, y + 1),  # DOWN
            (x - 1, y),  # LEFT
            (x + 1, y),  # RIGHT
        ]

        neighbors = []

        for next_x, next_y in possible_neighbors:

            if self.is_valid_cell(
                next_x,
                next_y
            ):
                neighbors.append(
                    (next_x, next_y)
                )

        return neighbors

    # =========================================================
    # BUILD DISTANCE MAPS
    # =========================================================

    def _build_distance_maps(self):
        """
        Calculate shortest road distances using BFS.

        IMPORTANT:
        BFS is only used to measure distance.

        BFS does NOT choose actions for the RL agent.
        """

        self.distance_maps = {}

        for destination in self.valid_cells:

            distances = {
                destination: 0
            }

            queue = deque([
                destination
            ])

            while queue:

                current = queue.popleft()

                current_distance = distances[
                    current
                ]

                for neighbor in self.get_neighbors(
                    current
                ):

                    if neighbor not in distances:

                        distances[neighbor] = (
                            current_distance + 1
                        )

                        queue.append(
                            neighbor
                        )

            self.distance_maps[
                destination
            ] = distances

    # =========================================================
    # GET DISTANCE
    # =========================================================

    def get_distance(
        self,
        cell,
        destination
    ):
        """
        Return the shortest road distance
        between cell and destination.
        """

        distances = self.distance_maps.get(
            destination,
            {}
        )

        return distances.get(
            cell,
            float("inf")
        )

    # =========================================================
    # GET TRAFFIC LEVEL
    # =========================================================

    def get_traffic_level(self):
        """
        Determine traffic around the taxi.

        0 = no traffic
        1 = light traffic
        2 = heavy traffic
        """

        if self.taxi_grid is None:
            return 0

        nearby_count = self.traffic.count_nearby_cars(
            self.taxi_grid,
            radius=self.TRAFFIC_RADIUS
        )

        if nearby_count == 0:
            return 0

        if nearby_count == 1:
            return 1

        return 2

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):
        """
        Start a new episode.

        Randomly selects:
            - taxi starting road cell
            - destination road cell
            - fresh traffic configuration
        """

        if not self.valid_cells:
            raise RuntimeError(
                "No valid road cells were found."
            )

        # -----------------------------------------------------
        # Choose connected start and destination
        # -----------------------------------------------------

        while True:

            start = random.choice(
                self.valid_cells
            )

            destination = random.choice(
                self.valid_cells
            )

            # They must be different
            if start == destination:
                continue

            # Calculate shortest road distance
            distance = self.get_distance(
                start,
                destination
            )

            # They must be connected
            if distance != float("inf"):
                break

        # Save grid coordinates
        self.taxi_grid = start
        self.destination_grid = destination

        # -----------------------------------------------------
        # Fresh traffic
        # -----------------------------------------------------

        self.traffic = TrafficManager(
            self,
            number_of_cars=self.NUMBER_OF_TRAFFIC_CARS
        )

        # -----------------------------------------------------
        # Position taxi
        # -----------------------------------------------------

        taxi_x, taxi_y = self.grid_to_pixel(
            start[0],
            start[1]
        )

        # IMPORTANT:
        # Use set_position so x/y and rect
        # stay synchronized.
        self.taxi.set_position(
            taxi_x,
            taxi_y
        )

        # -----------------------------------------------------
        # Position destination
        # -----------------------------------------------------

        destination_x, destination_y = (
            self.grid_to_pixel(
                destination[0],
                destination[1]
            )
        )

        self.world.destination = pygame.Rect(
            destination_x,
            destination_y,
            self.CELL_SIZE,
            self.CELL_SIZE
        )

        return self.get_state()

    # =========================================================
    # GET STATE
    # =========================================================

    def get_state(self):
        """
        Return the RL state.

        Format:
            (
                taxi_x,
                taxi_y,
                destination_x,
                destination_y,
                traffic_level
            )
        """

        if self.taxi_grid is None:
            raise RuntimeError(
                "Environment has not been reset."
            )

        if self.destination_grid is None:
            raise RuntimeError(
                "Destination has not been created."
            )

        traffic_level = self.get_traffic_level()

        return (
            self.taxi_grid[0],
            self.taxi_grid[1],
            self.destination_grid[0],
            self.destination_grid[1],
            traffic_level
        )

    # =========================================================
    # ACTION TO DELTA
    # =========================================================

    def get_action_delta(self, action):
        """
        Convert action number into grid movement.
        """

        if action == self.UP:
            return 0, -1

        if action == self.DOWN:
            return 0, 1

        if action == self.LEFT:
            return -1, 0

        if action == self.RIGHT:
            return 1, 0

        raise ValueError(
            f"Invalid action: {action}"
        )

    # =========================================================
    # STEP
    # =========================================================

    def step(self, action):
        """
        Perform one RL action.

        Base rewards:

            Destination reached = +100
            Move closer         = +2
            Same distance       = -1
            Move farther        = -4
            Invalid movement    = -10

        Traffic penalties:

            No traffic          = 0
            Light traffic       = -1
            Heavy traffic       = -3
        """

        # -----------------------------------------------------
        # Update traffic
        # -----------------------------------------------------

        self.traffic.update()

        # -----------------------------------------------------
        # Current taxi position
        # -----------------------------------------------------

        current_x, current_y = self.taxi_grid

        # -----------------------------------------------------
        # Distance before movement
        # -----------------------------------------------------

        old_distance = self.get_distance(
            self.taxi_grid,
            self.destination_grid
        )

        # -----------------------------------------------------
        # Traffic before movement
        # -----------------------------------------------------

        traffic_level = self.get_traffic_level()

        # -----------------------------------------------------
        # Convert action to movement
        # -----------------------------------------------------

        dx, dy = self.get_action_delta(
            action
        )

        next_x = current_x + dx
        next_y = current_y + dy

        # =====================================================
        # INVALID MOVEMENT
        # =====================================================

        if not self.is_valid_cell(
            next_x,
            next_y
        ):

            reward = -10
            done = False

            # Taxi stays in the same grid cell
            self.taxi_grid = (
                current_x,
                current_y
            )

            return (
                self.get_state(),
                reward,
                done
            )

        # =====================================================
        # VALID MOVEMENT
        # =====================================================

        self.taxi_grid = (
            next_x,
            next_y
        )

        # Convert grid -> pixels
        pixel_x, pixel_y = self.grid_to_pixel(
            next_x,
            next_y
        )

        # IMPORTANT:
        # Update BOTH taxi position and rect.
        self.taxi.set_position(
            pixel_x,
            pixel_y
        )

        # =====================================================
        # DESTINATION REACHED
        # =====================================================

        if self.taxi_grid == self.destination_grid:

            reward = 100
            done = True

            return (
                self.get_state(),
                reward,
                done
            )

        # =====================================================
        # DISTANCE AFTER MOVEMENT
        # =====================================================

        new_distance = self.get_distance(
            self.taxi_grid,
            self.destination_grid
        )

        # =====================================================
        # BASE REWARD
        # =====================================================

        if new_distance < old_distance:

            # Moved closer
            reward = 2

        elif new_distance > old_distance:

            # Moved farther
            reward = -4

        else:

            # Same distance
            reward = -1

        # =====================================================
        # TRAFFIC PENALTY
        # =====================================================

        if traffic_level == 0:

            traffic_penalty = 0

        elif traffic_level == 1:

            traffic_penalty = -1

        else:

            traffic_penalty = -3

        reward += traffic_penalty

        done = False

        return (
            self.get_state(),
            reward,
            done
        )

    # =========================================================
    # RENDER
    # =========================================================

    def render(self, screen):
        """
        Draw the complete environment.
        """

        # Draw world
        self.world.draw(screen)

        # Draw destination
        if self.world.destination is not None:

            pygame.draw.rect(
                screen,
                (0, 200, 0),
                self.world.destination
            )

        # Draw taxi
        self.taxi.draw(
            screen
        )

        # Draw traffic
        self.traffic.draw(
            screen
        )

