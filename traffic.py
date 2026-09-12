import random

import pygame


class TrafficCar:
    """
    Represents one traffic car in the city.

    Traffic cars move only between valid road cells.

    The movement directions are:

        (1, 0)   = RIGHT
        (-1, 0)  = LEFT
        (0, 1)   = DOWN
        (0, -1)  = UP
    """

    # =========================================================
    # CAR SETTINGS
    # =========================================================

    WIDTH = 30
    HEIGHT = 20

    # Pixel movement speed.
    # The road grid uses 50-pixel cells.
    SPEED = 2

    # =========================================================
    # CONSTRUCTOR
    # =========================================================

    def __init__(
        self,
        environment,
        start_cell=None
    ):
        self.environment = environment

        # -----------------------------------------------------
        # Choose starting road cell
        # -----------------------------------------------------

        if start_cell is None:

            if not self.environment.valid_cells:
                raise ValueError(
                    "No valid road cells are available."
                )

            start_cell = random.choice(
                self.environment.valid_cells
            )

        self.grid_x = start_cell[0]
        self.grid_y = start_cell[1]

        # -----------------------------------------------------
        # Convert grid position to pixels
        # -----------------------------------------------------

        self.x, self.y = (
            self.environment.grid_to_pixel(
                self.grid_x,
                self.grid_y
            )
        )

        # Current movement direction
        self.direction = None

        # Choose first direction
        self.choose_direction()


    # =========================================================
    # CHOOSE DIRECTION
    # =========================================================

    def choose_direction(self):
        """
        Choose one valid neighboring road cell.

        The traffic car does not move through buildings
        or non-road cells.
        """

        neighbors = (
            self.environment.get_neighbors(
                (self.grid_x, self.grid_y)
            )
        )

        # No available road neighbor
        if not neighbors:
            self.direction = None
            return

        # Choose random connected road cell
        next_cell = random.choice(
            neighbors
        )

        next_x, next_y = next_cell

        delta_x = next_x - self.grid_x
        delta_y = next_y - self.grid_y

        # -----------------------------------------------------
        # Convert movement into direction
        # -----------------------------------------------------

        if delta_x == 1:

            self.direction = (
                1,
                0
            )

        elif delta_x == -1:

            self.direction = (
                -1,
                0
            )

        elif delta_y == 1:

            self.direction = (
                0,
                1
            )

        elif delta_y == -1:

            self.direction = (
                0,
                -1
            )

        else:

            self.direction = None


    # =========================================================
    # UPDATE
    # =========================================================

    def update(self):
        """
        Move the traffic car.

        The car travels from its current grid cell toward
        the next connected road cell.

        When it reaches that cell, it chooses another
        connected road cell.
        """

        # -----------------------------------------------------
        # No direction
        # -----------------------------------------------------

        if self.direction is None:

            self.choose_direction()
            return

        dx, dy = self.direction

        # -----------------------------------------------------
        # Move in pixels
        # -----------------------------------------------------

        self.x += dx * self.SPEED
        self.y += dy * self.SPEED

        # -----------------------------------------------------
        # Calculate target grid cell
        # -----------------------------------------------------

        target_x = (
            self.grid_x + dx
        )

        target_y = (
            self.grid_y + dy
        )

        target_cell = (
            target_x,
            target_y
        )

        # -----------------------------------------------------
        # Safety check
        # -----------------------------------------------------

        if not self.environment.is_valid_cell(
            target_x,
            target_y
        ):

            # Something changed and target is no longer valid.
            # Put the car back at its current grid position.

            self.x, self.y = (
                self.environment.grid_to_pixel(
                    self.grid_x,
                    self.grid_y
                )
            )

            self.choose_direction()

            return

        # -----------------------------------------------------
        # Target pixel position
        # -----------------------------------------------------

        target_pixel_x, target_pixel_y = (
            self.environment.grid_to_pixel(
                target_x,
                target_y
            )
        )

        # -----------------------------------------------------
        # Detect arrival
        # -----------------------------------------------------

        reached_target = False

        if dx != 0:

            if abs(
                self.x - target_pixel_x
            ) <= self.SPEED:

                reached_target = True

        elif dy != 0:

            if abs(
                self.y - target_pixel_y
            ) <= self.SPEED:

                reached_target = True

        # -----------------------------------------------------
        # Arrived at next road cell
        # -----------------------------------------------------

        if reached_target:

            self.x = target_pixel_x
            self.y = target_pixel_y

            self.grid_x = target_x
            self.grid_y = target_y

            # Choose next connected road cell
            self.choose_direction()


    # =========================================================
    # GET GRID POSITION
    # =========================================================

    def get_grid_position(self):
        """
        Return the current grid position.
        """

        return (
            self.grid_x,
            self.grid_y
        )


    # =========================================================
    # GET PIXEL POSITION
    # =========================================================

    def get_pixel_position(self):
        """
        Return the current pixel position.
        """

        return (
            self.x,
            self.y
        )


    # =========================================================
    # DRAW
    # =========================================================

    def draw(self, screen):
        """
        Draw the traffic car.
        """

        car_rect = pygame.Rect(
            int(self.x + 10),
            int(self.y + 15),
            self.WIDTH,
            self.HEIGHT
        )

        pygame.draw.rect(
            screen,
            (30, 30, 30),
            car_rect
        )


# =============================================================
# TRAFFIC MANAGER
# =============================================================

class TrafficManager:
    """
    Manages multiple traffic cars.
    """

    # =========================================================
    # CONSTRUCTOR
    # =========================================================

    def __init__(
        self,
        environment,
        number_of_cars=10
    ):
        self.environment = environment

        self.cars = []

        # -----------------------------------------------------
        # Create traffic cars
        # -----------------------------------------------------

        for _ in range(number_of_cars):

            car = TrafficCar(
                environment
            )

            self.cars.append(
                car
            )


    # =========================================================
    # UPDATE
    # =========================================================

    def update(self):
        """
        Update every traffic car.
        """

        for car in self.cars:

            car.update()


    # =========================================================
    # DRAW
    # =========================================================

    def draw(self, screen):
        """
        Draw every traffic car.
        """

        for car in self.cars:

            car.draw(
                screen
            )


    # =========================================================
    # GET POSITIONS
    # =========================================================

    def get_positions(self):
        """
        Return the grid positions of all traffic cars.
        """

        return [
            car.get_grid_position()
            for car in self.cars
        ]


    # =========================================================
    # GET NEARBY CARS
    # =========================================================

    def get_nearby_cars(
        self,
        taxi_grid,
        radius=1
    ):
        """
        Find traffic cars near the taxi.

        Parameters
        ----------
        taxi_grid:
            Tuple containing:
                (taxi_x, taxi_y)

        radius:
            Manhattan-distance radius.

            radius=1 means:
                directly adjacent cells

            radius=2 means:
                up to two cells away

        Returns
        -------
        list
            List of nearby traffic-car grid positions.
        """

        taxi_x, taxi_y = taxi_grid

        nearby_cars = []

        for car in self.cars:

            car_x, car_y = (
                car.get_grid_position()
            )

            # Manhattan distance
            distance = (
                abs(car_x - taxi_x)
                +
                abs(car_y - taxi_y)
            )

            if distance <= radius:

                nearby_cars.append(
                    (car_x, car_y)
                )

        return nearby_cars


    # =========================================================
    # COUNT NEARBY CARS
    # =========================================================

    def count_nearby_cars(
        self,
        taxi_grid,
        radius=1
    ):
        """
        Return the number of traffic cars near the taxi.
        """

        return len(
            self.get_nearby_cars(
                taxi_grid,
                radius
            )
        )


    # =========================================================
    # IS TRAFFIC NEARBY?
    # =========================================================

    def is_traffic_nearby(
        self,
        taxi_grid,
        radius=1
    ):
        """
        Return True if at least one traffic car is
        within the specified radius.
        """

        return (
            self.count_nearby_cars(
                taxi_grid,
                radius
            )
            > 0
        )