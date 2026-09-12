import random

import pygame


class Passenger:
    """
    Represents one passenger in the taxi environment.

    Passenger states:

        waiting  -> passenger is waiting for taxi
        onboard  -> passenger is inside taxi
        completed -> passenger reached destination
    """

    # =========================================================
    # PASSENGER SETTINGS
    # =========================================================

    WIDTH = 18
    HEIGHT = 18

    # =========================================================
    # CONSTRUCTOR
    # =========================================================

    def __init__(
        self,
        environment,
        start_cell=None,
        destination_cell=None
    ):
        self.environment = environment

        # -----------------------------------------------------
        # Choose passenger starting cell
        # -----------------------------------------------------

        if start_cell is None:

            if not self.environment.valid_cells:
                raise ValueError(
                    "No valid road cells are available."
                )

            start_cell = random.choice(
                self.environment.valid_cells
            )

        # -----------------------------------------------------
        # Choose passenger destination
        # -----------------------------------------------------

        if destination_cell is None:

            while True:

                destination_cell = random.choice(
                    self.environment.valid_cells
                )

                if destination_cell != start_cell:

                    distance = (
                        self.environment.get_distance(
                            start_cell,
                            destination_cell
                        )
                    )

                    if distance != float("inf"):
                        break

        # -----------------------------------------------------
        # Save grid positions
        # -----------------------------------------------------

        self.start_cell = start_cell

        self.destination_cell = destination_cell

        self.grid_x = start_cell[0]
        self.grid_y = start_cell[1]

        # -----------------------------------------------------
        # Passenger status
        # -----------------------------------------------------

        self.status = "waiting"

        # -----------------------------------------------------
        # Convert passenger position to pixels
        # -----------------------------------------------------

        self.x, self.y = (
            self.environment.grid_to_pixel(
                self.grid_x,
                self.grid_y
            )
        )

    # =========================================================
    # GET CURRENT POSITION
    # =========================================================

    def get_grid_position(self):
        """
        Return the passenger's current grid position.
        """

        return (
            self.grid_x,
            self.grid_y
        )

    # =========================================================
    # GET DESTINATION
    # =========================================================

    def get_destination(self):
        """
        Return the passenger destination cell.
        """

        return self.destination_cell

    # =========================================================
    # CHECK WHETHER TAXI CAN PICK UP PASSENGER
    # =========================================================

    def can_pick_up(self, taxi_grid):
        """
        Return True when the taxi is at the passenger's
        waiting location.
        """

        return (
            self.status == "waiting"
            and tuple(taxi_grid)
            == tuple(self.start_cell)
        )

    # =========================================================
    # PICK UP PASSENGER
    # =========================================================

    def pick_up(self):
        """
        Put passenger inside the taxi.
        """

        if self.status == "waiting":

            self.status = "onboard"

            return True

        return False

    # =========================================================
    # CHECK WHETHER PASSENGER CAN BE DROPPED OFF
    # =========================================================

    def can_drop_off(self, taxi_grid):
        """
        Return True when:

            1. Passenger is onboard.
            2. Taxi has reached passenger destination.
        """

        return (
            self.status == "onboard"
            and tuple(taxi_grid)
            == tuple(self.destination_cell)
        )

    # =========================================================
    # DROP OFF PASSENGER
    # =========================================================

    def drop_off(self):
        """
        Complete the passenger trip.
        """

        if self.status == "onboard":

            self.status = "completed"

            return True

        return False

    # =========================================================
    # IS WAITING?
    # =========================================================

    def is_waiting(self):
        """
        Return True if passenger is waiting.
        """

        return self.status == "waiting"

    # =========================================================
    # IS ONBOARD?
    # =========================================================

    def is_onboard(self):
        """
        Return True if passenger is inside the taxi.
        """

        return self.status == "onboard"

    # =========================================================
    # IS COMPLETED?
    # =========================================================

    def is_completed(self):
        """
        Return True if passenger has completed the trip.
        """

        return self.status == "completed"

    # =========================================================
    # DRAW
    # =========================================================

    def draw(self, screen):
        """
        Draw the passenger when waiting.

        Once the passenger is onboard or completed,
        the passenger is not drawn at the waiting location.
        """

        if self.status != "waiting":
            return

        center_x = int(
            self.x + self.environment.CELL_SIZE // 2
        )

        center_y = int(
            self.y + self.environment.CELL_SIZE // 2
        )

        # Draw passenger body
        pygame.draw.circle(
            screen,
            (40, 80, 200),
            (
                center_x,
                center_y
            ),
            8
        )

        # Draw passenger head
        pygame.draw.circle(
            screen,
            (255, 220, 170),
            (
                center_x,
                center_y - 10
            ),
            5
        )


# =============================================================
# PASSENGER MANAGER
# =============================================================

class PassengerManager:
    """
    Manages passengers in the environment.
    """

    def __init__(
        self,
        environment
    ):
        self.environment = environment

        self.passenger = None

    # =========================================================
    # CREATE PASSENGER
    # =========================================================

    def create_passenger(self):
        """
        Create a new passenger with:

            random start
            random destination
        """

        self.passenger = Passenger(
            self.environment
        )

        return self.passenger

    # =========================================================
    # GET PASSENGER
    # =========================================================

    def get_passenger(self):
        """
        Return current passenger.
        """

        return self.passenger

    # =========================================================
    # GET PASSENGER POSITION
    # =========================================================

    def get_position(self):
        """
        Return passenger position.
        """

        if self.passenger is None:
            return None

        return self.passenger.get_grid_position()

    # =========================================================
    # GET PASSENGER DESTINATION
    # =========================================================

    def get_destination(self):
        """
        Return passenger's destination.
        """

        if self.passenger is None:
            return None

        return self.passenger.get_destination()

    # =========================================================
    # PICK UP
    # =========================================================

    def pick_up(self, taxi_grid):
        """
        Attempt to pick up the passenger.
        """

        if self.passenger is None:
            return False

        if self.passenger.can_pick_up(
            taxi_grid
        ):

            return self.passenger.pick_up()

        return False

    # =========================================================
    # DROP OFF
    # =========================================================

    def drop_off(self, taxi_grid):
        """
        Attempt to drop off passenger.
        """

        if self.passenger is None:
            return False

        if self.passenger.can_drop_off(
            taxi_grid
        ):

            return self.passenger.drop_off()

        return False

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self):
        """
        Return passenger status.
        """

        if self.passenger is None:
            return "none"

        return self.passenger.status

    # =========================================================
    # DRAW
    # =========================================================

    def draw(self, screen):
        """
        Draw the passenger.
        """

        if self.passenger is not None:

            self.passenger.draw(
                screen
            )