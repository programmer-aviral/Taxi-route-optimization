import pygame


class World:
    def __init__(self, width=1000, height=700):
        self.width = width
        self.height = height

        # One grid cell = 50 x 50 pixels
        self.cell_size = 50

        # ------------------------------------------
        # BUILDINGS
        # ------------------------------------------

        self.buildings = [
            pygame.Rect(50, 50, 180, 120),
            pygame.Rect(350, 50, 180, 120),
            pygame.Rect(650, 50, 180, 120),

            pygame.Rect(50, 300, 180, 120),
            pygame.Rect(350, 300, 180, 120),
            pygame.Rect(650, 300, 180, 120),

            pygame.Rect(50, 550, 180, 100),
            pygame.Rect(350, 550, 180, 100),
            pygame.Rect(650, 550, 180, 100),
        ]

        # ------------------------------------------
        # DESTINATION
        # ------------------------------------------

        self.destination = pygame.Rect(
            850,
            500,
            50,
            50
        )

    def is_valid_position(self, x, y, taxi_width=40, taxi_height=30):
        """
        Check whether the taxi can occupy this position.
        """

        taxi_rect = pygame.Rect(
            x,
            y,
            taxi_width,
            taxi_height
        )

        # Outside the world
        if taxi_rect.left < 0:
            return False

        if taxi_rect.right > self.width:
            return False

        if taxi_rect.top < 0:
            return False

        if taxi_rect.bottom > self.height:
            return False

        # Building collision
        for building in self.buildings:
            if taxi_rect.colliderect(building):
                return False

        return True

    def draw(self, screen):

        # Background
        screen.fill((210, 210, 210))

        # ------------------------------------------
        # HORIZONTAL ROADS
        # ------------------------------------------

        pygame.draw.rect(
            screen,
            (80, 80, 80),
            (0, 200, self.width, 100)
        )

        pygame.draw.rect(
            screen,
            (80, 80, 80),
            (0, 450, self.width, 100)
        )

        # ------------------------------------------
        # VERTICAL ROADS
        # ------------------------------------------

        pygame.draw.rect(
            screen,
            (80, 80, 80),
            (250, 0, 100, self.height)
        )

        pygame.draw.rect(
            screen,
            (80, 80, 80),
            (550, 0, 100, self.height)
        )

        pygame.draw.rect(
            screen,
            (80, 80, 80),
            (850, 0, 100, self.height)
        )

        # ------------------------------------------
        # BUILDINGS
        # ------------------------------------------

        for building in self.buildings:
            pygame.draw.rect(
                screen,
                (150, 150, 150),
                building
            )

        # ------------------------------------------
        # DESTINATION
        # ------------------------------------------

        pygame.draw.rect(
            screen,
            (0, 200, 0),
            self.destination
        )