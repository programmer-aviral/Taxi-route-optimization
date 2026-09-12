
import pygame


class Taxi:
    def __init__(self, x=250, y=200):
        # Position in PIXELS.
        self.x = x
        self.y = y

        # Taxi appearance.
        self.width = 40
        self.height = 30

        # One action = one grid cell = 50 pixels.
        self.speed = 50

        # Pygame rectangle used for drawing and collision.
        self.rect = pygame.Rect(
            self.x,
            self.y,
            self.width,
            self.height
        )

    def sync_rect(self):
        """
        Keep the Pygame rectangle synchronized
        with the taxi's x/y position.
        """
        self.rect.x = self.x
        self.rect.y = self.y

    def sync_position(self):
        """
        Keep x/y synchronized with the Pygame rectangle.
        """
        self.x = self.rect.x
        self.y = self.rect.y

    def set_position(self, x, y):
        """
        Set taxi position directly and update the rectangle.
        """
        self.x = x
        self.y = y
        self.sync_rect()

    def move(
        self,
        dx,
        dy,
        buildings,
        screen_width,
        screen_height
    ):
        # Make sure rect matches the current position
        # before attempting movement.
        self.sync_rect()

        # Calculate new position.
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed

        # Create a rectangle at the new position.
        new_rect = pygame.Rect(
            new_x,
            new_y,
            self.width,
            self.height
        )

        # Check collision with buildings.
        collision = any(
            new_rect.colliderect(building)
            for building in buildings
        )

        if not collision:
            # Move both coordinate systems together.
            self.x = new_x
            self.y = new_y
            self.rect.x = new_x
            self.rect.y = new_y

        # Keep taxi inside screen.
        self.x = max(
            0,
            min(
                self.x,
                screen_width - self.width
            )
        )

        self.y = max(
            0,
            min(
                self.y,
                screen_height - self.height
            )
        )

        # Final synchronization.
        self.sync_rect()

    def reached_destination(self, destination):
        return self.rect.colliderect(destination)

    def draw(self, screen):
        pygame.draw.rect(
            screen,
            (255, 200, 0),
            self.rect
        )
