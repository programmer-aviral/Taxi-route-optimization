import pygame

from environment import TaxiEnvironment
from passenger import PassengerManager


# =========================================================
# SETTINGS
# =========================================================

WIDTH = 1000
HEIGHT = 700
FPS = 60


# =========================================================
# INITIALIZE PYGAME
# =========================================================

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Taxi Route Optimization - Manual Taxi Test"
)

clock = pygame.time.Clock()

font = pygame.font.SysFont(
    None,
    28
)


# =========================================================
# CREATE ENVIRONMENT
# =========================================================

env = TaxiEnvironment()

env.reset()


# =========================================================
# CREATE PASSENGER
# =========================================================

passenger_manager = PassengerManager(
    env
)

passenger_manager.create_passenger()


# =========================================================
# GAME STATE
# =========================================================

running = True

message = ""

passenger_picked_up = False
trip_completed = False


# =========================================================
# MAIN LOOP
# =========================================================

while running:

    # =====================================================
    # EVENTS
    # =====================================================

    for event in pygame.event.get():

        # -------------------------------------------------
        # CLOSE WINDOW
        # -------------------------------------------------

        if event.type == pygame.QUIT:

            running = False

        # -------------------------------------------------
        # KEY PRESSED
        # -------------------------------------------------

        elif event.type == pygame.KEYDOWN:

            action = None

            # ---------------------------------------------
            # Arrow keys
            # ---------------------------------------------

            if event.key == pygame.K_UP:

                action = TaxiEnvironment.UP

            elif event.key == pygame.K_DOWN:

                action = TaxiEnvironment.DOWN

            elif event.key == pygame.K_LEFT:

                action = TaxiEnvironment.LEFT

            elif event.key == pygame.K_RIGHT:

                action = TaxiEnvironment.RIGHT

            # ---------------------------------------------
            # WASD support
            # ---------------------------------------------

            elif event.key == pygame.K_w:

                action = TaxiEnvironment.UP

            elif event.key == pygame.K_s:

                action = TaxiEnvironment.DOWN

            elif event.key == pygame.K_a:

                action = TaxiEnvironment.LEFT

            elif event.key == pygame.K_d:

                action = TaxiEnvironment.RIGHT

            # ---------------------------------------------
            # Execute taxi movement
            # ---------------------------------------------

            if action is not None:

                old_position = env.taxi_grid

                next_state, reward, done = env.step(
                    action
                )

                new_position = env.taxi_grid

                print(
                    f"Taxi: {old_position} -> "
                    f"{new_position} | "
                    f"Action: {action} | "
                    f"Reward: {reward}"
                )


            # =================================================
            # PICKUP
            # =================================================

            if passenger_manager.pick_up(
                env.taxi_grid
            ):

                passenger_picked_up = True

                message = "PASSENGER PICKED UP!"

                print()
                print("=" * 45)
                print("PASSENGER PICKED UP")
                print("=" * 45)
                print(
                    "Passenger destination:",
                    passenger_manager.get_destination()
                )


            # =================================================
            # CHANGE GOAL AFTER PICKUP
            # =================================================

            if passenger_picked_up:

                passenger_destination = (
                    passenger_manager.get_destination()
                )

                destination_x, destination_y = (
                    env.grid_to_pixel(
                        passenger_destination[0],
                        passenger_destination[1]
                    )
                )

                env.world.destination = pygame.Rect(
                    destination_x,
                    destination_y,
                    env.CELL_SIZE,
                    env.CELL_SIZE
                )


            # =================================================
            # DROP-OFF
            # =================================================

            if passenger_manager.drop_off(
                env.taxi_grid
            ):

                trip_completed = True

                message = "TRIP COMPLETED!"

                print()
                print("=" * 45)
                print("PASSENGER DROPPED OFF")
                print("TRIP COMPLETED")
                print("=" * 45)


    # =====================================================
    # UPDATE TRAFFIC
    # =====================================================

    env.traffic.update()


    # =====================================================
    # DRAW BACKGROUND
    # =====================================================

    screen.fill(
        (220, 220, 220)
    )


    # =====================================================
    # DRAW WORLD
    # =====================================================

    env.world.draw(
        screen
    )


    # =====================================================
    # DRAW DESTINATION
    # =====================================================

    if env.world.destination is not None:

        pygame.draw.rect(
            screen,
            (0, 200, 0),
            env.world.destination
        )


    # =====================================================
    # DRAW PASSENGER
    # =====================================================

    passenger_manager.draw(
        screen
    )


    # =====================================================
    # DRAW TRAFFIC
    # =====================================================

    env.traffic.draw(
        screen
    )


    # =====================================================
    # DRAW TAXI
    # =====================================================

    env.taxi.draw(
        screen
    )


    # =====================================================
    # STATUS
    # =====================================================

    status = (
        f"Passenger: "
        f"{passenger_manager.get_status()}"
    )

    status_text = font.render(
        status,
        True,
        (20, 20, 20)
    )

    screen.blit(
        status_text,
        (20, 20)
    )


    # =====================================================
    # SHOW MESSAGE
    # =====================================================

    if message:

        message_text = font.render(
            message,
            True,
            (20, 20, 20)
        )

        screen.blit(
            message_text,
            (20, 55)
        )


    # =====================================================
    # SHOW CONTROLS
    # =====================================================

    controls = font.render(
        "Use Arrow Keys or W A S D",
        True,
        (20, 20, 20)
    )

    screen.blit(
        controls,
        (20, 90)
    )


    # =====================================================
    # UPDATE DISPLAY
    # =====================================================

    pygame.display.flip()

    clock.tick(FPS)


# =========================================================
# EXIT
# =========================================================

pygame.quit()