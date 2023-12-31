from typing import Any, Dict

import numpy as np
import pygame


class Pipe:
    """
    This class creates a top pipe and bottom pipe pair which has a width, a spacing between the two
    pipes, and a speed at which to travel across the screen.

    The pipe is drawn to the display in the draw() method. The update() method moves the pipe along
    the screen.

    The pipes have an offscreen property which indicates whether or not the pipes have moved off
    the screen and get destroyed if they have since they are no longer needed. The
    get_closest_pipe() method takes a list of pipe pairs and determines which pipe pair is both
    closest to and in front of the birds on the screen.
    """

    def __init__(self, width: float = 50, spacing: float = 200, speed: float = 3.5):
        """
        Initialise a pipe pair with a width, spacing and speed.

        Parameters:
            width (float): Width of the top and bottom pipes
            spacing (float): Space between top and bottom pipe
            speed (float): Speed at which the pipes move across the screen
        """
        self.width = width
        self.spacing = spacing
        self.speed = speed
        self.screen = pygame.display.get_surface()
        self.color = (0, 255, 0)
        self.x = float(self.screen.get_size()[0])
        self.top = np.random.uniform(
            (1 / 5) * (self.screen.get_size()[1] - spacing),
            (4 / 5) * (self.screen.get_size()[1] - spacing),
        )
        self.bottom = self.screen.get_size()[1] - (self.top + spacing)

    @classmethod
    def create(cls, config_pipe: Dict[str, Any], speed: float) -> "Pipe":
        """
        Create a pipe from config file and speed.

        Parameters:
            config_pipe (Dict(str, Any)): Pipe configuration
            speed (float): Speed at which pipes move across the screen

        Returns:
            (Pipe): Configured pipe
        """
        pipe = cls(config_pipe["width"], config_pipe["spacing"], speed)
        pipe.rect_top = pygame.Rect(pipe.x, 0, config_pipe["width"], pipe.top)
        pipe.rect_bot = pygame.Rect(pipe.x, pipe.top + config_pipe["spacing"], config_pipe["width"], pipe.bottom)
        return pipe

    def draw(self) -> None:
        """
        Draw the pipes on the display.
        """
        pygame.draw.rect(self.screen, self.color, self.rect_top)
        pygame.draw.rect(self.screen, self.color, self.rect_bot)

    def update(self) -> None:
        """
        Move the pipes along the screen.
        """
        self.x -= self.speed

        self.rect_top = pygame.Rect(self.x, 0, self.width, self.top)
        self.rect_bot = pygame.Rect(self.x, self.top + self.spacing, self.width, self.bottom)

        self.draw()

    @property
    def offscreen(self) -> bool:
        """
        Return whether or not the pipes have moved off the screen.

        Returns:
            (bool): Is pipe off screen?
        """
        return self.x < -self.width
