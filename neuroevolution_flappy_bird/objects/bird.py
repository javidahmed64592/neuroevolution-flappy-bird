"""Bird object for Flappy Bird game."""

from __future__ import annotations

import numpy as np
import pygame
from numpy.typing import NDArray

from neuroevolution_flappy_bird.ga.bird_member import BirdMember
from neuroevolution_flappy_bird.objects.pipe import Pipe

rng = np.random.default_rng()


class Bird(BirdMember):
    """This class creates a Bird object which has a starting x and y position and a size.

    The Bird is drawn to the display in the draw() method. The update() method performs physics calculations and updates
    the Bird's position, velocity, and alive state accordingly. The Bird dies if it collides with a pipe.

    The Bird is assigned a neural network which acts as its brain and determines when the Bird should 'jump' based on
    its current position and the position of the nearest pipe. This brain evolves via crossover and mutations. Its
    fitness value is the square of its score which is incremented by 1 each time the update() method is called.
    """

    GRAV = 1
    LIFT = -25
    MIN_VELOCITY = -15

    def __init__(
        self,
        x: int,
        y: int,
        x_lim: int,
        y_lim: int,
        size: int,
        hidden_layer_sizes: list[int],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
    ) -> None:
        """Initialise Bird with a starting position, a width and a height.

        :param int x: x coordinate of Bird's start position
        :param int y: y coordinate of Bird's start position
        :param int x_lim: Screen width
        :param int y_lim: Screen height
        :param int size: Size of Bird
        :param list[int] hidden_layer_sizes: Neural network hidden layer sizes
        :param tuple[float, float] weights_range: Range for random weights
        :param tuple[float, float] bias_range: Range for random biases
        """
        self._x = x
        self._y = y
        self._x_lim = x_lim
        self._y_lim = y_lim
        self._start_y = y
        self._velocity = 0
        self._size = size
        self._closest_pipe: Pipe | None = None

        self._alive = True
        self._colour = rng.integers(low=0, high=256, size=3)
        super().__init__(hidden_layer_sizes, weights_range, bias_range)

    @property
    def nn_input(self) -> NDArray:
        """Get neural network input for Bird."""
        _nn_input = np.array([self._y / self._y_lim, self.velocity / self.MIN_VELOCITY, 0, 0, 0])
        if self._closest_pipe:
            _nn_input[2] = self._closest_pipe._top_height / self._y_lim
            _nn_input[3] = self._closest_pipe._bottom_height / self._y_lim
            _nn_input[4] = self._closest_pipe._x / self._x_lim
        return _nn_input

    @property
    def rect(self) -> pygame.Rect:
        """Get Bird's rectangle for collision detection."""
        return pygame.Rect(self._x, self._y, self._size, self._size)

    @property
    def velocity(self) -> int:
        """Get Bird's velocity."""
        return self._velocity

    @velocity.setter
    def velocity(self, new_velocity: int) -> None:
        """Set Bird's velocity, ensuring it does not go below MIN_VELOCITY."""
        self._velocity = max(new_velocity, self.MIN_VELOCITY)

    @property
    def offscreen(self) -> bool:
        """Check if Bird is offscreen."""
        return (0 > self._y) or (self._y + self._size > self._y_lim)

    @property
    def collide_with_closest_pipe(self) -> bool:
        """Check if Bird is colliding with closest Pipe.

        :return bool: True if Bird is colliding with closest Pipe
        """
        if not self._closest_pipe:
            return False
        return Bird.rect_collision(self.rect, self._closest_pipe.rects)

    @staticmethod
    def rect_collision(bird_rect: pygame.Rect, pipe_rects: list[pygame.Rect]) -> bool:
        """Check if Bird collides with any Pipe.

        :param Rect bird_rect: Bird's rectangle
        :param list[Rect] pipe_rects: List of Pipe rectangles
        :return bool: True if Bird is colliding with any Pipe
        """
        return any(bird_rect.colliderect(pipe_rect) for pipe_rect in pipe_rects)

    def _jump(self) -> None:
        """Make Bird 'jump' by accelerating upwards."""
        self.velocity += self.LIFT

    def _move(self) -> None:
        """Update Bird's position and velocity."""
        self.velocity += self.GRAV
        self._y += self.velocity

    def reset(self) -> None:
        """Reset to start positions."""
        self.velocity = 0
        self._y = self._start_y
        self._score = 0
        self._alive = True

    def draw(self, screen: pygame.Surface) -> None:
        """Draw Bird on the display.

        :param Surface screen: Screen to draw Bird to
        """
        if not self._alive:
            return
        pygame.draw.rect(screen, self._colour.tolist(), self.rect)

    def update(self, closest_pipe: Pipe) -> None:
        """Use neural network to determine whether or not Bird should jump, and kill if it collides with a Pipe.

        :param Pipe closest_pipe: Pipe closest to Bird
        """
        if not self._alive:
            return

        self._closest_pipe = closest_pipe
        output = self._nn.feedforward(self.nn_input)

        if output[0] < output[1]:
            self._jump()

        self._move()

        if self.offscreen or self.collide_with_closest_pipe:
            self._alive = False
            return

        self._score += 1
