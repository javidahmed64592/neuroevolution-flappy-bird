"""Genetic algorithm for Flappy Bird training."""

from __future__ import annotations

import numpy as np
from genetic_algorithm.ga import GeneticAlgorithm

from neuroevolution_flappy_bird.objects.bird import Bird


class FlappyBirdGA(GeneticAlgorithm):
    """Genetic algorithm for Flappy Bird training."""

    def __init__(
        self,
        birds: list[Bird],
        mutation_rate: float,
    ) -> None:
        """Initialise FlappyBirdGA with a mutation rate.

        :param list[Bird] birds: Population of Birds
        :param float mutation_rate: Population mutation rate
        """
        super().__init__(birds, mutation_rate)
        self._lifetime: int

    @property
    def num_alive(self) -> int:
        """Get number of alive Birds in population."""
        _alive_array = np.array([_bird._alive for _bird in self._population._members])
        return int(np.sum(_alive_array))

    @classmethod
    def create(
        cls,
        population_size: int,
        mutation_rate: float,
        lifetime: int,
        x: int,
        y: int,
        x_lim: int,
        y_lim: int,
        size: int,
        hidden_layer_sizes: list[int],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
    ) -> FlappyBirdGA:
        """Create genetic algorithm and configure neural network.

        :param int population_size: Number of Birds in population
        :param float mutation_rate: Mutation rate for Birds
        :param int lifetime: Time of each generation in seconds
        :param int x: x coordinate of Bird's start position
        :param int y: y coordinate of Bird's start position
        :param int x_lim: Screen width
        :param int y_lim: Screen height
        :param int size: Size of Bird
        :param list[int] hidden_layer_sizes: Neural network hidden layer sizes
        :param tuple[float, float] weights_range: Range for random weights
        :param tuple[float, float] bias_range: Range for random bias
        :return FlappyBirdGA: Flappy Bird app
        """
        flappy_bird = cls(
            [
                Bird(x, y, x_lim, y_lim, size, hidden_layer_sizes, weights_range, bias_range)
                for _ in range(population_size)
            ],
            mutation_rate,
        )
        flappy_bird._lifetime = lifetime
        return flappy_bird

    def reset(self) -> None:
        """Reset all Birds."""
        for _bird in self._population._members:
            _bird.reset()
