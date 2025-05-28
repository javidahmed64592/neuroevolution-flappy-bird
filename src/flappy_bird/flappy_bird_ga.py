from __future__ import annotations

import numpy as np
from genetic_algorithm.ga import GeneticAlgorithm

from flappy_bird.objects.bird import Bird


class FlappyBirdGA(GeneticAlgorithm):
    """
    Genetic algorithm for Flappy Bird training.
    """

    def __init__(
        self,
        birds: list[Bird],
        mutation_rate: float,
        shift_vals: float,
    ) -> None:
        """
        Initialise FlappyBirdGA with a mutation rate.

        Parameters:
            birds (list[Bird]): Population of Birds
            mutation_rate (float): Population mutation rate
            shift_vals (float): Values to shift weights and biases by
        """
        super().__init__(birds, mutation_rate)
        self._lifetime: int
        self._shift_vals = shift_vals

    @property
    def num_alive(self) -> int:
        _alive_array = np.array([_bird._alive for _bird in self._population._population])
        return int(np.sum(_alive_array))

    @classmethod
    def create(
        cls,
        population_size: int,
        mutation_rate: float,
        lifetime: int,
        x: int,
        y: int,
        size: int,
        hidden_layer_sizes: list[int],
        weights_range: list[float],
        bias_range: list[float],
        shift_vals: float,
    ) -> FlappyBirdGA:
        """
        Create genetic algorithm and configure neural network.

        Parameters:
            population_size (int): Number of Birds in population
            mutation_rate (float): Mutation rate for Birds
            lifetime (int): Time of each generation in seconds
            x (int): x coordinate of Bird's start position
            y (int): y coordinate of Bird's start position
            size (int): Size of Bird
            hidden_layer_sizes (list[int]): Neural network hidden layer sizes
            weights_range (list[float]): Range for random weights
            bias_range (list[float]): Range for random bias
            shift_vals (float): Values to shift weights and biases by

        Returns:
            flappy_bird (FlappyBirdGA): Flappy Bird app
        """
        flappy_bird = cls(
            [Bird(x, y, size, hidden_layer_sizes, weights_range, bias_range) for _ in range(population_size)],
            mutation_rate,
            shift_vals,
        )
        flappy_bird._lifetime = lifetime
        return flappy_bird

    def reset(self) -> None:
        """
        Reset all Birds.
        """
        for _bird in self._population._population:
            _bird.reset()

    def mutate_birds(self) -> None:
        """
        Mutate all Birds.
        """
        for _bird in self._population._population:
            _bird._nn.mutate(self._shift_vals)
