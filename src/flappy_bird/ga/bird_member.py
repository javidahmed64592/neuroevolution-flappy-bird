from __future__ import annotations

import numpy as np
from genetic_algorithm.ga import Member
from neural_network.layer import HiddenLayer, InputLayer, OutputLayer
from neural_network.math.activation_functions import LinearActivation, ReluActivation
from neural_network.math.matrix import Matrix
from neural_network.neural_network import NeuralNetwork
from numpy.typing import NDArray

rng = np.random.default_rng()


class BirdMember(Member):
    """
    TODO: UPDATE THIS DOCSTRING
    This class creates a Bird object which has a starting x and y position and a size.

    The Bird is drawn to the display in the draw() method. The update() method performs physics calculations and updates
    the Bird's position, velocity, and alive state accordingly. The Bird dies if it collides with a pipe.

    The Bird is assigned a neural network which acts as its brain and determines when the Bird should 'jump' based on
    its current position and the position of the nearest pipe. This brain evolves via crossover and mutations. Its
    fitness value is the square of its score which is incremented by 1 each time the update() method is called.
    """

    def __init__(
        self,
        hidden_layer_sizes: list[int],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
    ) -> None:
        """
        Initialise Bird with a starting position, a width and a height.

        Parameters:
            x (int): x coordinate of Bird's start position
            y (int): y coordinate of Bird's start position
            size (int): Size of Bird
            hidden_layer_sizes (list[int]): Neural network hidden layer sizes
            weights_range (tuple[float, float]): Range for random weights
            bias_range (tuple[float, float]): Range for random biases
        """
        super().__init__()

        self._hidden_layer_sizes = hidden_layer_sizes
        self._weights_range = weights_range
        self._bias_range = bias_range
        self._nn: NeuralNetwork = None
        self._score = 0

    @property
    def neural_network(self) -> NeuralNetwork:
        if not self._nn:
            input_layer = InputLayer(size=len(self.nn_input), activation=LinearActivation)
            hidden_layers = [
                HiddenLayer(
                    size=size, activation=ReluActivation, weights_range=self._weights_range, bias_range=self._bias_range
                )
                for size in self._hidden_layer_sizes
            ]
            output_layer = OutputLayer(
                size=2, activation=LinearActivation, weights_range=self._weights_range, bias_range=self._bias_range
            )

            self._nn = NeuralNetwork.from_layers(layers=[input_layer, *hidden_layers, output_layer])

        return self._nn

    @property
    def nn_input(self) -> NDArray:
        return np.zeros(5)

    @property
    def chromosome(self) -> list[list[Matrix]]:
        return [self.neural_network.weights, self.neural_network.bias]

    @chromosome.setter
    def chromosome(self, new_chromosome: list[list[Matrix]]) -> None:
        self.neural_network.weights = new_chromosome[0]
        self.neural_network.bias = new_chromosome[1]

    @property
    def fitness(self) -> int:
        return self._score**2

    def crossover(self, parent_a: BirdMember, parent_b: BirdMember, mutation_rate: int) -> None:
        """
        Crossover the chromosomes of two Birds to create a new chromosome.

        Parameters:
            parent_a (BirdMember): Used to construct new chromosome
            parent_b (BirdMember): Used to construct new chromosome
            mutation_rate (int): Probability for mutations to occur
        """

        def crossover_genes(
            element: float, other_element: float, roll: float, random_range: tuple[float, float]
        ) -> float:
            if roll < mutation_rate:
                return rng.uniform(low=random_range[0], high=random_range[1])

            return float(rng.choice([element, other_element], p=[0.5, 0.5]))

        def crossover_weights(element: float, other_element: float, roll: float) -> float:
            return crossover_genes(element, other_element, roll, self._weights_range)

        def crossover_biases(element: float, other_element: float, roll: float) -> float:
            return crossover_genes(element, other_element, roll, self._bias_range)

        self._new_chromosome = NeuralNetwork.crossover(
            parent_a.neural_network,
            parent_b.neural_network,
            crossover_weights,
            crossover_biases,
        )
