from __future__ import annotations

import numpy as np
from genetic_algorithm.ga import Member
from neural_network.layer import HiddenLayer, InputLayer, Layer, OutputLayer
from neural_network.math.activation_functions import LinearActivation, ReluActivation
from neural_network.math.matrix import Matrix
from neural_network.neural_network import NeuralNetwork
from numpy.typing import NDArray

rng = np.random.default_rng()


class BirdMember(Member):
    """
    This class creates a Member for the genetic algorithm.

    The bird is assigned a neural network which acts as its brain and determines when the bird should 'jump'.
    This brain evolves via crossover and mutations.
    """

    def __init__(
        self,
        hidden_layer_sizes: list[int],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
    ) -> None:
        """
        Initialise BirdMember with a starting position, a width and a height.

        Parameters:
            hidden_layer_sizes (list[int]): Neural network hidden layer sizes
            weights_range (tuple[float, float]): Range for random weights
            bias_range (tuple[float, float]): Range for random biases
        """
        super().__init__()

        self._hidden_layer_sizes = hidden_layer_sizes
        self._weights_range = weights_range
        self._bias_range = bias_range
        self._nn = NeuralNetwork.from_layers(layers=self.nn_layers)
        self._score = 0

    @property
    def nn_layers(self) -> list[Layer]:
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

        return [input_layer, *hidden_layers, output_layer]

    @property
    def nn_input(self) -> NDArray:
        return np.zeros(5)

    @property
    def chromosome(self) -> tuple[list[Matrix], list[Matrix]]:
        return self._nn.weights, self._nn.bias

    @chromosome.setter
    def chromosome(self, new_chromosome: tuple[list[Matrix], list[Matrix]]) -> None:
        self._nn.weights = new_chromosome[0]
        self._nn.bias = new_chromosome[1]

    @property
    def fitness(self) -> int:
        return self._score**2

    @staticmethod
    def crossover_genes(
        element: float, other_element: float, roll: float, mutation_rate: float, random_range: tuple[float, float]
    ) -> float:
        if roll < mutation_rate:
            return rng.uniform(low=random_range[0], high=random_range[1])

        return float(rng.choice([element, other_element], p=[0.5, 0.5]))

    def crossover(self, parent_a: BirdMember, parent_b: BirdMember, mutation_rate: int) -> None:
        """
        Crossover the chromosomes of two birds to create a new chromosome.

        Parameters:
            parent_a (BirdMember): Used to construct new chromosome
            parent_b (BirdMember): Used to construct new chromosome
            mutation_rate (int): Probability for mutations to occur
        """

        def crossover_weights(element: float, other_element: float, roll: float) -> float:
            return BirdMember.crossover_genes(element, other_element, roll, mutation_rate, self._weights_range)

        def crossover_biases(element: float, other_element: float, roll: float) -> float:
            return BirdMember.crossover_genes(element, other_element, roll, mutation_rate, self._bias_range)

        self._new_chromosome = NeuralNetwork.crossover(
            parent_a._nn,
            parent_b._nn,
            crossover_weights,
            crossover_biases,
        )
