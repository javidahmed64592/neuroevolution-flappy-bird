"""Unit tests for the neuroevolution_flappy_bird.ga.bird_member module."""

import numpy as np
import pytest
from neural_network.neural_network import NeuralNetwork

from neuroevolution_flappy_bird.ga.bird_member import BirdMember

rng = np.random.default_rng()

MOCK_NUM_HIDDEN_LAYERS = 2
MOCK_HIDDEN_LAYER_SIZE = 5
MOCK_WEIGHTS_RANGE = (-1.0, 1.0)
MOCK_BIAS_RANGE = (-1.0, 1.0)

NUM_INPUTS = 5
NUM_OUTPUTS = 2
MOCK_MUTATION_RATE = 0.2


@pytest.fixture
def bird_member_a() -> BirdMember:
    """Mock BirdMember instance."""
    return BirdMember(
        hidden_layer_sizes=[MOCK_HIDDEN_LAYER_SIZE] * MOCK_NUM_HIDDEN_LAYERS,
        weights_range=MOCK_WEIGHTS_RANGE,
        bias_range=MOCK_BIAS_RANGE,
    )


@pytest.fixture
def bird_member_b() -> BirdMember:
    """Mock BirdMember instance."""
    return BirdMember(
        hidden_layer_sizes=[MOCK_HIDDEN_LAYER_SIZE] * MOCK_NUM_HIDDEN_LAYERS,
        weights_range=MOCK_WEIGHTS_RANGE,
        bias_range=MOCK_BIAS_RANGE,
    )


@pytest.fixture
def bird_member_c() -> BirdMember:
    """Mock BirdMember instance."""
    return BirdMember(
        hidden_layer_sizes=[MOCK_HIDDEN_LAYER_SIZE] * MOCK_NUM_HIDDEN_LAYERS,
        weights_range=MOCK_WEIGHTS_RANGE,
        bias_range=MOCK_BIAS_RANGE,
    )


class TestBirdMember:
    """Unit tests for the BirdMember class."""

    def test_initialization(self, bird_member_a: BirdMember) -> None:
        """Test BirdMember initialization."""
        assert bird_member_a._hidden_layer_sizes == [MOCK_HIDDEN_LAYER_SIZE] * MOCK_NUM_HIDDEN_LAYERS
        assert bird_member_a._weights_range == MOCK_WEIGHTS_RANGE
        assert bird_member_a._bias_range == MOCK_BIAS_RANGE
        assert isinstance(bird_member_a._nn, NeuralNetwork)
        assert bird_member_a._score == 0

    def test_nn_layers(self, bird_member_a: BirdMember) -> None:
        """Test nn_layers property."""
        assert len(bird_member_a.nn_layers) == MOCK_NUM_HIDDEN_LAYERS + 2
        assert bird_member_a.nn_layers[0].size == NUM_INPUTS
        assert bird_member_a.nn_layers[1].size == MOCK_HIDDEN_LAYER_SIZE
        assert bird_member_a.nn_layers[2].size == MOCK_HIDDEN_LAYER_SIZE
        assert bird_member_a.nn_layers[3].size == NUM_OUTPUTS

    def test_nn_input(self, bird_member_a: BirdMember) -> None:
        """Test nn_input property."""
        assert np.all(bird_member_a.nn_input == np.zeros(NUM_INPUTS))

    def test_chromosome(self, bird_member_a: BirdMember) -> None:
        """Test chromosome property."""
        weights, bias = bird_member_a.chromosome
        assert len(weights) == MOCK_NUM_HIDDEN_LAYERS + 2
        assert len(bias) == MOCK_NUM_HIDDEN_LAYERS + 2

    def test_chromosome_setter(self, bird_member_a: BirdMember, bird_member_b: BirdMember) -> None:
        """Test chromosome setter."""
        bird_member_a.chromosome = bird_member_b.chromosome

        for weight_a, weight_b in zip(bird_member_a._nn.weights, bird_member_b._nn.weights, strict=False):
            assert np.array_equal(weight_a.vals, weight_b.vals)

        for bias_a, bias_b in zip(bird_member_a._nn.bias, bird_member_b._nn.bias, strict=False):
            assert np.array_equal(bias_a.vals, bias_b.vals)

    def test_fitness(self, bird_member_a: BirdMember) -> None:
        """Test fitness property."""
        bird_member_a._score = 10
        assert bird_member_a.fitness == bird_member_a._score**2

    def test_crossover_genes(self) -> None:
        """Test crossover_genes static method."""
        element = 0.5
        other_element = 0.8

        roll = 0.6
        result = BirdMember.crossover_genes(element, other_element, roll, MOCK_MUTATION_RATE, MOCK_WEIGHTS_RANGE)
        assert MOCK_WEIGHTS_RANGE[0] <= result <= MOCK_WEIGHTS_RANGE[1]
        assert result in (element, other_element)

        roll = 0.1
        result = BirdMember.crossover_genes(element, other_element, roll, MOCK_MUTATION_RATE, MOCK_WEIGHTS_RANGE)
        assert MOCK_WEIGHTS_RANGE[0] <= result <= MOCK_WEIGHTS_RANGE[1]
        assert result not in (element, other_element)

    def test_crossover(self, bird_member_a: BirdMember, bird_member_b: BirdMember, bird_member_c: BirdMember) -> None:
        """Test crossover method."""
        bird_member_c.crossover(bird_member_a, bird_member_b, MOCK_MUTATION_RATE)

        assert len(bird_member_c._nn.weights) == len(bird_member_a._nn.weights)
        assert len(bird_member_c._nn.bias) == len(bird_member_a._nn.bias)
        assert bird_member_c._hidden_layer_sizes == bird_member_a._hidden_layer_sizes
        assert bird_member_c._weights_range == bird_member_a._weights_range
        assert bird_member_c._bias_range == bird_member_a._bias_range
