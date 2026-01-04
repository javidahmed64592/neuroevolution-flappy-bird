"""Unit tests for the neuroevolution_flappy_bird.ga.bird_ga module."""

from unittest.mock import MagicMock, patch

import pytest

from neuroevolution_flappy_bird.ga.bird_ga import FlappyBirdGA
from neuroevolution_flappy_bird.objects.bird import Bird

MOCK_POPULATION_SIZE = 5
MOCK_MUTATION_RATE = 0.1
MOCK_LIFETIME = 100
MOCK_X = 100
MOCK_Y = 400
MOCK_X_LIM = 1000
MOCK_Y_LIM = 800
MOCK_SIZE = 50
MOCK_HIDDEN_LAYER_SIZES = [5, 5]
MOCK_WEIGHTS_RANGE = (-1.0, 1.0)
MOCK_BIAS_RANGE = (-1.0, 1.0)
MOCK_NUM_ALIVE = 3


@pytest.fixture
def mock_birds() -> list[MagicMock]:
    """Mock list of Bird instances."""
    birds = []
    for i in range(MOCK_POPULATION_SIZE):
        bird = MagicMock(spec=Bird)
        bird._alive = i < MOCK_NUM_ALIVE
        birds.append(bird)
    return birds


@pytest.fixture
def bird_ga(mock_birds: list[MagicMock]) -> FlappyBirdGA:
    """Mock FlappyBirdGA instance."""
    ga = FlappyBirdGA(mock_birds, MOCK_MUTATION_RATE)
    ga._lifetime = MOCK_LIFETIME
    return ga


class TestFlappyBirdGA:
    """Unit tests for the FlappyBirdGA class."""

    def test_initialization(self, bird_ga: FlappyBirdGA, mock_birds: list[MagicMock]) -> None:
        """Test FlappyBirdGA initialization."""
        assert all(bird_ga._population._members == mock_birds)
        assert bird_ga._mutation_rate == MOCK_MUTATION_RATE
        assert bird_ga._lifetime == MOCK_LIFETIME

    def test_num_alive(self, bird_ga: FlappyBirdGA) -> None:
        """Test num_alive property."""
        assert bird_ga.num_alive == MOCK_NUM_ALIVE

        # Test when all birds are alive
        for bird in bird_ga._population._members:
            bird._alive = True
        assert bird_ga.num_alive == MOCK_POPULATION_SIZE

        # Test when all birds are dead
        for bird in bird_ga._population._members:
            bird._alive = False
        assert bird_ga.num_alive == 0

    @patch("neuroevolution_flappy_bird.ga.bird_ga.Bird")
    def test_create(self, mock_bird_class: MagicMock, mock_birds: list[MagicMock]) -> None:
        """Test FlappyBirdGA.create class method."""
        mock_bird_class.side_effect = mock_birds

        ga = FlappyBirdGA.create(
            MOCK_POPULATION_SIZE,
            MOCK_MUTATION_RATE,
            MOCK_LIFETIME,
            MOCK_X,
            MOCK_Y,
            MOCK_X_LIM,
            MOCK_Y_LIM,
            MOCK_SIZE,
            MOCK_HIDDEN_LAYER_SIZES,
            MOCK_WEIGHTS_RANGE,
            MOCK_BIAS_RANGE,
        )

        # Verify GA was created properly
        assert len(ga._population._members) == MOCK_POPULATION_SIZE
        assert all(ga._population._members == mock_birds)
        assert ga._mutation_rate == MOCK_MUTATION_RATE
        assert ga._lifetime == MOCK_LIFETIME

        # Verify Bird constructor was called correctly
        assert mock_bird_class.call_count == MOCK_POPULATION_SIZE
        for call in mock_bird_class.call_args_list:
            args, _ = call
            assert args == (
                MOCK_X,
                MOCK_Y,
                MOCK_X_LIM,
                MOCK_Y_LIM,
                MOCK_SIZE,
                MOCK_HIDDEN_LAYER_SIZES,
                MOCK_WEIGHTS_RANGE,
                MOCK_BIAS_RANGE,
            )

    def test_reset(self, bird_ga: FlappyBirdGA, mock_birds: list[MagicMock]) -> None:
        """Test reset method."""
        bird_ga.reset()

        # Verify reset was called on each bird
        for bird in mock_birds:
            bird.reset.assert_called_once()
