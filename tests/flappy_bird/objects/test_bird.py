from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flappy_bird.objects.bird import Bird
from flappy_bird.objects.pipe import Pipe

MOCK_X = 100
MOCK_Y = 400
MOCK_X_LIM = 1000
MOCK_Y_LIM = 800
MOCK_SIZE = 50
MOCK_HIDDEN_LAYER_SIZES = [5, 5]
MOCK_WEIGHTS_RANGE = (-1.0, 1.0)
MOCK_BIAS_RANGE = (-1.0, 1.0)


@pytest.fixture
def bird() -> Bird:
    return Bird(
        MOCK_X,
        MOCK_Y,
        MOCK_X_LIM,
        MOCK_Y_LIM,
        MOCK_SIZE,
        MOCK_HIDDEN_LAYER_SIZES,
        MOCK_WEIGHTS_RANGE,
        MOCK_BIAS_RANGE,
    )


@pytest.fixture
def pipe() -> Pipe:
    pipe = Pipe(MOCK_X_LIM, MOCK_Y_LIM, 5)
    pipe._x = MOCK_X
    return pipe


@pytest.fixture
def mock_no_collision() -> Generator[MagicMock, None, None]:
    with patch("flappy_bird.objects.bird.Bird.rect_collision", return_value=False):
        yield


@pytest.fixture
def mock_collision() -> Generator[MagicMock, None, None]:
    with patch("flappy_bird.objects.bird.Bird.rect_collision", return_value=True):
        yield


@pytest.fixture
def mock_not_offscreen() -> Generator[MagicMock, None, None]:
    with patch("flappy_bird.objects.bird.Bird.offscreen", property(lambda self: False)):
        yield


@pytest.fixture
def mock_nn_jump() -> Generator[MagicMock, None, None]:
    with patch("neural_network.neural_network.NeuralNetwork.feedforward", return_value=np.array([0.0, 1.0])):
        yield


@pytest.fixture
def mock_nn_no_jump() -> Generator[MagicMock, None, None]:
    with patch("neural_network.neural_network.NeuralNetwork.feedforward", return_value=np.array([1.0, 0.0])):
        yield


class TestBird:
    def test_initialization(self, bird: Bird) -> None:
        assert bird._x == MOCK_X
        assert bird._y == MOCK_Y
        assert bird._x_lim == MOCK_X_LIM
        assert bird._y_lim == MOCK_Y_LIM
        assert bird._start_y == MOCK_Y
        assert bird._velocity == 0
        assert bird._size == MOCK_SIZE
        assert bird._closest_pipe is None
        assert bird._alive is True
        assert bird._hidden_layer_sizes == MOCK_HIDDEN_LAYER_SIZES
        assert bird._weights_range == MOCK_WEIGHTS_RANGE
        assert bird._bias_range == MOCK_BIAS_RANGE

    def test_nn_input(self, bird: Bird, pipe: Pipe) -> None:
        nn_input = bird.nn_input
        assert nn_input.shape == (5,)
        assert nn_input[0] == MOCK_Y / MOCK_Y_LIM
        assert nn_input[1] == bird.velocity / Bird.MIN_VELOCITY
        assert nn_input[2] == 0
        assert nn_input[3] == 0
        assert nn_input[4] == 0

        bird._closest_pipe = pipe
        nn_input = bird.nn_input
        assert nn_input[0] == MOCK_Y / MOCK_Y_LIM
        assert nn_input[1] == bird.velocity / Bird.MIN_VELOCITY
        assert nn_input[2] == pipe._top_height / MOCK_Y_LIM
        assert nn_input[3] == pipe._bottom_height / MOCK_Y_LIM
        assert nn_input[4] == pipe._x / MOCK_X_LIM

    def test_rect(self, bird: Bird) -> None:
        rect = bird.rect
        assert rect.x == MOCK_X
        assert rect.y == MOCK_Y
        assert rect.width == MOCK_SIZE
        assert rect.height == MOCK_SIZE

    def test_velocity(self, bird: Bird) -> None:
        assert bird.velocity == 0

        velocity_below_min = -20
        bird.velocity = velocity_below_min
        assert bird.velocity == Bird.MIN_VELOCITY

        velocity_above_min = 10
        bird.velocity = velocity_above_min
        assert bird.velocity == velocity_above_min

    def test_offscreen(self, bird: Bird) -> None:
        assert not bird.offscreen

        # Offscreen above
        bird._y = -1
        assert bird.offscreen

        # Offscreen below
        bird._y = MOCK_Y_LIM
        assert bird.offscreen

    def test_collide_with_closest_pipe_false(self, bird: Bird, pipe: Pipe, mock_no_collision: MagicMock) -> None:
        # Closest pipe but no collision
        bird._closest_pipe = pipe
        assert not bird.collide_with_closest_pipe

    def test_collide_with_closest_pipe_true(self, bird: Bird, pipe: Pipe, mock_collision: MagicMock) -> None:
        # Collision with pipe
        bird._closest_pipe = pipe
        assert bird.collide_with_closest_pipe

    def test_collide_with_closest_pipe_no_pipe(self, bird: Bird) -> None:
        bird._closest_pipe = None
        assert not bird.collide_with_closest_pipe

    def test_jump(self, bird: Bird) -> None:
        initial_velocity = 20
        bird.velocity = initial_velocity
        bird._jump()
        assert bird.velocity == initial_velocity + Bird.LIFT

    def test_move(self, bird: Bird) -> None:
        initial_y = bird._y
        initial_velocity = bird.velocity
        bird._move()
        assert bird.velocity == initial_velocity + Bird.GRAV
        assert bird._y == initial_y + bird.velocity

    def test_reset(self, bird: Bird) -> None:
        bird.velocity = -10
        bird._y = 100
        bird._score = 50
        bird._alive = False

        bird.reset()
        assert bird.velocity == 0
        assert bird._y == MOCK_Y
        assert bird._score == 0
        assert bird._alive is True

    def test_draw(self, bird: Bird) -> None:
        with patch("pygame.draw.rect") as mock_draw:
            mock_screen = patch("pygame.Surface").start()
            bird.draw(mock_screen)
            mock_draw.assert_called_once()

            # Test not drawing if bird is dead
            mock_draw.reset_mock()
            bird._alive = False
            bird.draw(mock_screen)
            mock_draw.assert_not_called()

    def test_update_alive(
        self,
        bird: Bird,
        pipe: Pipe,
        mock_no_collision: MagicMock,
        mock_not_offscreen: MagicMock,
        mock_nn_no_jump: MagicMock,
    ) -> None:
        initial_y = bird._y
        bird.update(pipe)
        assert bird._closest_pipe == pipe
        assert bird._y > initial_y
        assert bird._score == 1
        assert bird._alive is True

    def test_update_jump(
        self,
        bird: Bird,
        pipe: Pipe,
        mock_no_collision: MagicMock,
        mock_not_offscreen: MagicMock,
        mock_nn_jump: MagicMock,
    ) -> None:
        initial_y = bird._y
        bird.update(pipe)
        expected_velocity = max(Bird.LIFT, Bird.MIN_VELOCITY) + Bird.GRAV
        expected_y = initial_y + expected_velocity
        assert bird._y == expected_y
        assert bird._score == 1
        assert bird._alive is True

    def test_update_offscreen_death(self, bird: Bird, pipe: Pipe, mock_nn_no_jump: MagicMock) -> None:
        bird._y = -1
        bird.update(pipe)
        assert bird._alive is False
        assert bird._score == 0

    def test_update_collision_death(
        self, bird: Bird, pipe: Pipe, mock_collision: MagicMock, mock_nn_no_jump: MagicMock
    ) -> None:
        bird._closest_pipe = pipe
        bird.update(pipe)
        assert bird._alive is False
        assert bird._score == 0

    def test_update_dead_bird(self, bird: Bird, pipe: Pipe) -> None:
        bird._alive = False
        initial_y = bird._y
        initial_score = bird._score
        bird.update(pipe)
        assert bird._y == initial_y
        assert bird._score == initial_score
