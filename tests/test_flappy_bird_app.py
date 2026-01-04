"""Unit tests for the neuroevolution_flappy_bird.flappy_bird_app module."""

from collections.abc import Generator
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from neuroevolution_flappy_bird.flappy_bird_app import FlappyBirdApp

MOCK_NAME = "Flappy Bird"
MOCK_WIDTH = 800
MOCK_HEIGHT = 600
MOCK_FPS = 60
MOCK_FONT = "Arial"
MOCK_FONT_SIZE = 20
MOCK_POPULATION_SIZE = 10
MOCK_MUTATION_RATE = 0.1
MOCK_LIFETIME = 30
MOCK_BIRD_X = 100
MOCK_BIRD_Y = 300
MOCK_BIRD_SIZE = 20
MOCK_HIDDEN_LAYER_SIZES = [4, 4]
MOCK_WEIGHTS_RANGE = (-1.0, 1.0)
MOCK_BIAS_RANGE = (-1.0, 1.0)


@pytest.fixture
def app() -> FlappyBirdApp:
    """Mock FlappyBirdApp instance."""
    return FlappyBirdApp(
        name=MOCK_NAME,
        width=MOCK_WIDTH,
        height=MOCK_HEIGHT,
        fps=MOCK_FPS,
        font=MOCK_FONT,
        font_size=MOCK_FONT_SIZE,
    )


@pytest.fixture
def mock_flappy_bird_ga() -> Generator[MagicMock]:
    """Mock FlappyBirdGA class."""
    with patch("neuroevolution_flappy_bird.flappy_bird_app.FlappyBirdGA") as mock:
        yield mock


@pytest.fixture
def mock_pipe() -> Generator[MagicMock]:
    """Mock Pipe class."""
    with patch("neuroevolution_flappy_bird.flappy_bird_app.Pipe") as mock:
        yield mock


@pytest.fixture
def mock_display_set_mode() -> Generator[MagicMock]:
    """Mock pygame.display.set_mode function."""
    with patch("pygame.display.set_mode", return_value=MagicMock()) as mock:
        yield mock


@pytest.fixture
def mock_sys_font() -> Generator[MagicMock]:
    """Mock pygame.font.SysFont function."""
    with patch("pygame.font.SysFont", return_value=MagicMock()) as mock:
        yield mock


@pytest.fixture
def mock_pygame_init() -> Generator[MagicMock]:
    """Mock pygame.init function."""
    with patch("pygame.init") as mock:
        yield mock


@pytest.fixture
def mock_pygame_draw_rect() -> Generator[MagicMock]:
    """Mock pygame.draw.rect function."""
    with patch("pygame.draw.rect") as mock:
        yield mock


@pytest.fixture
def configured_app(
    app: FlappyBirdApp, mock_display_set_mode: MagicMock, mock_sys_font: MagicMock, mock_flappy_bird_ga: MagicMock
) -> FlappyBirdApp:
    """Configured FlappyBirdApp with a mock GA."""
    app._configure()
    app._clock = MagicMock()

    # Mock GA
    mock_ga_instance = MagicMock()
    mock_ga_instance._lifetime = MOCK_LIFETIME
    mock_ga_instance._generation = 1
    mock_ga_instance.num_alive = 5
    mock_ga_instance._population._members = []
    mock_flappy_bird_ga.create.return_value = mock_ga_instance

    app.add_ga(
        MOCK_POPULATION_SIZE,
        MOCK_MUTATION_RATE,
        MOCK_LIFETIME,
        MOCK_BIRD_X,
        MOCK_BIRD_Y,
        MOCK_BIRD_SIZE,
        MOCK_HIDDEN_LAYER_SIZES,
        MOCK_WEIGHTS_RANGE,
        MOCK_BIAS_RANGE,
    )

    return app


class TestFlappyBirdApp:
    """Unit tests for the FlappyBirdApp class."""

    def test_initialization(self, app: FlappyBirdApp) -> None:
        """Test FlappyBirdApp initialization."""
        assert app._name == MOCK_NAME
        assert app._width == MOCK_WIDTH
        assert app._height == MOCK_HEIGHT
        assert app._fps == MOCK_FPS
        assert app._font == MOCK_FONT
        assert app._font_size == MOCK_FONT_SIZE
        assert app._game_counter == 0
        assert app._pipes == []
        assert app._current_pipes == 0
        assert app._pipe_counter == 0

    def test_create_game(
        self, mock_pygame_init: MagicMock, mock_display_set_mode: MagicMock, mock_sys_font: MagicMock
    ) -> None:
        """Test FlappyBirdApp.create_game class method."""
        app = FlappyBirdApp.create_game(
            name=MOCK_NAME,
            width=MOCK_WIDTH,
            height=MOCK_HEIGHT,
            fps=MOCK_FPS,
            font=MOCK_FONT,
            font_size=MOCK_FONT_SIZE,
        )

        mock_pygame_init.assert_called_once()
        mock_display_set_mode.assert_called_once_with((MOCK_WIDTH, MOCK_HEIGHT))
        mock_sys_font.assert_called_once_with(MOCK_FONT, MOCK_FONT_SIZE)

        assert isinstance(app, FlappyBirdApp)
        assert app._name == MOCK_NAME
        assert app._width == MOCK_WIDTH
        assert app._height == MOCK_HEIGHT

    def test_max_count_property(self, configured_app: FlappyBirdApp) -> None:
        """Test max_count property."""
        expected_max_count = MOCK_LIFETIME * MOCK_FPS
        assert configured_app.max_count == expected_max_count

    def test_closest_pipe_no_pipes(self, configured_app: FlappyBirdApp) -> None:
        """Test closest_pipe property when there are no pipes."""
        assert configured_app.closest_pipe is None

    def test_closest_pipe_with_pipes(self, configured_app: FlappyBirdApp, mock_pipe: MagicMock) -> None:
        """Test closest_pipe property with existing pipes."""
        # Create mock pipes
        mock_pipe1 = MagicMock()
        mock_pipe1._x = 200
        mock_pipe1.WIDTH = 50

        mock_pipe2 = MagicMock()
        mock_pipe2._x = 300
        mock_pipe2.WIDTH = 50

        mock_pipe3 = MagicMock()
        mock_pipe3._x = 50  # Behind the bird
        mock_pipe3.WIDTH = 50

        configured_app._pipes = [mock_pipe1, mock_pipe2, mock_pipe3]

        closest = configured_app.closest_pipe
        assert closest == mock_pipe1  # Closest pipe in front

    def test_closest_pipe_all_behind(self, configured_app: FlappyBirdApp) -> None:
        """Test closest_pipe property when all pipes are behind the bird."""
        # Create mock pipes all behind the bird
        mock_pipe1 = MagicMock()
        mock_pipe1._x = 20
        mock_pipe1.WIDTH = 50

        mock_pipe2 = MagicMock()
        mock_pipe2._x = 30
        mock_pipe2.WIDTH = 50

        configured_app._pipes = [mock_pipe1, mock_pipe2]

        assert configured_app.closest_pipe is None

    def test_add_ga(self, app: FlappyBirdApp, mock_flappy_bird_ga: MagicMock) -> None:
        """Test add_ga method."""
        mock_ga_instance = MagicMock()
        mock_flappy_bird_ga.create.return_value = mock_ga_instance

        app.add_ga(
            MOCK_POPULATION_SIZE,
            MOCK_MUTATION_RATE,
            MOCK_LIFETIME,
            MOCK_BIRD_X,
            MOCK_BIRD_Y,
            MOCK_BIRD_SIZE,
            MOCK_HIDDEN_LAYER_SIZES,
            MOCK_WEIGHTS_RANGE,
            MOCK_BIAS_RANGE,
        )

        mock_flappy_bird_ga.create.assert_called_once_with(
            MOCK_POPULATION_SIZE,
            MOCK_MUTATION_RATE,
            MOCK_LIFETIME,
            MOCK_BIRD_X,
            MOCK_BIRD_Y,
            MOCK_WIDTH,
            MOCK_HEIGHT,
            MOCK_BIRD_SIZE,
            MOCK_HIDDEN_LAYER_SIZES,
            MOCK_WEIGHTS_RANGE,
            MOCK_BIAS_RANGE,
        )

        assert app._ga == mock_ga_instance
        assert app._bird_x == MOCK_BIRD_X

    def test_add_pipe(self, configured_app: FlappyBirdApp, mock_pipe: MagicMock) -> None:
        """Test _add_pipe method."""
        mock_speed = 5.0
        mock_pipe_instance = MagicMock()
        mock_pipe.return_value = mock_pipe_instance

        configured_app._add_pipe(mock_speed)

        mock_pipe.assert_called_once_with(MOCK_WIDTH, MOCK_HEIGHT, mock_speed)
        assert mock_pipe_instance in configured_app._pipes
        assert configured_app._current_pipes == 1

    def test_write_stats(self, configured_app: FlappyBirdApp) -> None:
        """Test _write_stats method."""
        configured_app.write_text = MagicMock()  # type: ignore[method-assign]
        configured_app._game_counter = 120  # 2 seconds at 60 FPS

        configured_app._write_stats()

        expected_calls = [
            f"Generation: {configured_app._ga._generation}",
            f"Birds alive: {configured_app._ga.num_alive}",
            f"Score: {int(configured_app._game_counter / configured_app._fps)}",
        ]

        calls = configured_app.write_text.call_args_list
        for i, expected_text in enumerate(expected_calls):
            assert calls[i][0][0] == expected_text

    def test_update_game_reset_max_count(self, configured_app: FlappyBirdApp, mock_pipe: MagicMock) -> None:
        """Test update method when max_count is reached."""
        configured_app._game_counter = configured_app.max_count
        configured_app._pipes = [MagicMock()]
        configured_app._current_pipes = 1
        configured_app._pipe_counter = 10

        # Mock methods
        configured_app._ga._analyse = MagicMock()
        configured_app._ga._evolve = MagicMock()
        configured_app._ga.reset = MagicMock()  # type: ignore[method-assign]
        configured_app._ga._evaluate = MagicMock()
        configured_app._write_stats = MagicMock()  # type: ignore[method-assign]

        configured_app.update()

        # Verify reset behavior
        configured_app._ga._analyse.assert_called_once()
        configured_app._ga._evolve.assert_called_once()
        configured_app._ga.reset.assert_called_once()
        assert configured_app._game_counter == 1  # Incremented after reset
        assert configured_app._pipes == []
        assert configured_app._current_pipes == 0
        assert configured_app._pipe_counter == 1

    def test_update_game_reset_no_alive(self, configured_app: FlappyBirdApp, mock_pygame_draw_rect: MagicMock) -> None:
        """Test update method when no birds are alive."""
        configured_app._ga.num_alive = 0  # type: ignore[misc]

        # Mock methods
        configured_app._ga._analyse = MagicMock()
        configured_app._ga._evolve = MagicMock()
        configured_app._ga.reset = MagicMock()  # type: ignore[method-assign]
        configured_app._ga._evaluate = MagicMock()
        configured_app._write_stats = MagicMock()  # type: ignore[method-assign]

        configured_app.update()

        # Verify reset behavior
        configured_app._ga._analyse.assert_called_once()
        configured_app._ga._evolve.assert_called_once()
        configured_app._ga.reset.assert_called_once()

    def test_update_normal_gameplay(
        self, configured_app: FlappyBirdApp, mock_pipe: MagicMock, mock_pygame_draw_rect: MagicMock
    ) -> None:
        """Test update method during normal gameplay."""
        # Setup normal game state
        start_counter = 50
        configured_app._game_counter = start_counter
        configured_app._ga.num_alive = 5  # type: ignore[misc]

        # Mock pipe spawning
        mock_pipe.get_spawn_time.return_value = 60
        mock_pipe.get_speed.return_value = 300

        # Mock existing pipes and birds with proper attributes
        mock_existing_pipe = MagicMock()
        configured_app._pipes = [mock_existing_pipe]

        mock_bird = MagicMock()
        configured_app._ga._population._members = [mock_bird]

        # Mock methods
        configured_app._ga._evaluate = MagicMock()
        configured_app._write_stats = MagicMock()  # type: ignore[method-assign]

        # Mock closest_pipe property
        mock_closest_pipe = MagicMock()
        with patch.object(FlappyBirdApp, "closest_pipe", new_callable=PropertyMock) as mock_closest_pipe_property:
            mock_closest_pipe_property.return_value = mock_closest_pipe

            configured_app.update()

            # Verify pipe and bird updates
            mock_existing_pipe.update.assert_called_once()
            mock_existing_pipe.draw.assert_called_once_with(configured_app.screen)

            mock_bird.update.assert_called_once_with(mock_closest_pipe)
            mock_bird.draw.assert_called_once_with(configured_app.screen)

            # Verify game progression
            configured_app._ga._evaluate.assert_called_once()
            configured_app._write_stats.assert_called_once()
            assert configured_app._game_counter == start_counter + 1
            assert configured_app._pipe_counter == 1

    def test_update_pipe_spawning(self, configured_app: FlappyBirdApp, mock_pipe: MagicMock) -> None:
        """Test pipe spawning in update method."""
        # Setup for pipe spawning
        configured_app._pipe_counter = 0
        configured_app._current_pipes = 0

        mock_pipe.get_spawn_time.return_value = 1  # Spawn every frame
        mock_pipe.get_speed.return_value = 300
        mock_pipe_instance = MagicMock()
        mock_pipe.return_value = mock_pipe_instance

        # Mock methods
        configured_app._ga._evaluate = MagicMock()
        configured_app._write_stats = MagicMock()  # type: ignore[method-assign]

        configured_app.update()

        # Verify pipe was spawned
        mock_pipe.assert_called_once_with(MOCK_WIDTH, MOCK_HEIGHT, 300 / MOCK_FPS)
        assert mock_pipe_instance in configured_app._pipes
        assert configured_app._current_pipes == 1
        assert configured_app._pipe_counter == 1
