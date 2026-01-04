"""Unit tests for the neuroevolution_flappy_bird.    module."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from neuroevolution_flappy_bird.pg.app import App

MOCK_NAME = "Test App"
MOCK_WIDTH = 800
MOCK_HEIGHT = 600
MOCK_FPS = 60
MOCK_FONT = "Arial"
MOCK_FONT_SIZE = 20


@pytest.fixture
def app() -> App:
    """Mock FlappyBirdApp instance."""
    return App(
        name=MOCK_NAME,
        width=MOCK_WIDTH,
        height=MOCK_HEIGHT,
        fps=MOCK_FPS,
        font=MOCK_FONT,
        font_size=MOCK_FONT_SIZE,
    )


@pytest.fixture
def mock_display_set_mode() -> Generator[MagicMock]:
    """Mock pygame.display.set_mode function."""
    with patch("pygame.display.set_mode", return_value=MagicMock()) as mock:
        yield mock


@pytest.fixture
def mock_display_set_caption() -> Generator[MagicMock]:
    """Mock pygame.display.set_caption function."""
    with patch("pygame.display.set_caption") as mock:
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
def configured_app(app: App, mock_display_set_mode: MagicMock, mock_sys_font: MagicMock) -> App:
    """Configured App instance."""
    app._configure()
    app._clock = MagicMock()
    return app


class TestApp:
    """Unit tests for the App class."""

    def test_initialization(self, app: App) -> None:
        """Test App initialization."""
        assert app._name == MOCK_NAME
        assert app._width == MOCK_WIDTH
        assert app._height == MOCK_HEIGHT
        assert app._fps == MOCK_FPS
        assert app._font == MOCK_FONT
        assert app._font_size == MOCK_FONT_SIZE
        assert app._running is False

    def test_create_app(
        self, mock_pygame_init: MagicMock, mock_display_set_mode: MagicMock, mock_sys_font: MagicMock
    ) -> None:
        """Test App.create_app class method."""
        app = App.create_app(
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

        assert app._name == MOCK_NAME
        assert app._width == MOCK_WIDTH
        assert app._height == MOCK_HEIGHT
        assert app._fps == MOCK_FPS
        assert app._font == MOCK_FONT
        assert app._font_size == MOCK_FONT_SIZE

    def test_screen_property(self, configured_app: App) -> None:
        """Test screen property."""
        assert configured_app.screen is configured_app._display_surf

    def test_configure(
        self, app: App, mock_display_set_mode: MagicMock, mock_display_set_caption: MagicMock, mock_sys_font: MagicMock
    ) -> None:
        """Test _configure method."""
        app._configure()

        mock_display_set_caption.assert_called_once_with(MOCK_NAME)
        mock_display_set_mode.assert_called_once_with((MOCK_WIDTH, MOCK_HEIGHT))
        mock_sys_font.assert_called_once_with(MOCK_FONT, MOCK_FONT_SIZE)

    def test_write_text(self, configured_app: App) -> None:
        """Test write_text method."""
        mock_text = "Test Text"
        mock_x = 100
        mock_y = 100
        mock_rendered_text = MagicMock()

        configured_app._pg_font.render = MagicMock(return_value=mock_rendered_text)  # type: ignore[method-assign]
        configured_app._display_surf.blit = MagicMock()  # type: ignore[method-assign]

        configured_app.write_text(mock_text, mock_x, mock_y)

        configured_app._pg_font.render.assert_called_once_with(mock_text, 1, (255, 255, 255))
        configured_app._display_surf.blit.assert_called_once_with(mock_rendered_text, (mock_x, mock_y))

    def test_update(self, configured_app: App) -> None:
        """Test update method."""
        configured_app.write_text = MagicMock()  # type: ignore[method-assign]
        configured_app._clock = MagicMock()
        configured_app._clock.get_fps.return_value = MOCK_FPS

        configured_app.update()

        # Check that write_text was called for all the expected text items
        expected_calls = [
            "App info:",
            f"Name: {MOCK_NAME}",
            f"Width: {MOCK_WIDTH}",
            f"Height: {MOCK_HEIGHT}",
            f"Font: {MOCK_FONT}",
            f"Font size: {MOCK_FONT_SIZE}",
            f"FPS: {MOCK_FPS}",
        ]

        calls = configured_app.write_text.call_args_list
        for i, call in enumerate(expected_calls):
            assert calls[i][0][0] == call
