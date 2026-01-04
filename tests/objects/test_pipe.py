"""Unit tests for the neuroevolution_flappy_bird.objects.pipe module."""

from unittest.mock import call, patch

import pytest

from neuroevolution_flappy_bird.objects.pipe import Pipe

MOCK_WIDTH = 1000
MOCK_HEIGHT = 1000
MOCK_SPEED = 5


@pytest.fixture
def pipe() -> Pipe:
    """Mock Pipe instance."""
    return Pipe(MOCK_WIDTH, MOCK_HEIGHT, MOCK_SPEED)


class TestPipe:
    """Unit tests for the Pipe class."""

    def test_initialization(self, pipe: Pipe) -> None:
        """Test Pipe initialization."""
        assert pipe._x == MOCK_WIDTH
        assert pipe.SPACING < pipe._top_height < (MOCK_HEIGHT - (2 * pipe.SPACING))
        assert pipe._bottom_height == MOCK_HEIGHT - pipe._top_height + pipe.SPACING
        assert pipe._speed == MOCK_SPEED

    def test_rects(self, pipe: Pipe) -> None:
        """Test rects property."""
        rects = pipe.rects
        expected_num_rects = 2
        assert len(rects) == expected_num_rects
        assert rects[0].width == Pipe.WIDTH
        assert rects[1].width == Pipe.WIDTH
        assert rects[0].height == int(pipe._top_height)
        assert rects[1].height == int(pipe._bottom_height)

    def test_top_pos(self, pipe: Pipe) -> None:
        """Test top_pos property."""
        top_pos = pipe.top_pos
        assert top_pos[0] == pipe._x
        assert top_pos[1] == 0

    def test_bottom_pos(self, pipe: Pipe) -> None:
        """Test bottom_pos property."""
        bottom_pos = pipe.bottom_pos
        assert bottom_pos[0] == pipe._x
        assert bottom_pos[1] == pipe._top_height + Pipe.SPACING

    def test_offscreen(self, pipe: Pipe) -> None:
        """Test offscreen property."""
        assert not pipe.offscreen
        pipe._x = -Pipe.WIDTH
        assert pipe.offscreen

    def test_normalised_speed(self, pipe: Pipe) -> None:
        """Test normalised_speed property."""
        assert pipe.normalised_speed == MOCK_SPEED / Pipe.MAX_SPEED

    def test_draw(self, pipe: Pipe) -> None:
        """Test draw method."""
        with patch("pygame.draw.rect") as mock_draw:
            mock_screen = patch("pygame.Surface").start()
            pipe.draw(mock_screen)
            mock_draw.assert_has_calls(
                [
                    call(mock_screen, Pipe.COLOUR, pipe.rects[0]),
                    call(mock_screen, Pipe.COLOUR, pipe.rects[1]),
                ]
            )

    def test_draw_offscreen(self, pipe: Pipe) -> None:
        """Test draw method when Pipe is offscreen."""
        with patch("pygame.draw.rect") as mock_draw:
            mock_screen = patch("pygame.Surface").start()
            pipe._x = -Pipe.WIDTH
            pipe.draw(mock_screen)
            mock_draw.assert_not_called()

    def test_update(self, pipe: Pipe) -> None:
        """Test update method."""
        initial_x = pipe._x
        pipe.update()
        assert pipe._x == initial_x - pipe._speed

    def test_update_offscreen(self, pipe: Pipe) -> None:
        """Test update method when Pipe is offscreen."""
        pipe._x = -Pipe.WIDTH
        initial_x = pipe._x
        pipe.update()
        assert pipe._x == initial_x

    def test_get_speed(self) -> None:
        """Test get_speed static method."""
        pipes_spawned = 10
        expected_speed = min(Pipe.START_SPEED + (pipes_spawned * Pipe.ACC_SPEED), Pipe.MAX_SPEED)
        assert Pipe.get_speed(pipes_spawned) == expected_speed

    def test_get_spawn_time(self) -> None:
        """Test get_spawn_time static method."""
        pipes_spawned = 10
        expected_time = Pipe.START_SPAWNTIME - (pipes_spawned * Pipe.ACC_SPAWNTIME)
        assert Pipe.get_spawn_time(pipes_spawned) == expected_time
