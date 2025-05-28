from __future__ import annotations

from typing import cast

from flappy_bird.ga.bird_ga import FlappyBirdGA
from flappy_bird.objects.bird import Bird
from flappy_bird.objects.pipe import Pipe
from flappy_bird.pg.app import App


class FlappyBirdApp(App):
    """
    This class creates a version of Flappy Bird and uses neuroevolution to train AI to play the game.
    """

    def __init__(self, name: str, width: int, height: int, fps: int, font: str, font_size: int) -> None:
        """
        Initialise FlappyBirdApp.

        Parameters:
            name (str): App name
            width (int): Screen width
            height (int): Screen height
            fps (int): Game FPS
            font (str): Font style
            font_size (int): Font size
        """
        super().__init__(name, width, height, fps, font, font_size)
        self._ga: FlappyBirdGA
        self._game_counter = 0
        self._pipes: list[Pipe] = []
        self._current_pipes = 0
        self._pipe_counter = 0
        self._bird_x: int

    @property
    def max_count(self) -> int:
        return self._ga._lifetime * self._fps

    @property
    def closest_pipe(self) -> Pipe | None:
        """
        Determine which Pipe is closest to and in front of the Birds.

        Returns:
            closest (Pipe): Pipe closest to the Birds
        """
        _dist = float(self._width)
        closest = None

        for _pipe in self._pipes:
            pipe_dist = _pipe._x + _pipe.WIDTH - self._bird_x
            if 0 < pipe_dist < _dist:
                _dist = pipe_dist
                closest = _pipe

        return closest

    @classmethod
    def create_game(cls, name: str, width: int, height: int, fps: int, font: str, font_size: int) -> FlappyBirdApp:
        """
        Create App and configure limits for Bird and genetic algorithm.

        Parameters:
            name (str): Application name
            width (int): Screen width
            height (int): Screen height
            fps (int): Application FPS
            font (str): Font style
            font_size (int): Font size

        Returns:
            fba (FlappyBirdApp): Flappy Bird application
        """
        Bird.X_LIM = width
        Bird.Y_LIM = height
        Pipe.X_LIM = width
        Pipe.Y_LIM = height
        return cast(FlappyBirdApp, super().create_app(name, width, height, fps, font, font_size))

    def _write_stats(self) -> None:
        """
        Write algorithm statistics to screen.
        """
        _start_x = 20
        _start_y = 30
        self.write_text(f"Generation: {self._ga._generation}", _start_x, _start_y)
        self.write_text(f"Birds alive: {self._ga.num_alive}", _start_x, _start_y * 3)
        self.write_text(f"Score: {int(self._game_counter / self._fps)}", _start_x, _start_y * 4)

    def _add_pipe(self, speed: float) -> None:
        """
        Spawn a new Pipe with a given speed.

        Parameters:
            speed (float): Pipe speed
        """
        self._pipes.append(Pipe(speed))
        self._current_pipes += 1

    def add_ga(
        self,
        population_size: int,
        mutation_rate: float,
        lifetime: int,
        bird_x: int,
        bird_y: int,
        bird_size: int,
        hidden_layer_sizes: list[int],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
    ) -> None:
        """
        Add genetic algorithm to app.

        Parameters:
            population_size (int): Number of members in population
            mutation_rate (float): Mutation rate for members
            lifetime (int): Time of each generation in seconds
            bird_x (int): x coordinate of Bird's start position
            bird_y (int): y coordinate of Bird's start position
            bird_size (int): Size of Bird
            hidden_layer_sizes (list[int]): Neural network hidden layer sizes
            weights_range (tuple[float, float]): Range for random weights
            bias_range (tuple[float, float]): Range for random bias
        """
        self._bird_x = bird_x
        self._ga = FlappyBirdGA.create(
            population_size,
            mutation_rate,
            lifetime,
            bird_x,
            bird_y,
            bird_size,
            hidden_layer_sizes,
            weights_range,
            bias_range,
        )

    def update(self) -> None:
        """
        Run genetic algorithm, update Birds and draw to screen.
        """
        if self._game_counter == self.max_count or self._ga.num_alive == 0:
            self._ga._analyse()
            self._ga._evolve()
            self._ga.reset()
            self._game_counter = 0
            self._pipes = []
            self._current_pipes = 0
            self._pipe_counter = 0

        _next_pipe_spawntime = Pipe.get_spawn_time(self._current_pipes)
        _next_pipe_speed = Pipe.get_speed(self._current_pipes) / self._fps
        if int(self._pipe_counter) % _next_pipe_spawntime == 0:
            self._add_pipe(_next_pipe_speed)
            self._pipe_counter = 0

        for _pipe in self._pipes:
            _pipe.update()
            _pipe.draw(self.screen)

        for _bird in self._ga._population._members:
            _bird.update(self.closest_pipe)
            _bird.draw(self.screen)

        self._ga._evaluate()
        self._game_counter += 1
        self._pipe_counter += 1
        self._write_stats()
