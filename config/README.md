# App Configuration

- `app`: Pygame application settings
  - `name` (str): Name of window
  - `width` (int): Width of window
  - `height` (int): Height of window
  - `fps` (int): App FPS
  - `font` (str): Font style
  - `font_size` (int): Font size
- `genetic_algorithm`: Training parameters
  - `population_size` (int): Number of Birds in population
  - `mutation_rate` (float): Mutation rate for Birds
  - `lifetime` (int): Time of each generation in seconds
  - `bird_x` (int): x coordinate of Bird's start position
  - `bird_y` (int): y coordinate of Bird's start position
  - `bird_size` (int): Size of Bird
  - `hidden_layer_sizes` (list[int]): Neural network hidden layer sizes
  - `weights_range` (list[float]): Range for random weights
  - `bias_range` (list[float]): Range for random bias
