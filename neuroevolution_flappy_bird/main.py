"""Main module to run the Flappy Bird neuroevolution application."""

import json

from neuroevolution_flappy_bird.flappy_bird_app import FlappyBirdApp

CONFIG_FILEPATH = "./config/config.json"


def run() -> None:
    """Run the Flappy Bird neuroevolution simulation."""
    with open(CONFIG_FILEPATH) as config_file:
        config = json.load(config_file)
    app_config = config["app"]
    ga_config = config["genetic_algorithm"]

    fba = FlappyBirdApp.create_game(
        name=app_config["name"],
        width=app_config["width"],
        height=app_config["height"],
        fps=app_config["fps"],
        font=app_config["font"],
        font_size=app_config["font_size"],
    )
    fba.add_ga(
        population_size=ga_config["population_size"],
        mutation_rate=ga_config["mutation_rate"],
        lifetime=ga_config["lifetime"],
        bird_x=ga_config["bird_x"],
        bird_y=ga_config["bird_y"],
        bird_size=ga_config["bird_size"],
        hidden_layer_sizes=ga_config["hidden_layer_sizes"],
        weights_range=ga_config["weights_range"],
        bias_range=ga_config["bias_range"],
    )
    fba.run()
