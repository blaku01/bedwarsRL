from collections import OrderedDict

import numpy as np
from javascript import require

from src.utils.helpers import (
    calculate_distance_and_angle,
    distance_to_all_rectangle_walls,
)

mineflayer = require("mineflayer")
corner_1 = (-8, 84)
corner_2 = (7, 66)


class MinecraftAgent:
    def __init__(self, server_ip="localhost", port=25565, username="bot1", enemy="bot2"):
        self.control_state = OrderedDict(
            [
                ("forward", False),
                ("back", False),
                ("left", False),
                ("right", False),
                ("jump", False),
                ("sprint", False),
                ("sneak", False),
                ("swing", False),
            ]
        )
        self.enemy = enemy
        self.bot = mineflayer.createBot({"host": server_ip, "port": port, "username": username})

    def set_control_state(self, control_state_array):
        """Set the control state of the bot based on the provided array of boolean values.

        Args:
            control_state_array (list): A list of 7 boolean values representing the control states.
                The order of the states should follow: ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak', 'swing'].
        Returns:
            None
        """
        self.bot.look(control_state_array[8], control_state_array[9], False)  # rotate head
        for i, (key, value) in enumerate(self.control_state.items()):
            if value != control_state_array[i]:
                if key == "swing":
                    entity_on_cursor = self.bot.entityAtCursor()
                    if entity_on_cursor is not None:
                        self.bot.attack(entity_on_cursor)
                else:
                    self.control_state[key] = control_state_array[i]
                    self.bot.controlState[key] = control_state_array[i]

    def get_model_input(self):
        """Prepares the model input by gathering information about the enemy and bot's
        surroundings.

        Returns:
            A numpy array containing:
                - Distance to the enemy (float)
                - Angle to the enemy (float)
                - Distances to the four walls (4x float)
                - Bot's head rotation (2x float, float)

        **Example:**

        ```python
        model_input = minecraft_agent.get_model_input()
        print(model_input)

        # array([ 1.234,  56.789,  0.987,  2.345, 21.987, 7.345, 10.0, -20.0])
        ```
        """

        # Find enemy positions (excluding self)
        enemy_positions = []
        for entity_name, entity_data in self.bot.entities.valueOf().items():
            if (
                entity_data.get("type") == "player"
                and entity_data.get("username") != self.bot.username
                and entity_data.get("username") == self.enemy
            ):
                enemy_positions.append(
                    (entity_data["position"]["x"], entity_data["position"]["z"])
                )

        bot_position = [self.bot.entity.position["x"], self.bot.entity.position["z"]]

        distance_angle = np.array(calculate_distance_and_angle(bot_position, enemy_positions[0]))

        distance_to_walls = distance_to_all_rectangle_walls(bot_position, corner_1, corner_2)

        head_rotation = np.array([self.bot.entity.yaw, self.bot.entity.pitch])

        model_input = np.concatenate((distance_angle, distance_to_walls.flatten(), head_rotation))
        return model_input
