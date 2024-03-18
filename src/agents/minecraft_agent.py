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
            ]
        )
        self.enemy = enemy
        self.bot = mineflayer.createBot({"host": server_ip, "port": port, "username": username})

    def set_control_state(self, control_state_array: list) -> None:
        """Set the control state of the bot based on the provided array of boolean values.

        Args:
            control_state_array (list): A list of 7 boolean values representing the control states.
                The order of the states should follow: ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak'].
        Returns:
            None
        """
        for i, (key, value) in enumerate(self.control_state.items()):
            if value != control_state_array[i]:
                self.control_state[key] = control_state_array[i]
                self.bot.controlState[key] = control_state_array[i]

    def swing(self) -> bool:
        """Attempts to attack the entity currently under the bot's cursor.

        Args: None
        Returns:
            bool: True if an attack was performed, False otherwise.
        """
        entity_on_cursor = self.bot.entityAtCursor()
        if entity_on_cursor is not None:
            self.bot.attack(entity_on_cursor)
            return True
        return False

    def look_around(self, pitch: float, yaw: float) -> None:
        """Orients the bot's viewpoint in the Minecraft world.

        This function takes two arguments, pitch and yaw, which represent the desired rotation

        Args:
            pitch (float): The rotation around the X axis (up/down).
            yaw (float): The rotation around the Y axis (left/right).
        Returns:
            None
        """
        self.bot.look(pitch, yaw, True)

    def update_agent_state(self, state: list) -> None:
        """Updates the bot's control state and orientation based on a provided state list.

        This function takes a list containing the desired state of the agent. The array should
        have exactly 10 elements in the following order:

        - The first 7 elements are booleans representing the bot's movement controls:
            - state[0] (bool): Move forward
            - state[1] (bool): Move backward
            - state[2] (bool): Move left
            - state[3] (bool): Move right
            - state[4] (bool): Jump
            - state[5] (bool): Sprint
            - state[6] (bool): Sneak
        - The next 2 elements are floats representing the desired rotation of the bot's head:
            - state[7] (float): Pitch (rotation around the X axis, up/down)
            - state[8] (float): Yaw (rotation around the Y axis, left/right)
        - The last element is a boolean representing the swing action
            - state[9] (bool): swing (attack or not)
        Args:
            state (list): A list representing the desired state of the agent (format: [bool, bool, ..., bool, float, float]).

        Returns:
            None
        """
        control_state = state[:7]
        self.set_control_state(control_state)

        self.look_around(state[7], state[8])

        if state[9]:
            self.swing()

    def get_model_input(self) -> np.ndarray:
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

        distance_angle = (
            np.array(calculate_distance_and_angle(bot_position, enemy_positions[0]))
            if enemy_positions
            else None
        )

        distance_to_walls = distance_to_all_rectangle_walls(bot_position, corner_1, corner_2)

        head_rotation = np.array([self.bot.entity.yaw, self.bot.entity.pitch])

        model_input = np.concatenate((distance_angle, distance_to_walls.flatten(), head_rotation))
        return model_input


# %%
