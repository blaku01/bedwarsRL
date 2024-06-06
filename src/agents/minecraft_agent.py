from collections import OrderedDict
import numpy as np
from javascript import require
import threading
from loguru import logger

from src.utils.helpers import (
    calculate_distance,
    calculate_distance_and_angle,
    distance_to_all_rectangle_walls,
)

mineflayer = require("mineflayer")
corner_1 = (-8, 84)
corner_2 = (7, 66)

logger.add("minecraft_agent.log", rotation="500 MB", level="DEBUG")


class MinecraftAgent:
    def __init__(self, server_ip="localhost", port=25565, username="bot1", enemy="bot2"):
        logger.debug(f"Initializing MinecraftAgent with username {username} and enemy {enemy}")
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
        self.bot = mineflayer.createBot({"host": server_ip, "port": port, "username": username, "checkTimeoutInterval": 60 * 100})
        self.previous_distance = None
        self.attack_count = 0
        self.enemy_bot = None
        logger.info("MinecraftAgent initialized")

    def set_control_state(self, control_state_array: list) -> None:
        """Set the control state of the bot based on the provided array of boolean values."""
        logger.debug(f"Setting control state: {control_state_array}")
        for i, (key, value) in enumerate(self.control_state.items()):
            if value != control_state_array[i]:
                self.control_state[key] = control_state_array[i]
                self.bot.controlState[key] = control_state_array[i]
        logger.info(f"Control state set to: {self.control_state}")

    def swing(self) -> bool:
        """Attempts to attack the entity currently under the bot's cursor."""
        logger.debug("Attempting to swing")
        entity_on_cursor = self.bot.entityAtCursor()
        if entity_on_cursor is not None:
            self.bot.attack(entity_on_cursor)
            logger.info("Swing successful")
            return True
        logger.info("Swing failed, no entity on cursor")
        return False

    def look_around(self, pitch: float, yaw: float) -> None:
        """Orients the bot's viewpoint in the Minecraft world."""
        logger.debug(f"Looking around with pitch: {pitch}, yaw: {yaw}")
        self.bot.look(pitch, yaw, True)
        logger.info("Look around executed")

    def get_enemy(self):
        logger.debug("Getting enemy")
        for entity_name, entity_data in self.bot.entities.valueOf().items():
            logger.debug(f"entity_name: {entity_name},type: {entity_data.get('type')}")
            if (
                entity_data.get("type") == "player"
                and entity_data.get("username") != self.bot.username
                and entity_data.get("username") == self.enemy
            ):
                logger.info(f"Enemy found: {entity_data}")
                return entity_data
        logger.info("Enemy not found")
        return None

    def update_agent_state(self, state: list) -> float:
        """Updates the bot's control state and orientation based on a provided state list."""
        logger.debug(f"Updating agent state: {state}")
        enemy = self.enemy_bot.entity

        enemy_x = enemy["position"]["x"]
        enemy_z = enemy["position"]["z"]

        if self.previous_distance is None:
            bot_pos = self.bot.entity["position"]
            self.previous_distance = calculate_distance(enemy_x, enemy_z, bot_pos["x"], bot_pos["z"])
            logger.info(f"Initial distance to enemy set: {self.previous_distance}")

        control_state = state[:7]
        self.set_control_state(control_state)

        self.look_around(state[7], state[8])

        if state[9]:
            attacked = self.swing()
            self.attack_count += 1
        else:
            attacked = 0

        bot_pos = self.bot.entity["position"]
        current_distance = calculate_distance(enemy_x, enemy_z, bot_pos["x"], bot_pos["z"])
        delta_distance = current_distance - self.previous_distance
        self.previous_distance = current_distance
        logger.debug(f"Distance to enemy updated: {current_distance}")

        q_value = float(attacked) + 1 / 10 * delta_distance
        logger.info(f"Agent state updated, Q value: {q_value}")
        return q_value

    def get_model_input(self) -> np.ndarray:
        """Prepares the model input by gathering information about the enemy and bot's surroundings."""
        logger.debug("Getting model input")

        bot_position = [self.bot.entity.position["x"], self.bot.entity.position["z"]]
        enemy_position = [self.enemy_bot.entity["position"]["x"], self.enemy_bot.entity["position"]["z"]]
        distance_angle = np.array(calculate_distance_and_angle(bot_position, enemy_position))
        distance_to_walls = distance_to_all_rectangle_walls(bot_position, corner_1, corner_2)
        head_rotation = np.array([self.bot.entity.yaw, self.bot.entity.pitch])  # 2
        model_input = np.concatenate((distance_angle, distance_to_walls.flatten(), head_rotation))
        logger.info(f"Model input: {model_input}")
        return model_input


