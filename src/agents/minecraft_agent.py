from collections import OrderedDict

from javascript import require

mineflayer = require("mineflayer")


class MinecraftAgent:
    def __init__(self, server_ip, port, username):
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
        self.bot = mineflayer.createBot({"host": server_ip, "port": port, "username": username})

    def setControlState(self, control_state_array):
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
