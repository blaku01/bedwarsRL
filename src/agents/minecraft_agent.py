from javascript import require

mineflayer = require("mineflayer")


class MinecraftAgent:
    def __init__(self, server_ip, port, username):
        self.forward, self.back, self.left, self.right, self.jump, self.sprint, self.sneak = (
            False,
            False,
            False,
            False,
            False,
            False,
            False,
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
        states = ["forward", "back", "left", "right", "jump", "sprint", "sneak"]
        for x in range(len(control_state_array)):
            self.bot.controlState[states[x]] = control_state_array[x]
