import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
from src.agents.minecraft_agent import MinecraftAgent

class MinecraftGym(gym.Env):
    def __init__(self, device=""):
        super(MinecraftGym, self).__init__()
        self.agents = self.get_agents()  # Create an instance of MinecraftAgent
        self.action_space = spaces.MultiBinary(7)  # 7 discrete actions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )  # Observation space
        self.iters = 0
        self.device = device

    def get_observations(self):
        return [agent.get_model_input() for agent in self.agents]

    def get_agents(self):
        bot_1 = MinecraftAgent(server_ip="26.70.149.205", port=25565, username="bot1", enemy="bot2")
        time.sleep(5)
        bot_2 = MinecraftAgent(server_ip="26.70.149.205", port=25565, username="bot2", enemy="bot1")
        return bot_1, bot_2

    def reset(self):
        for agent in self.agents:
            agent.bot.chat(f"/kill {agent.bot.username}")
        # Return initial observation
        return self.get_observations(), {}

    def _reset(self):
        self.reset()

    def step(self, actions):
        self.iters += 1
        # Execute one time step within the environment
        # Apply action on the agent
        reward = 0
        for agent, action in zip(self.agents, actions):
            reward += agent.update_agent_state(action)
        # Get observation after the action
        observations = self.get_observations()

        terminated = any([agent.attack_count > 3 for agent in self.agents])
        # Return observation, reward, done, info

        truncated = self.iters > 500

        return observations, reward, terminated, truncated, {}

    def render(self, mode="human"):
        # Render the environment to the screen
        pass

    def close(self):
        for agent in self.agents:
            agent.bot.end()
            time.sleep(5)
