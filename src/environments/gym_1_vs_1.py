import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import logging
from src.agents.minecraft_agent import MinecraftAgent

logging.basicConfig(level=logging.INFO)

class MinecraftGym(gym.Env):
    def __init__(self, device=""):
        super(MinecraftGym, self).__init__()
        self.device = device
        self.agents = self.get_agents()  # Initialize agents
        self.action_space = spaces.MultiDiscrete([2] * 7)  # 7 binary discrete actions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )  # Observation space
        self.iters = 0

    def get_observations(self):
        try:
            observations = np.array([agent.get_model_input() for agent in self.agents])
            if observations.shape != (2, 8):
                raise ValueError(f"Invalid observation shape: {observations.shape}")
            return observations
        except Exception as e:
            logging.error(f"Error in getting observations: {e}")
            return np.zeros((2, 8))  # Return a default valid observation

    def get_agents(self):
        try:
            bot_1 = MinecraftAgent(server_ip="127.0.0.1", port=25565, username="bot1", enemy="bot2")
            time.sleep(5)  # Ensure the agent is ready
            bot_2 = MinecraftAgent(server_ip="127.0.0.1", port=25565, username="bot2", enemy="bot1")
            time.sleep(5)  # Ensure the agent is ready
            return bot_1, bot_2
        except Exception as e:
            logging.error(f"Error in initializing agents: {e}")
            return None, None

    def reset(self):
        try:
            for agent in self.agents:
                agent.bot.chat(f"/kill {agent.bot.username}")
            self.iters = 0
            # Return initial observation
            observations = self.get_observations()
            if observations.shape != (2, 8):
                raise ValueError(f"Invalid observation shape on reset: {observations.shape}")
            return observations, {}
        except Exception as e:
            logging.error(f"Error in reset: {e}")
            return np.zeros((2, 8)), {}

    def step(self, actions):
        self.iters += 1
        try:
            reward = 0
            for agent, action in zip(self.agents, actions):
                reward += agent.update_agent_state(action)
            observations = self.get_observations()
            if observations.shape != (2, 8):
                raise ValueError(f"Invalid observation shape in step: {observations.shape}")

            terminated = any([agent.attack_count > 3 for agent in self.agents])
            truncated = self.iters > 500

            return observations, reward, terminated, truncated, {}
        except Exception as e:
            logging.error(f"Error in step: {e}")
            return np.zeros((2, 8)), 0, False, False, {}

    def render(self, mode="human"):
        # Optionally implement rendering for visualization
        pass

    def close(self):
        try:
            for agent in self.agents:
                agent.bot.end()
                time.sleep(1)
        except Exception as e:
            logging.error(f"Error in close: {e}")

# Ensure the environment is registered with Gym
from gymnasium.envs.registration import register

register(
    id='MinecraftGym-v0',
    entry_point='path.to.this.module:MinecraftGym',
)
