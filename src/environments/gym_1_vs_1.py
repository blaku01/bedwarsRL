import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
from src.agents.minecraft_agent import MinecraftAgent
import itertools
from loguru import logger

# Configure loguru to log into a file
logger.add("dqn_training.log", rotation="500 MB", level="DEBUG")

class MinecraftGym(gym.Env):
    def __init__(self, device=""):
        super(MinecraftGym, self).__init__()
        self.device = device
        self.agents = self.get_agents()  # Initialize agents
        self.action_space = spaces.Tuple((
            spaces.MultiDiscrete([2] * 7),  # 7 booleans
            spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32),  # 2 floats
            spaces.Discrete(2)  # 1 boolean
        ))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )  # Observation space
        self.iters = 0
        self.current_agent = None
        self.agents_iter = itertools.cycle(self.agents)
        self.current_time = 0
    def get_observation(self):
        try:
            observation = self.current_agent.get_model_input()
            if observation.shape != (8,):
                raise ValueError(f"Invalid observation shape: {observation.shape}")
            return observation
        except Exception as e:
            logger.error(f"Error in getting observations: {e}")
            return np.zeros(8)  # Return a default valid observation

    def get_agents(self):
        try:
            bot_1 = MinecraftAgent(server_ip="26.70.149.205", port=25565, username="bot1", enemy="bot2")
            time.sleep(5)  # Ensure the agent is ready
            bot_2 = MinecraftAgent(server_ip="26.70.149.205", port=25565, username="bot2", enemy="bot1")
            time.sleep(5)  # Ensure the agent is ready
            bot_1.enemy_bot = bot_2.bot
            bot_2.enemy_bot = bot_1.bot
            return bot_1, bot_2
        except Exception as e:
            logger.error(f"Error in initializing agents: {e}")
            return None, None

    def reset(self):
        try:
            for agent in self.agents:
                agent.bot.chat(f"/kill {agent.bot.username}")
            time.sleep(2)
            self.iters = 0
            self.current_agent = next(self.agents_iter)

            bot_time = self.current_agent.bot.time.age
            if bot_time == self.current_time:
                logger.error("bot bugged?")
                raise SystemExit(0)
                raise RuntimeError("bot bugged?")
                
            self.current_time = bot_time
            # Return initial observation
            observation = self.get_observation()
            if observation.shape != (8,):
                raise ValueError(f"Invalid observation shape on reset: {observation.shape}")
            return observation, {}
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            return np.zeros(8), {}

    def step(self, action):
        action = action.tolist()
        action[:7] = [bool(a) for a in action[:7]]
        action[9] = bool(action[9])
        logger.info(f"Action: {action}")
        self.iters += 1
        try:
            reward = 0
            reward += self.current_agent.update_agent_state(action)
            logger.debug(f"Updated agent state with action {action}, reward: {reward}")

            observation = self.get_observation()
            logger.debug(f"Obtained observation: {observation}")

            self.current_agent = next(self.agents_iter)
            logger.debug(f"Switched to next agent: {self.current_agent}")

            if observation.shape != (8,):
                raise ValueError(f"Invalid observation shape in step: {observation.shape}")

            terminated = 0 #any([agent.attack_count > 3 for agent in self.agents])
            truncated = 0
            logger.info(f"Step result - Terminated: {terminated}, Truncated: {truncated}")

            return observation, reward, terminated, truncated, {}
        except Exception as e:
            logger.error(f"Error in step: {e}")
            return np.zeros(8), 0, False, False, {}

    def render(self, mode="human"):
        # Optionally implement rendering for visualization
        pass

    def close(self):
        try:
            for agent in self.agents:
                agent.bot.end()
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error in close: {e}")

# Ensure the environment is registered with Gym
from gymnasium.envs.registration import register

register(
    id='MinecraftGym-v0',
    entry_point='path.to.this.module:MinecraftGym',
)
