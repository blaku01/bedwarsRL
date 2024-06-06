import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple
from loguru import logger

# Configure loguru to log into a file
logger.add("dqn_training.log", rotation="500 MB", level="DEBUG")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Ensure outputs are in range [0, 1]
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        logger.info(f"Initialized ReplayMemory with capacity {capacity}")

    def push(self, *args):
        self.memory.append(self.Transition(*args))
        logger.debug(f"Pushed to memory: {args}")

    def sample(self, batch_size):
        logger.info(f"Sampling {batch_size} transitions from memory")
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def train_dqn(env, num_episodes, batch_size, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500,
              target_update=10, memory_capacity=1000000):
    logger.info("Starting DQN training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize networks
    policy_net = DQN(env.observation_space.shape[0], 10).to(device)
    target_net = DQN(env.observation_space.shape[0], 10).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    logger.info("Initialized policy and target networks")

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(memory_capacity)

    epsilon = epsilon_start
    steps_done = 0

    def select_action(state):
        nonlocal epsilon, steps_done
        sample = random.random()
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
        steps_done += 1
        logger.debug(f"Selecting action with epsilon {epsilon} at step {steps_done}")

        if sample > epsilon:
            with torch.no_grad():
                policy_output = policy_net(state)[0]
                logger.info(f"policy net output: {policy_output}")

                action_booleans = (policy_output[:7] > 0.5).int()  # First 7 values as booleans
                action_floats = policy_output[7:9]  # Next 2 values as floats
                action_last_boolean = (policy_output[9] > 0.5).int()  # Last value as boolean

                action = torch.cat((action_booleans, action_floats, action_last_boolean.unsqueeze(0)))
                logger.debug(f"Selected action from policy network: {action}")
                return action
        else:
            action_booleans = torch.tensor(random.choices([0, 1], k=7), device=device, dtype=torch.int)
            action_floats = torch.tensor([random.uniform(0.0, 1.0) for _ in range(2)], device=device, dtype=torch.float)
            action_last_boolean = torch.tensor(random.choice([0, 1]), device=device, dtype=torch.int)

            action = torch.cat((action_booleans, action_floats, action_last_boolean.unsqueeze(0)))
            logger.debug(f"Selected random action: {action}")
            return action

    def optimize_model():
        if len(memory) < batch_size:
            logger.info("Not enough memory to sample a batch")
            return
        logger.info("Optimizing model")
        transitions = memory.sample(batch_size)
        batch = memory.Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.stack(batch.action).to(device)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        logger.info(f"action_batch shape: {action_batch.shape}")
        logger.info(f"state_batch shape: {state_batch.shape}")
        logger.info(f"policy_net(state_batch) shape: {policy_net(state_batch).shape}")
        # Ensure action_batch is reshaped correctly for gather
        if action_batch.dim() == 1:
            action_batch = action_batch.unsqueeze(1)

        # Note: Ensure action_batch has appropriate indices for gather
        state_action_values = policy_net(state_batch).gather(1, action_batch.to(dtype=torch.int64))

        next_state_values = target_net(next_state_batch).detach()
        logger.info(f"next_state_values shape: {next_state_values.shape}")
        expected_state_action_values = (next_state_values.T * gamma) + reward_batch

        logger.info(f"state_action_values shape: {state_action_values.shape}")
        logger.info(f"expected_state_action_values shape: {expected_state_action_values.shape}")
        loss = nn.MSELoss()(state_action_values.T, expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info(f"Model optimized with loss {loss.item()}")

    for episode in range(num_episodes):
        state, _ = env.reset()
        if state is None or state.shape != (8,):
            logger.error(f"Skipping episode {episode} due to failed reset or invalid state shape.")
            continue
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        logger.info(f"Starting episode {episode}")

        for t in range(1000):
            action = select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            if next_state is None or next_state.shape != (8,):
                logger.error(f"Skipping step {t} in episode {episode} due to invalid next state.")
                break
            reward = torch.tensor([reward], device=device)

            next_state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            logger.debug(f"Transition added to memory: state={state}, action={action}, next_state={next_state}, reward={reward}")

            state = next_state

            optimize_model()

            if done or truncated:
                logger.info(f"Episode {episode} finished after {t+1} steps")
                break

        if episode % target_update == 0:
            logger.debug("starting to load state dict")
            target_net.load_state_dict(policy_net.state_dict())
            logger.info(f"Updated target network at episode {episode}")

    env.close()
    logger.info("Training completed")
