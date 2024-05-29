import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple
import gymnasium as gym
import logging

logging.basicConfig(level=logging.INFO)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def train_dqn(env, num_episodes, batch_size, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500, target_update=10, memory_capacity=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize networks
    policy_net = DQN(env.observation_space.shape[0], env.action_space.nvec.sum()).to(device)
    target_net = DQN(env.observation_space.shape[0], env.action_space.nvec.sum()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(memory_capacity)

    epsilon = epsilon_start
    steps_done = 0

    def select_action(state):
        nonlocal epsilon, steps_done
        sample = random.random()
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
        steps_done += 1
        if sample > epsilon:
            with torch.no_grad():
                return policy_net(state).view(-1, 7).argmax(dim=1).view(1, -1)
        else:
            return torch.tensor([random.choices(range(n), k=7) for n in env.action_space.nvec], device=device, dtype=torch.long).view(1, -1)

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
        batch = memory.Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = policy_net(state_batch).view(-1, 7).gather(1, action_batch)

        next_state_values = target_net(next_state_batch).view(-1, 7).max(1)[0].detach()
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for episode in range(num_episodes):
        state, _ = env.reset()
        if state is None or state.shape != (2, 8):
            logging.error(f"Skipping episode {episode} due to failed reset or invalid state shape.")
            continue
        state = torch.tensor(state, device=device, dtype=torch.float32)
        for t in range(1000):
            action = select_action(state)
            next_state, reward, done, truncated, _ = env.step(action.view(-1).cpu().numpy())
            if next_state is None or next_state.shape != (2, 8):
                logging.error(f"Skipping step {t} in episode {episode} due to invalid next state.")
                break
            reward = torch.tensor([reward], device=device)

            next_state = torch.tensor(next_state, device=device, dtype=torch.float32)

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()

            if done or truncated:
                break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()
