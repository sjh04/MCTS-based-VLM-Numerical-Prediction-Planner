import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import rl_utils


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, 
                 num_throttle_bins, num_steering_bins, # Added: parameters for action discretization
                 learning_rate, gamma,
                 epsilon, target_update, device):
        
        self.num_throttle_bins = num_throttle_bins
        self.num_steering_bins = num_steering_bins
        
        # Define the continuous action ranges and discretized bins
        self.throttle_values = np.linspace(-1.0, 1.0, num_throttle_bins)
        self.steering_values = np.linspace(-1.0, 1.0, num_steering_bins)
        
        self.action_dim = num_throttle_bins * num_steering_bins # Derived action_dim

        self.state_dim = state_dim  # 状态维度
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def get_closest_discrete_action(self, continuous_throttle, continuous_steering):
        """
        Converts a continuous (throttle, steering) pair to the closest discrete action index.
        """
        # Find the closest throttle value and its index
        throttle_diff = np.abs(self.throttle_values - continuous_throttle)
        throttle_idx = np.argmin(throttle_diff)

        # Find the closest steering value and its index
        steering_diff = np.abs(self.steering_values - continuous_steering)
        steering_idx = np.argmin(steering_diff)

        discrete_action_idx = throttle_idx * self.num_steering_bins + steering_idx
        return discrete_action_idx

    def take_action(self, state, llm_suggested_continuous_action=None, p_follow_llm=0.3):  # epsilon-贪婪策略采取动作
        # defualt llm suggested bias is None
        # llm_suggested_continuous_action should be a tuple (throttle, steering) or None
        # p_follow_llm is the probability of following the LLM's suggestion
        
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)
            print("state_tensor", state_tensor.shape)
            print("state_dim", self.state_dim)
            action = self.q_net(state_tensor)
            
        return action

    def get_continuous_action_pair(self, discrete_action_idx):
        """
        Converts a discrete action index into a continuous (throttle, steering) pair.
        """
        if not (0 <= discrete_action_idx < self.action_dim):
            raise ValueError(f"discrete_action_idx {discrete_action_idx} is out of bounds for action_dim {self.action_dim}")

        throttle_idx = discrete_action_idx // self.num_steering_bins
        steering_idx = discrete_action_idx % self.num_steering_bins
        
        throttle = self.throttle_values[throttle_idx]
        steering = self.steering_values[steering_idx]
        
        return throttle, steering

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1