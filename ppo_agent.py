import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Kiến trúc Actor (Stochastic Policy) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.mu_head = nn.Linear(300, action_dim)
        
        # Học độ lệch chuẩn (log_std) để tạo phân phối xác suất
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mu = self.max_action * torch.tanh(self.mu_head(a))
        return mu

    def get_action_and_log_prob(self, state, action=None):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mu = self.max_action * torch.tanh(self.mu_head(a))
        
        action_std = self.action_log_std.exp()
        dist = Normal(mu, action_std)

        if action is None:
            action = dist.sample()

        # Tính log probability cần thiết cho PPO
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        return action, action_log_prob, dist_entropy

# --- 2. Kiến trúc Critic (Value Function) ---
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        # PPO Critic chỉ đánh giá State V(s)
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state):
        v = F.relu(self.l1(state))
        v = F.relu(self.l2(v))
        v = self.l3(v)
        return v

# --- 3. Rollout Buffer (Bộ nhớ ngắn hạn) ---
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, state, action, logprob, reward, state_value, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.is_terminals.append(done)

# --- 4. Class PPO Chính ---
class PPO(object):
    def __init__(self, state_dim, action_dim, max_action, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.max_action = max_action
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            # Lấy Value của state hiện tại
            state_value = self.critic(state)
            # Lấy Action và Log Prob
            action, action_log_prob, _ = self.actor.get_action_and_log_prob(state)
            
        return action.cpu().numpy().flatten(), action_log_prob.cpu().item(), state_value.cpu().item()
    
    # Hàm hỗ trợ lưu vào buffer cho gọn code main
    def store_transition(self, state, action, logprob, reward, state_value, done):
        self.buffer.add(state, action, logprob, reward, state_value, done)

    def update(self):
        # Chuyển dữ liệu từ buffer sang Tensor
        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals
        old_states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(device)
        old_actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32).to(device)
        old_state_values = torch.tensor(np.array(self.buffer.state_values), dtype=torch.float32).to(device)

        # Tính toán Monte Carlo Rewards
        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)
            
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(device)
        
        # Chuẩn hóa rewards (Normalize) để training ổn định hơn
        rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-7)

        # Tính Advantage
        advantages = rewards_to_go.detach() - old_state_values.detach().squeeze()

        # Update Policy K lần
        for _ in range(self.K_epochs):
            # Đánh giá lại action cũ dưới policy mới
            _, logprobs, dist_entropy = self.actor.get_action_and_log_prob(old_states, old_actions)
            state_values = self.critic(old_states).squeeze()
            
            # Tính tỷ lệ Ratio (pi_new / pi_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Tính Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Loss Actor
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            
            # Loss Critic
            critic_loss = self.MseLoss(state_values, rewards_to_go)

            # Gradient Descent
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Xóa buffer sau khi update
        self.buffer.clear()